"""Native compilation for maximum performance.

Automatically downloads Rust toolchain if needed, compiles IR to native binary.
Uses raw binary I/O (f64 arrays) instead of JSON for speed.
"""

import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

from .compiler import IR
from .codegen.rust import generate_rust

CACHE_DIR = Path.home() / ".cache" / "rac"
RUSTUP_URL = "https://sh.rustup.rs"


def _get_cargo() -> Path | None:
    """Find cargo binary."""
    cargo = shutil.which("cargo")
    if cargo:
        return Path(cargo)

    cargo_home = Path.home() / ".cargo" / "bin" / "cargo"
    if cargo_home.exists():
        return cargo_home

    return None


def _install_rust() -> Path:
    """Install Rust via rustup."""
    print("Installing Rust toolchain (one-time setup)...")

    if platform.system() == "Windows":
        import urllib.request
        rustup_init = CACHE_DIR / "rustup-init.exe"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve("https://win.rustup.rs/x86_64", rustup_init)
        subprocess.run([str(rustup_init), "-y", "--quiet"], check=True)
    else:
        subprocess.run(
            ["sh", "-c", f"curl --proto '=https' --tlsv1.2 -sSf {RUSTUP_URL} | sh -s -- -y --quiet"],
            check=True,
            capture_output=True
        )

    cargo = Path.home() / ".cargo" / "bin" / "cargo"
    if not cargo.exists():
        raise RuntimeError("Failed to install Rust")

    print("Rust installed successfully")
    return cargo


def ensure_cargo() -> Path:
    """Ensure cargo is available, installing if needed."""
    cargo = _get_cargo()
    if cargo:
        return cargo
    return _install_rust()


def _ir_hash(ir: IR) -> str:
    """Hash IR for caching."""
    data = json.dumps({
        "order": ir.order,
        "vars": {k: str(v.expr) for k, v in ir.variables.items()}
    }, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:16]


class CompiledBinary:
    """A compiled RAC binary for maximum performance."""

    def __init__(self, binary_path: Path, ir: IR, input_fields: list[str], output_fields: list[str]):
        self.binary_path = binary_path
        self.ir = ir
        self.input_fields = input_fields
        self.output_fields = output_fields

    def run(self, data) -> dict[str, list[dict]]:
        """Run the binary on data and return results.

        Uses raw binary format: [n_rows as u64][f64 * n_fields * n_rows]

        Args:
            data: Either dict of entity -> list of row dicts, or numpy array
                  If numpy array, shape must be (n_rows, n_input_fields)
        """
        import numpy as np

        # Handle numpy array input (fast path)
        if isinstance(data, np.ndarray):
            input_arr = data.astype(np.float64, copy=False)
            entity_name = "data"
            n_rows = len(input_arr)
        else:
            # Get entity name and rows
            entity_name = list(data.keys())[0]
            rows = data[entity_name]
            n_rows = len(rows)

            # Build input array efficiently
            input_arr = np.array([
                [float(row.get(field, 0.0)) for field in self.input_fields]
                for row in rows
            ], dtype=np.float64)

        # Write binary input
        input_path = tempfile.mktemp(suffix='.bin')
        with open(input_path, 'wb') as f:
            f.write(struct.pack('<Q', n_rows))
            input_arr.tofile(f)

        output_path = tempfile.mktemp(suffix='.bin')

        try:
            result = subprocess.run(
                [str(self.binary_path), input_path, output_path],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"Binary failed: {result.stderr}")

            # Read binary output
            with open(output_path, 'rb') as f:
                out_n = struct.unpack('<Q', f.read(8))[0]
                output_arr = np.fromfile(f, dtype=np.float64).reshape(out_n, len(self.output_fields))

            # Return numpy array if input was numpy
            if isinstance(data, np.ndarray):
                return output_arr

            # Convert to list of dicts
            output_rows = [
                {field: output_arr[i, j] for j, field in enumerate(self.output_fields)}
                for i in range(out_n)
            ]

            return {entity_name: output_rows}
        finally:
            os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


def compile_to_binary(ir: IR, cache: bool = True) -> CompiledBinary:
    """Compile IR to native binary for maximum performance.

    Args:
        ir: Compiled IR from rac.compile()
        cache: Cache compiled binaries (default True)

    Returns:
        CompiledBinary that can run data at high speed

    Example:
        >>> from rac import parse, compile
        >>> from rac.native import compile_to_binary
        >>>
        >>> module = parse(open('rules.rac').read())
        >>> ir = compile([module], as_of=date(2024, 6, 1))
        >>> binary = compile_to_binary(ir)
        >>>
        >>> result = binary.run({'person': [{'income': 50000}, ...]})
    """
    cargo = ensure_cargo()

    # Get entity info
    entity_name = None
    input_fields = []
    output_fields = []
    for path in ir.order:
        var = ir.variables[path]
        if var.entity:
            entity_name = var.entity
            output_fields.append(path)

    if entity_name and entity_name in ir.schema_.entities:
        input_fields = list(ir.schema_.entities[entity_name].fields.keys())

    ir_hash = _ir_hash(ir)
    project_dir = CACHE_DIR / "projects" / ir_hash

    binary_name = "rac_native.exe" if platform.system() == "Windows" else "rac_native"
    binary_path = project_dir / "target" / "release" / binary_name

    if cache and binary_path.exists():
        return CompiledBinary(binary_path, ir, input_fields, output_fields)

    # Create Cargo project
    project_dir.mkdir(parents=True, exist_ok=True)

    # Cargo.toml - no serde needed for binary I/O
    (project_dir / "Cargo.toml").write_text('''[package]
name = "rac_native"
version = "0.1.0"
edition = "2021"

[dependencies]
rayon = "1.10"

[profile.release]
lto = true
codegen-units = 1
''')

    # Generate Rust code
    rust_code = generate_rust(ir)
    main_code = _generate_main(ir, input_fields, output_fields)

    src_dir = project_dir / "src"
    src_dir.mkdir(exist_ok=True)
    (src_dir / "main.rs").write_text(rust_code + "\n" + main_code)

    # Build
    print("Compiling native binary...")
    result = subprocess.run(
        [str(cargo), "build", "--release", "--quiet"],
        cwd=project_dir,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed:\n{result.stderr}")

    print("Compilation complete")
    return CompiledBinary(binary_path, ir, input_fields, output_fields)


def _generate_main(ir: IR, input_fields: list[str], output_fields: list[str]) -> str:
    """Generate main function with binary I/O.

    Binary format:
    - Input: [n_rows: u64][f64 * n_input_fields * n_rows]
    - Output: [n_rows: u64][f64 * n_output_fields * n_rows]
    """
    # Find entity info
    entity_name = None
    for path in ir.order:
        var = ir.variables[path]
        if var.entity:
            entity_name = var.entity
            break

    if not entity_name:
        return """
fn main() {
    eprintln!("No entity variables to compute");
}
"""

    type_name = "".join(part.capitalize() for part in entity_name.split("_"))
    n_inputs = len(input_fields)
    n_outputs = len(output_fields)

    # Generate field assignments from binary data
    field_reads = []
    for i, f in enumerate(input_fields):
        field_reads.append(f"                {f}: row[{i}]")

    # Generate output writes
    output_writes = []
    for i, path in enumerate(output_fields):
        safe_name = path.replace("/", "_")
        output_writes.append(f"            out[{i}] = o.{safe_name};")

    return f'''
use rayon::prelude::*;
use std::env;
use std::fs::File;
use std::io::{{Read, Write, BufReader, BufWriter}};

fn main() {{
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {{
        eprintln!("Usage: {{}} <input.bin> <output.bin>", args[0]);
        std::process::exit(1);
    }}

    // Read binary input
    let mut file = BufReader::new(File::open(&args[1]).expect("Failed to open input"));
    let mut buf8 = [0u8; 8];
    file.read_exact(&mut buf8).expect("Failed to read count");
    let n_rows = u64::from_le_bytes(buf8) as usize;

    let n_input_fields = {n_inputs};
    let mut input_data = vec![0.0f64; n_rows * n_input_fields];
    for i in 0..n_rows * n_input_fields {{
        file.read_exact(&mut buf8).expect("Failed to read data");
        input_data[i] = f64::from_le_bytes(buf8);
    }}

    // Compute scalars
    let scalars = Scalars::compute();

    // Process rows in parallel
    let n_output_fields = {n_outputs};
    let mut output_data = vec![0.0f64; n_rows * n_output_fields];

    input_data
        .par_chunks(n_input_fields)
        .zip(output_data.par_chunks_mut(n_output_fields))
        .for_each(|(row, out)| {{
            let input = {type_name}Input {{
{chr(10).join(field_reads)}
            }};
            let o = {type_name}Output::compute(&input, &scalars);
{chr(10).join(output_writes)}
        }});

    // Write binary output
    let mut out_file = BufWriter::new(File::create(&args[2]).expect("Failed to create output"));
    out_file.write_all(&(n_rows as u64).to_le_bytes()).expect("Failed to write count");
    for v in output_data {{
        out_file.write_all(&v.to_le_bytes()).expect("Failed to write data");
    }}
}}
'''
