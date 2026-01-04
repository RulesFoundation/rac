# Session state

## Branch
`refactor/clean-parser-engine` (PR #2 open)

## What was done
1. Clean parser engine with temporal resolution, amendments, entity variables
2. Executor supports entity variable chaining
3. **Rust codegen working** - compiles to native code
   - 10M households in 58ms (171M/sec)
   - vs Python: 35k/sec (4800x speedup)

## Files
- `src/rac/ast.py` - Pydantic AST nodes
- `src/rac/parser.py` - Lexer + recursive descent
- `src/rac/compiler.py` - Temporal resolution, amendments, topo sort
- `src/rac/executor.py` - Evaluate IR against relational data
- `src/rac/schema.py` - Entity/FK/PK data model
- `src/rac/codegen/rust.py` - Rust code generator

## Examples
- `examples/niit.rac` - US net investment income tax
- `examples/uk_income_tax.rac` - UK income tax, NICs, child benefit

## Rust compilation workflow
```python
from datetime import date
from rac import parse, compile
from rac.codegen.rust import generate_rust

module = parse(open('examples/uk_income_tax.rac').read())
ir = compile([module], as_of=date(2024, 6, 1))
rust_code = generate_rust(ir)
# Write to .rs file, add main(), compile with rustc -O
```

## What's next
- More codegen targets (JS, Python, SQL)
- SIMD/vectorisation in Rust codegen
- Foreign key lookups in Rust
