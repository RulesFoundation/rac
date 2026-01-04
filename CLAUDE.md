# RAC

Core DSL parser, compiler, and native executor for encoding tax and benefit law.

## Structure

- `src/rac/parser.py` - Lexer and recursive descent parser
- `src/rac/compiler.py` - Temporal resolution, dependency ordering
- `src/rac/executor.py` - Pure Python executor (debugging)
- `src/rac/native.py` - Rust compilation (production)
- `src/rac/codegen/rust.py` - Rust code generation
- `examples/` - UK tax-benefit example and reform runner

## Commands

```bash
uv sync
uv run pytest tests/ -v
uv run python examples/run_reform.py examples/uk_tax_benefit.rac examples/reform.rac
```
