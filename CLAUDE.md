# cosilico-engine

Core DSL parser, executor, and vectorized runtime for Cosilico.

## ⚠️ CRITICAL: No Country-Specific Rules ⚠️

**This repo contains ONLY the DSL infrastructure. NO statute files.**

Country-specific rules (.cosilico files) belong in:
- `cosilico-us/` - US federal statutes (Title 26 IRC, Title 7 SNAP)
- `cosilico-uk/` - UK statutes (future)

This separation has been violated multiple times. DO NOT add statute files here.

## What Belongs Here

- `src/cosilico/dsl_parser.py` - DSL parser
- `src/cosilico/dsl_executor.py` - Single-case executor
- `src/cosilico/vectorized_executor.py` - Microsimulation executor
- `src/cosilico/microsim.py` - CPS microdata runner (loads from cosilico-us)
- `tests/` - Unit tests with inline test fixtures only

## What Does NOT Belong Here

- `statute/` directory - DELETE if it exists
- `.cosilico` files with real statute encodings
- `parameters.yaml` with real IRS values

## Commands

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Tests
pytest tests/ -v

# Microsim (requires cosilico-us and cosilico-data-sources)
python -m cosilico.microsim --year 2024
```

## Related Repos

- **cosilico-us** - US statute encodings
- **cosilico-data-sources** - CPS microdata, parameters
- **cosilico-compile** - Multi-target compiler
