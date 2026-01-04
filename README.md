# RAC

High-performance DSL for encoding tax and benefit law. Compiles to native Rust for ~10M households/sec.

## Install

```bash
pip install -e .
```

## Run a reform

```bash
python examples/run_reform.py examples/uk_tax_benefit.rac examples/reform.rac
```

Output:
```
Population: 1,000,000
Baseline run: 0.24s (4.1M/sec)
Reform run:   0.21s (4.7M/sec)

Aggregate impact:
  Total cost: £10.04bn/year
  Avg gain:   £836/month
  Winners:    858,609 (85.9%)

By income decile:
  Decile   Avg Income   Avg Gain  % Winners
       1 £     7,644 £        0         0%
       2 £    12,941 £      152        59%
       3 £    17,299 £      532       100%
       ...
```

## DSL syntax

```rac
# Entity with typed fields
entity person:
    gross_income: float
    age: int

# Temporal parameters
variable gov/hmrc/it/personal_allowance:
    from 2024-04-06: 12570

# Entity-scoped computation
variable person/income_tax:
    entity: person
    from 2024-04-06:
        max(0, gross_income - gov/hmrc/it/personal_allowance) * 0.20

# Reform via amendment
amend gov/hmrc/it/personal_allowance:
    from 2024-04-06: 15000
```

## Python API

```python
from datetime import date
from rac import parse, compile, compile_to_binary
import numpy as np

# Parse and compile
module = parse(open('rules.rac').read())
ir = compile([module], as_of=date(2024, 6, 1))

# Compile to native binary (auto-installs Rust on first run)
binary = compile_to_binary(ir)

# Run on numpy arrays
data = {'person': np.array([[50000.0, 30], [75000.0, 45]])}  # income, age
result = binary.run(data)  # Returns numpy array of computed values
```

## Structure

```
src/rac/
├── parser.py    # Lexer and recursive descent parser
├── compiler.py  # Temporal resolution, dependency ordering
├── executor.py  # Pure Python executor (for debugging)
├── native.py    # Rust compilation (production speed)
└── codegen/     # Rust code generation
```

## Tests

```bash
pytest tests/ -v
```
