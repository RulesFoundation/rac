# Cosilico DSL Specification

Version 2.0 - December 2025

## Overview

The Cosilico DSL encodes tax and benefit statutes as executable rules. Design principles:
- **Law-faithful**: Structure mirrors statute organization
- **Readable**: Accessible to policy experts, not just programmers
- **Strict**: Parser catches errors early, no ambiguity
- **Minimal**: Few built-ins, compose from primitives

## File Structure

### One Variable Per File

Each `.rac` file defines exactly one variable. The filename IS the variable name.

```
statute/26/32/a/1/earned_income_credit.rac
```

Defines the variable `earned_income_credit`.

### Basic Structure

```cosilico
# statute/26/32/a/1/earned_income_credit.rac

imports:
  earned_income: statute/26/32/c/2/A/earned_income
  agi: statute/26/62/a/adjusted_gross_income
  phase_in_rate: params/eitc/2024#phase_in_rate
  max_credit: params/eitc/2024#max_credit

entity TaxUnit
period Year
dtype Money
default 0

rounding:
  type: floor
  citation: 26/32/j/2/A

formula:
  credit = earned_income * phase_in_rate
  return min(credit, max_credit)
```

## Syntax Rules

### Block Syntax: Python-style

Use `:` plus indentation, not `{}` braces:

```cosilico
# Correct
imports:
  foo: path/to/foo
  bar: path/to/bar

formula:
  x = a + b
  return x

# Wrong
imports {
  foo: path/to/foo
}
```

### Variable Assignments: No `let` Keyword

Python-style assignments:

```cosilico
# Correct
credit = income * rate
threshold = get_threshold(filing_status)

# Wrong
let credit = income * rate
```

### No Reassignment

Variables are immutable. Use distinct names:

```cosilico
# Correct
tax_10 = bracket_10 * rate_10
tax_12 = bracket_12 * rate_12
total_tax = tax_10 + tax_12

# Wrong
tax = 0
tax = tax + bracket_10 * rate_10
tax = tax + bracket_12 * rate_12
```

### Conditionals: `if/then/else` Only

No ternary `?:` operator:

```cosilico
# Correct
result = if income > threshold then credit else 0

# Multi-line
result = if income > threshold then
  credit * phase_out_rate
else
  0

# Wrong
result = income > threshold ? credit : 0
```

### Numeric Literals: Only 0, 1, -1

All other values must come from parameters:

```cosilico
# Correct
imports:
  rate: params/eitc/2024#phase_in_rate

formula:
  credit = income * rate
  return max(0, credit)  # 0 is allowed

# Wrong
formula:
  credit = income * 0.34  # Hardcoded rate not allowed
```

### No Explicit Loops

Use functional primitives:

```cosilico
# Correct
total = sum(income_sources)
has_income = any(source > 0 for source in sources)
all_eligible = all(Person.is_eligible)

# Wrong
total = 0
for source in sources:
  total = total + source
```

## Entity Hierarchy

### Going UP (Child to Parent)

Reference parent entity directly:

```cosilico
# In a Person-level variable
filing_status = TaxUnit.filing_status
state = Household.state_code
```

### Going DOWN (Parent to Child)

Use aggregation functions:

```cosilico
# In a TaxUnit-level variable
total_wages = sum(Person.wages)
has_elderly = any(Person.age >= elderly_age)
max_income = max(Person.income)
```

### Going UP then DOWN

Path through parent to siblings:

```cosilico
# In a Person-level variable, sum all people in household
household_total = sum(Household.Person.wages)
unit_total = sum(TaxUnit.Person.income)
```

## Temporal References

Access prior period values with brackets:

```cosilico
# Prior year
prior_income = income[year - 1]

# Two years ago
old_loss = loss[year - 2]

# Carryforward sum
carryforward = sum(loss[year - i] for i in 1..carryforward_years)
```

## Types and Units

### Data Types

| dtype | Description | Default |
|-------|-------------|---------|
| Money | Currency amount (USD) | 0 |
| Rate | Decimal rate (0.0 to 1.0) | 0 |
| Boolean | True/False | False |
| Integer | Whole number | 0 |
| Enum | Enumerated type | Must specify |

### Unit Enforcement

The compiler enforces unit consistency:

```cosilico
# Valid
tax = income * rate      # Money * Rate = Money
total = wages + salary   # Money + Money = Money

# Compile error
bad = income + rate      # Money + Rate = ???
```

### Enum Defaults

Enums must have explicit defaults:

```cosilico
dtype FilingStatus
default SINGLE
```

## Rounding

Rounding is a variable attribute, not a formula function:

```cosilico
rounding:
  type: floor          # floor, ceil, or nearest
  citation: 26/32/j/2  # Statute subsection specifying rounding

# For nearest:
rounding:
  type: nearest
  increment: 10        # Round to nearest $10
  citation: 26/24/d/1
```

Rounding is applied automatically to the formula output.

## Imports

### Variable Imports

```cosilico
imports:
  # Local alias: full/path/to/variable
  earned_income: statute/26/32/c/2/A/earned_income
  agi: statute/26/62/a/adjusted_gross_income
```

### Parameter Imports

```cosilico
imports:
  # path/to/params#key
  phase_in_rate: params/eitc/2024#phase_in_rate
  max_credit: params/eitc/2024#max_credit[n_children]
```

### Parameter Breakdowns

Access parameter dimensions with brackets:

```cosilico
threshold = bracket_threshold[filing_status]
rate = credit_rate[n_children]
```

## Built-in Functions

### Aggregation

| Function | Description |
|----------|-------------|
| `sum(...)` | Sum values |
| `any(...)` | True if any value is true |
| `all(...)` | True if all values are true |
| `max(...)` | Maximum value |
| `min(...)` | Minimum value |
| `count(...)` | Count of values |

### Math

| Function | Description |
|----------|-------------|
| `max(a, b)` | Greater of two values |
| `min(a, b)` | Lesser of two values |
| `abs(x)` | Absolute value |
| `floor(x)` | Round down (use in formula if not output rounding) |
| `ceil(x)` | Round up |

### Tax-specific

| Function | Description |
|----------|-------------|
| `bracket_tax(income, brackets, filing_status)` | Progressive bracket calculation |

## Testing

All tests go in separate `.test.yaml` files:

```
statute/26/32/a/1/earned_income_credit.rac
statute/26/32/a/1/earned_income_credit.test.yaml
```

Test format:

```yaml
- name: Single filer with 2 children
  period: 2024
  input:
    earned_income: 20000
    n_qualifying_children: 2
    filing_status: SINGLE
  output:
    earned_income_credit: 6604

- name: Income too high
  period: 2024
  input:
    earned_income: 60000
    n_qualifying_children: 1
  output:
    earned_income_credit: 0
```

## Parser Strictness

### Unknown Fields

The parser errors on unknown fields:

```cosilico
# Error: Unknown field 'reference'
entity TaxUnit
period Year
dtype Money
reference: "26 USC 32"  # Not allowed
```

### Missing Required Fields

Required fields: `entity`, `period`, `dtype`, `formula`

### Helpful Error Messages

```
SyntaxError: Unknown field 'reference' at line 5.
Valid fields: entity, period, dtype, label, description, unit,
              formula, defined_for, default, rounding, imports
```

## Credits Architecture

Credits follow the statute structure. Section 26 maintains the list of sections excluded from the nonrefundable limitation:

```cosilico
# statute/26/26/a/nonrefundable_credit_limitation.rac

imports:
  tax_liability: statute/26/1/tax_liability
  excluded_sections: params/26/26/excluded_credit_sections

formula:
  # Sum credits in subpart A, limited by tax liability
  # Sections in excluded_sections are not limited
  ...
```

Reforms that change refundability modify the `excluded_sections` parameter, just as Congress amends Section 26.

## Vectorization

Write a single formula; the compiler auto-vectorizes:

```cosilico
# Single formula works for both:
# - Individual calculation
# - Microsimulation over millions of records

formula:
  credit = income * rate
  return min(credit, max_credit)
```

## Operator Precedence

From highest to lowest:

| Precedence | Operators |
|------------|-----------|
| 1 | `()` grouping |
| 2 | `.` member access |
| 3 | `[]` temporal/parameter access |
| 4 | function calls |
| 5 | unary `-`, `not` |
| 6 | `*`, `/` |
| 7 | `+`, `-` |
| 8 | `>`, `<`, `>=`, `<=`, `==`, `!=` |
| 9 | `and` |
| 10 | `or` |
| 11 | `if/then/else` |

## Range Syntax

Ranges are **inclusive on start, exclusive on end** (like Python):

```cosilico
# 1..5 means 1, 2, 3, 4 (not 5)
for i in 1..5

# For inclusive end, use 1..=5
for i in 1..=5  # 1, 2, 3, 4, 5

# Empty range if start >= end
for i in 5..3  # empty, no iterations
```

Note: Explicit `for` loops are discouraged. Prefer functional:
```cosilico
# Instead of: for i in 1..n: total += loss[year - i]
# Use:
carryforward = sum(loss[year - 1], loss[year - 2], loss[year - 3])
```

## Error Handling

### Compile-time Errors

The compiler catches:
- Type mismatches (`Money + Rate`)
- Unknown imports (file not found)
- Circular dependencies
- Unknown fields in variable definition
- Numeric literals other than 0, 1, -1
- Reassignment of variables

### Runtime Behavior

| Situation | Behavior |
|-----------|----------|
| Division by zero | Returns 0 (with warning in trace mode) |
| Temporal reference before data exists | Returns variable's default value |
| Missing parameter dimension | Compile error (all dimensions must be defined) |
| Entity aggregation over empty set | Returns 0 for sum, False for any, True for all |

## Temporal Semantics

### Type Consistency

`variable[year - n]` returns the same type as `variable`:

```cosilico
# If income is Money, income[year - 1] is also Money
prior_income = income[year - 1]  # Money
```

### Missing Periods

If the referenced period has no data, the variable's default is used:

```cosilico
# If year is 2024 and no data exists for 2020:
old_income = income[year - 4]  # Returns 0 (default for Money)
```

### Carryforward Pattern

For multi-year carryforwards, list explicitly or use parameter:

```cosilico
imports:
  max_carryforward: params/nol#max_years  # e.g., 20

formula:
  # Sum losses from prior years (up to max)
  available_nol = sum(
    nol[year - 1],
    nol[year - 2],
    nol[year - 3]
    # ... or generate from parameter in compiler
  )
```

## Source Paths

### Path Prefixes

| Prefix | Source Type | Example |
|--------|-------------|---------|
| `statute/` | US Code | `statute/26/32/a/1/eitc` |
| `reg/` | CFR Regulations | `reg/26/1.32-2/earned_income` |
| `notice/` | IRS Notices | `notice/2020-23/deadline` |
| `params/` | Parameters | `params/eitc/2024#rate` |

## Tracing

No trace syntax in source. Use tooling:

```bash
cosilico run --trace earned_income_credit
```

Trace output includes:
```yaml
variable: earned_income_credit
period: 2024
inputs:
  earned_income: {value: 20000, source: "input"}
  phase_in_rate: {value: 0.34, source: "params/eitc/2024"}
intermediates:
  phase_in_amount: 6800
  tentative_credit: 6604
output: 6604
```

## Full Example

```cosilico
# statute/26/32/a/1/earned_income_credit.rac
#
# 26 USC 32(a)(1): "In the case of an eligible individual, there shall
# be allowed as a credit against the tax imposed by this subtitle..."

imports:
  earned_income: statute/26/32/c/2/A/earned_income
  agi: statute/26/62/a/adjusted_gross_income
  n_children: statute/26/32/c/3/qualifying_child_count
  filing_status: statute/26/1/filing_status
  phase_in_rate: params/eitc/2024#phase_in_rate[n_children]
  max_credit: params/eitc/2024#max_credit[n_children]
  phase_out_start: params/eitc/2024#phase_out_start[filing_status][n_children]
  phase_out_rate: params/eitc/2024#phase_out_rate[n_children]

entity TaxUnit
period Year
dtype Money
default 0

rounding:
  type: floor
  citation: 26/32/j/2/A

formula:
  # Phase-in: credit increases with earned income
  phase_in_amount = earned_income * phase_in_rate

  # Cap at maximum credit
  tentative_credit = min(phase_in_amount, max_credit)

  # Phase-out: credit decreases above threshold
  phase_out_income = max(earned_income, agi)
  excess_income = max(0, phase_out_income - phase_out_start)
  phase_out_amount = excess_income * phase_out_rate

  # Final credit
  credit = max(0, tentative_credit - phase_out_amount)
  return credit

defined_for:
  earned_income > 0
```
