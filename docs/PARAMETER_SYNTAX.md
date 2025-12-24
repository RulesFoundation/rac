# Cosilico Parameter System

## Overview

Cosilico uses YAML files for parameter declarations, separate from the DSL formulas.
This allows parameters to be updated independently of code, enables reform modeling,
and provides a clear audit trail for policy values.

## Parameter Patterns

### 1. Simple Time-Varying Parameters

```yaml
# parameters/irs/payroll/social_security/cap.yaml
description: Social Security wage base
unit: currency-USD
period: year
values:
  2024-01-01: 168_600
  2023-01-01: 160_200
  2022-01-01: 147_000
reference:
  - title: Rev. Proc. 2023-34
    href: https://www.irs.gov/pub/irs-drop/rp-23-34.pdf
```

### 2. Parameters Varying by Filing Status

For parameters that differ by filing status, use top-level keys:

```yaml
# parameters/irs/niit/threshold.yaml
description: Net Investment Income Tax modified AGI threshold
unit: currency-USD

SINGLE:
  values:
    2013-01-01: 200_000
HEAD_OF_HOUSEHOLD:
  values:
    2013-01-01: 200_000
JOINT:
  values:
    2013-01-01: 250_000
SEPARATE:
  values:
    2013-01-01: 125_000
SURVIVING_SPOUSE:
  values:
    2013-01-01: 250_000

reference:
  - title: 26 USC § 1411
    href: https://www.law.cornell.edu/uscode/text/26/1411
```

### 3. Bracket/Scale Parameters

For parameters indexed by a numeric dimension (children, income):

```yaml
# parameters/irs/credits/eitc/max.yaml
description: EITC maximum credit amount by number of children
unit: currency-USD
index: num_children

brackets:
  - threshold: 0
    values:
      2024-01-01: 632
      2023-01-01: 600
  - threshold: 1
    values:
      2024-01-01: 4_213
      2023-01-01: 3_995
  - threshold: 2
    values:
      2024-01-01: 6_960
      2023-01-01: 6_604
  - threshold: 3
    values:
      2024-01-01: 7_830
      2023-01-01: 7_430

reference:
  - title: Rev. Proc. 2023-34
    href: https://www.irs.gov/pub/irs-drop/rp-23-34.pdf#page=10
```

### 4. Combined: Filing Status + Brackets

When both dimensions are needed:

```yaml
# parameters/irs/credits/eitc/phase_out_start.yaml
description: EITC phase-out start threshold
unit: currency-USD
index: num_children

SINGLE:
  brackets:
    - threshold: 0
      values:
        2024-01-01: 10_330
    - threshold: 1
      values:
        2024-01-01: 22_720

JOINT:
  brackets:
    - threshold: 0
      values:
        2024-01-01: 17_250
    - threshold: 1
      values:
        2024-01-01: 29_640
```

## DSL Reference Syntax

### Simple Parameter Access

```cosilico
# In formula block:
let wage_base = parameter("irs.payroll.social_security.cap")
```

### Filing Status Lookup

```cosilico
# Automatic lookup based on tax unit's filing_status
let threshold = parameter("irs.niit.threshold", filing_status)
```

### Bracket Lookup

```cosilico
# Lookup by numeric index
let max_credit = parameter("irs.credits.eitc.max", num_children)
```

### Combined Lookup

```cosilico
# Both filing status and index
let phase_out_start = parameter(
  "irs.credits.eitc.phase_out_start",
  filing_status,
  num_children
)
```

## Parameter Resolution

1. **Path Resolution**: `irs.payroll.social_security.cap` maps to
   `parameters/irs/payroll/social_security/cap.yaml`

2. **Time Resolution**: The executor uses the simulation period to select
   the appropriate value from the `values` block

3. **Filing Status Resolution**: If a filing status key is provided, lookup
   uses that key; otherwise uses the tax unit's filing status

4. **Bracket Resolution**: For bracket parameters, finds the appropriate
   bracket based on the index value (typically uses <= comparison)

## Metadata Fields

| Field | Description |
|-------|-------------|
| `description` | Human-readable description of the parameter |
| `unit` | Data type: `currency-USD`, `/1` (rate), `year`, `child`, etc. |
| `period` | Time period: `year`, `month`, etc. |
| `reference` | Array of {title, href} citations to authoritative sources |
| `index` | For brackets: name of the indexing dimension |

## Benefits

1. **Separation of Concerns**: Policy values separate from calculation logic
2. **Reform Modeling**: Easy to create policy variants by overriding parameters
3. **Auditability**: Clear citations to authoritative sources
4. **Time Travel**: Built-in support for historical and future values
5. **Validation**: YAML schema can enforce structure and types
