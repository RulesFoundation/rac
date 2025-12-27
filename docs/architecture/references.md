# References System

Variables in Cosilico have no function arguments. Instead, they declare dependencies through a **references block** - a named mapping from local aliases to absolute statutory paths.

## Why No Arguments?

Traditional approach:
```python
def earned_income_credit(agi, earned_income, filing_status, children):
    ...
```

Cosilico approach:
```python
references:
  agi: us/irc/.../§62/(a)/adjusted_gross_income
  earned_income: us/irc/.../§32/(c)/(2)/(A)/earned_income
  filing_status: us/irc/.../§1/(h)/filing_status
  children: us/irc/.../§32/(c)/(3)/qualifying_children_count

def earned_income_credit() -> Money:
    ...
```

Benefits:
1. **Dependencies are explicit** - All inputs declared upfront
2. **Cross-references are legal citations** - The reference IS the law
3. **Engine handles resolution** - Entity and period injection is automatic
4. **Name collisions handled** - Aliases disambiguate

## Reference Block Syntax

```python
references:
  <local_alias>: <absolute_statutory_path>[@vintage]
```

### Basic References

```python
references:
  # Variable from same jurisdiction
  earned_income: us/irc/.../§32/(c)/(2)/(A)/earned_income

  # Parameter from same section
  credit_percentage: us/irc/.../§32/(b)/(1)/credit_percentage

  # Cross-jurisdiction reference
  federal_agi: us/irc/.../§62/(a)/adjusted_gross_income
```

### Aliasing for Name Collisions

When two references have the same leaf name:

```python
# BAD - collision
references:
  adjusted_gross_income: us/irc/.../§62/(a)/adjusted_gross_income
  adjusted_gross_income: us-ca/rtc/.../§17072/adjusted_gross_income  # Collision!

# GOOD - aliased
references:
  federal_agi: us/irc/.../§62/(a)/adjusted_gross_income
  ca_agi: us-ca/rtc/.../§17072/adjusted_gross_income

def ca_taxable_income() -> Money:
    return federal_agi + ca_additions  # Clear which AGI
```

## Vintage Pinning

By default, references resolve to the current simulation period. But you can pin to specific vintages:

### Relative Vintages

```python
references:
  # Prior year
  prior_year_agi: us/irc/.../§62/(a)/adjusted_gross_income@year-1

  # Prior month (for monthly variables)
  last_month_benefit: us-ca/wic/.../§14101/benefit_amount@month-1

  # Prior quarter
  q1_wages: us-ca/uic/.../§1281/quarterly_wages@quarter-2
```

### Absolute Vintages

```python
references:
  # Specific date (for conformity/sunset)
  tcja_rates: us/irc/.../§1/(j)/(2)/rate_brackets@2018-01-01

  # Pre-TCJA reference
  pre_tcja_exemption: us/irc/.../§151/(d)/personal_exemption@2017-12-31
```

### Use Cases

**Safe Harbor Elections**
```python
references:
  current_agi: us/irc/.../§62/(a)/adjusted_gross_income
  prior_agi: us/irc/.../§62/(a)/adjusted_gross_income@year-1

def safe_harbor_agi() -> Money:
    return max(current_agi, prior_agi)
```

**State Conformity to Specific Federal Year**
```python
# California conforms to IRC as of specific date
references:
  federal_agi: us/irc/.../§62/(a)/adjusted_gross_income@2015-01-01

def ca_conformity_agi() -> Money:
    return federal_agi  # CA frozen to 2015 federal definitions
```

**Sunset Provisions**
```python
references:
  tcja_exemption: us/irc/.../§151/(d)/personal_exemption@2017-12-31
  current_exemption: us/irc/.../§151/(d)/personal_exemption

def personal_exemption() -> Money:
    if year >= 2026:
        return tcja_exemption  # Reverts to 2017 law
    return current_exemption
```

**Lookback Periods (Unemployment)**
```python
references:
  q1_wages: us-ca/uic/.../§1281/quarterly_wages@quarter-2
  q2_wages: us-ca/uic/.../§1281/quarterly_wages@quarter-3
  q3_wages: us-ca/uic/.../§1281/quarterly_wages@quarter-4
  q4_wages: us-ca/uic/.../§1281/quarterly_wages@quarter-5

def base_period_wages() -> Money:
    return q1_wages + q2_wages + q3_wages + q4_wages
```

## Parameters in References

Parameters use the same syntax as variables:

```python
references:
  # Variables
  earned_income: us/irc/.../§32/(c)/(2)/(A)/earned_income
  qualifying_children: us/irc/.../§32/(c)/(3)/qualifying_children_count

  # Parameters (same syntax - engine knows by file type)
  credit_percentage: us/irc/.../§32/(b)/(1)/credit_percentage
  earned_income_amount: us/irc/.../§32/(b)/(2)/(A)/earned_income_amount

def initial_credit_amount() -> Money:
    return credit_percentage[qualifying_children] * min(
        earned_income,
        earned_income_amount[qualifying_children]
    )
```

The engine distinguishes by what's at the path:
- `.rac` file → variable (formula)
- `.yaml` file → parameter (time-varying data)

## Engine Resolution

The engine handles all dependency resolution:

```python
def resolve(variable_path: str, entity: Entity, period: Period) -> Value:
    var = load_variable(variable_path)

    deps = {}
    for local_name, ref in var.references.items():
        # Parse vintage if present
        path, vintage = parse_vintage(ref)
        resolved_period = apply_vintage(period, vintage)

        # Recursively resolve
        deps[local_name] = resolve(path, entity, resolved_period)

    # Execute formula with resolved dependencies
    return var.formula(**deps)
```

Variables don't know:
- Which entity they're being called for (Person A vs Person B)
- Which period (2024 vs 2025)
- How dependencies are resolved

The engine injects everything.

## Dependency Graph

References create an explicit dependency graph:

```
earned_income_credit
├── has_qualifying_child
├── meets_age_requirement
├── meets_filing_requirement
├── investment_income_disqualified
│   └── investment_income
├── initial_credit_amount
│   ├── earned_income
│   ├── credit_percentage [param]
│   └── earned_income_amount [param]
└── credit_reduction_amount
    ├── phaseout_income
    │   ├── adjusted_gross_income
    │   └── earned_income
    ├── phaseout_rate [param]
    └── phaseout_amount [param]
```

This graph is:
- **Explicit** - Declared, not inferred
- **Static** - Known at compile time
- **Acyclic** - No circular dependencies allowed

## Comparison: Inline vs Block

We considered inline path references:

```python
# Inline approach (rejected)
def ca_taxable_income() -> Money:
    return ref("us/irc/.../§62/(a)/adjusted_gross_income") + ref("us-ca/rtc/.../§17220/additions")
```

**Why we chose the references block:**

| Aspect | References Block | Inline Paths |
|--------|------------------|--------------|
| Readability | Formula is clean math | Cluttered with paths |
| Duplication | Alias once, use many | Repeat full path |
| Static analysis | Easy dep extraction | Parse formula AST |
| Legal clarity | All citations in one place | Scattered through formula |
| Refactoring | Change path once | Find/replace everywhere |
