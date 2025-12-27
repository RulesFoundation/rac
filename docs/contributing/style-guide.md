# Style Guide

## File Naming

- Variables: `snake_case.rac`
- Parameters: `snake_case.yaml`
- Directories: Match statutory structure

## Variable Naming

- Use statutory terminology where possible
- Prefix intermediate calculations with parent variable name
- Examples: `eitc`, `eitc_phase_in`, `eitc_phase_out`

## References

Always use explicit aliases:

```cosilico
# Good
references:
  federal_agi: us/irc/.../§62/(a)/adjusted_gross_income
  ca_agi: us-ca/rtc/.../§17072/adjusted_gross_income

# Avoid collision-prone leaf names
```

## Citations

- Every variable must have a `reference` field
- Use full statutory citation format
- Example: `"26 USC § 32(a)(2)(A)"`

## Formulas

- Use `let` bindings for clarity
- Keep formulas readable, not compact
- Comment complex logic

```cosilico
formula {
  # Phase-in: credit percentage of earned income up to cap
  let phase_in = credit_percentage * min(earned_income, earned_income_amount)

  # Phase-out: reduction for income above threshold
  let phase_out = phaseout_rate * max(0, phaseout_income - phaseout_start)

  return max(0, phase_in - phase_out)
}
```
