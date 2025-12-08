# Statute-Organized Code

Cosilico's defining architectural choice: **code structure mirrors legal structure**.

## The Core Insight

Traditional tax software organizes by calculation type:
```
calculations/
├── credits/
│   ├── eitc.py
│   └── ctc.py
├── deductions/
│   └── standard_deduction.py
└── income/
    └── agi.py
```

Cosilico organizes by statutory citation:
```
us/
└── title_26/              # 26 USC (Internal Revenue Code)
    └── §32/               # Section 32 - EITC
        ├── a/1/earned_income_credit.cosilico     # §32(a)(1)
        ├── a/2/A/initial_credit_amount.cosilico  # §32(a)(2)(A)
        ├── a/2/B/credit_reduction_amount.cosilico # §32(a)(2)(B)
        ├── b/1/credit_percentage.yaml            # §32(b)(1) parameters
        ├── b/2/A/amounts.yaml                    # §32(b)(2)(A) parameters
        ├── c/2/A/earned_income.cosilico          # §32(c)(2)(A) definition
        ├── i/1/disqualified_income_limit.yaml    # §32(i)(1) parameter
        ├── j/1/indexing_rule.yaml                # §32(j)(1) indexing
        └── j/2/rounding_rules.yaml               # §32(j)(2) rounding
```

**The path IS the legal citation.** `us/26/32/a/1/` maps to "26 USC §32(a)(1)".

Section numbers are unique within a title, so we don't need the full subtitle/chapter/subchapter hierarchy.

## Why This Matters

### 1. Citation is Path

```python
# Variable at:
# us/26/32/a/1/earned_income_credit

# Maps directly to:
# 26 USC §32(a)(1)
```

Legal citation and code location are one and the same.

### 2. Auditability

When a regulator asks "where does this calculation come from?", the answer is the folder name. No documentation lookup required.

### 3. Legal Diff = Code Diff

When Congress amends §32(b)(2), the git diff shows exactly what changed:
```diff
- us/26/32/b/2/parameters/earned_income_amount.yaml
+ us/26/32/b/2/parameters/earned_income_amount.yaml
```

### 4. AI Training Signal

The encoder learns that statute structure maps to code structure. The path becomes training metadata for free.

## One Variable Per Clause

Each statutory clause gets exactly one variable. Complex provisions become compositions of atomic pieces.

### Example: EITC (26 USC §32)

**§32(a)(1)** - "there shall be allowed as a credit..."
```
§32/a/1/variables/earned_income_credit.cosilico
```

**§32(a)(2)(A)** - "credit percentage of earned income..."
```
§32/a/2/A/variables/initial_credit_amount.cosilico
```

**§32(a)(2)(B)** - "the greater of AGI or earned income..."
```
§32/a/2/B/variables/phaseout_income.cosilico
```

**§32(b)(1)** - Credit percentages (parameter, not formula)
```
§32/b/1/parameters/credit_percentage.yaml
```

**§32(c)(1)(A)(i)** - "has qualifying child"
```
§32/c/1/A/i/variables/has_qualifying_child.cosilico
```

### The Final Credit Composes Everything

```cosilico
# us/26/32/a/1/earned_income_credit.cosilico
#
# 26 USC §32(a)(1)
#
# "In the case of an eligible individual, there shall be allowed as a
# credit against the tax imposed by this subtitle for the taxable year
# an amount equal to the credit percentage of so much of the taxpayer's
# earned income for the taxable year as does not exceed the earned
# income amount, over [the phaseout reduction]."

module us.irc.subtitle_a.chapter_1.subchapter_a.part_iv.subpart_c.§32.a.1
version "2024.1"
jurisdiction us

references {
  # Eligibility from §32(c)(1)
  is_eligible_individual: us/26/32/c/1/A/i/is_eligible_individual

  # Credit components from §32(a)(2)
  initial_credit_amount: us/26/32/a/2/A/initial_credit_amount
  credit_reduction_amount: us/26/32/a/2/B/credit_reduction_amount
}

variable earned_income_credit {
  entity TaxUnit
  period Year
  dtype Money
  unit "USD"
  reference "26 USC § 32(a)(1)"
  label "Earned Income Tax Credit"

  formula {
    if not is_eligible_individual then
      return 0

    return max(0, initial_credit_amount - credit_reduction_amount)
  }
}
```

## Handling Non-Statute Sources

Not everything comes from statute. Regulations, agency guidance, and case law also define rules.

### Regulations

```
cosilico-us/
├── irc/                           # Statutes (primary)
│   └── .../§32/
│
└── cfr/                           # Code of Federal Regulations
    └── title_26/
        └── §1.32-1/               # Reg interpreting §32
            └── variables/
                └── qualifying_child_tiebreaker.cosilico
```

The path `us/cfr/title_26/§1.32-1` maps to "26 CFR §1.32-1".

### Agency Guidance

```
cosilico-us/
└── irs/
    └── notices/
        └── 2024-01/
            └── parameters/
                └── safe_harbor_threshold.yaml
```

### Hierarchy of Authority

When sources conflict, statute wins:
```
statute > regulation > agency_guidance > case_law
```

The engine resolves by source authority.

## Cross-References as Imports

Legal cross-references become code imports:

```python
# us-ca/rtc/division_2/.../§17041/(a)/variables/ca_taxable_income.cosilico

"""
RTC §17041(a) - California taxable income starts with federal AGI
"""

references:
  # "as defined in section 62" becomes a reference to §62
  federal_agi: us/irc/subtitle_a/chapter_1/subchapter_b/part_1/§62/(a)/adjusted_gross_income
  ca_additions: us-ca/rtc/division_2/part_10/chapter_2.5/§17220/additions
  ca_subtractions: us-ca/rtc/division_2/part_10/chapter_3/§17250/subtractions

def ca_taxable_income() -> Money:
    return federal_agi + ca_additions - ca_subtractions
```

When statute says "as defined in section 62", the code literally points to `§62`.

## Depth Guidelines

How deep to go in the hierarchy?

- **Leaf nodes are sections** - The smallest citable unit (§32, §62, etc.)
- **Subsections become folders** when they define distinct concepts
- **Single-clause sections** have one variable
- **Multi-clause sections** (like §32) get folder trees

### Example: Simple vs Complex

**Simple section (§63(c)(2) - Standard Deduction Amount)**
```
§63/(c)/(2)/parameters/standard_deduction_amount.yaml
```
One parameter, one file.

**Complex section (§32 - EITC)**
```
§32/
├── (a)/(1)/...
├── (a)/(2)/(A)/...
├── (a)/(2)/(B)/...
├── (b)/(1)/...
├── (b)/(2)/...
├── (c)/(1)/(A)/(i)/...
├── (c)/(1)/(A)/(ii)/...
├── (c)/(1)/(B)/...
├── (c)/(1)/(F)/...
├── (c)/(2)/(A)/...
├── (c)/(2)/(B)/...
├── (i)/(1)/...
└── (i)/(2)/...
```
Full tree mirroring statute structure.

## Benefits Summary

| Benefit | How It Works |
|---------|--------------|
| **Traceability** | Path = citation |
| **Auditability** | Folder name answers "where does this come from?" |
| **Legal diffs** | Amendment = file change |
| **AI training** | Structure is metadata |
| **Debugging** | Trace clause-by-clause |
| **No mapping** | Filesystem IS the citation system |
