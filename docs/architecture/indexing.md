# Inflation Indexing

Inflation indexing is how statutory dollar amounts change over time. This document specifies how Cosilico handles indexed parameters.

## The Problem

Consider EITC's earned income amount (26 USC §32(b)(2)(A)):
- 2024: $7,840 (0 children), $12,390 (1 child), etc.
- 2025: TBD by IRS Rev. Proc. ~November 2024
- 2030: Unknown - depends on future inflation

PolicyEngine's approach: hardcode known values, manually update annually. This doesn't scale.

Cosilico's approach: **encode the indexing rule itself**, compute indexed values, but prefer official published values when available.

## Core Concepts

### 1. Indexing Rule (from statute)

The statute specifies HOW to index. For EITC, it's 26 USC §32(j):

```
us/irc/subtitle_a/chapter_1/subchapter_a/part_iv/subpart_c/§32/j/
├── 1/indexing_rule.yaml    # The rule itself
└── 2/rounding_rule.yaml    # Rounding specification
```

```yaml
# §32/j/1/indexing_rule.yaml
#
# 26 USC §32(j)(1) - Cost-of-Living Adjustments
#
# "(A) IN GENERAL.—In the case of any taxable year beginning after 2015,
# each of the dollar amounts in subsections (b)(2) and (i)(1) shall be
# increased by an amount equal to—
#   (i) such dollar amount, multiplied by
#   (ii) the cost-of-living adjustment determined under section 1(f)(3)
#        for the calendar year in which the taxable year begins..."

indexing_rule:
  description: EITC cost-of-living adjustment
  reference: "26 USC § 32(j)(1)"

  applies_to:
    - us/irc/.../§32/b/2/A/earned_income_amount
    - us/irc/.../§32/b/2/A/phaseout_amount
    - us/irc/.../§32/b/2/B/joint_return_adjustment
    - us/irc/.../§32/i/1/disqualified_income_limit

  method:
    type: cost_of_living_adjustment
    reference_section: "§1(f)(3)"  # Points to where index is defined
    base_year: 2015
    # Note: §1(f)(3) specifies C-CPI-U since TCJA (2018+), CPI-U before

  rounding:
    reference: "26 USC § 32(j)(2)(A)"
    rule: round_down_to_nearest
    amount: 10  # Nearest $10
```

### 2. Index Definition (the actual index)

The index itself lives at its statutory home:

```
us/irc/subtitle_a/chapter_1/subchapter_a/part_i/§1/f/3/
└── cost_of_living_index.yaml
```

```yaml
# §1/f/3/cost_of_living_index.yaml
#
# 26 USC §1(f)(3) - Cost-of-Living Adjustment
#
# Uses C-CPI-U (Chained CPI) since TCJA, CPI-U before

cost_of_living_index:
  description: Cost-of-living adjustment for tax provisions
  reference: "26 USC § 1(f)(3)"

  # Which underlying index to use (changes over time)
  underlying_index:
    - effective_from: 1993-01-01
      effective_to: 2017-12-31
      index: cpi_u
      reference: "26 USC § 1(f)(3) (pre-TCJA)"

    - effective_from: 2018-01-01
      effective_to: 2025-12-31
      index: chained_cpi_u
      reference: "26 USC § 1(f)(3)(B) as amended by TCJA § 11002"

    - effective_from: 2026-01-01
      index: cpi_u  # TCJA sunset reverts to CPI-U
      reference: "26 USC § 1(f)(3) (post-TCJA sunset)"
      uncertainty: tcja_sunset
```

### 3. Index Values (the data)

Raw index values live in a separate data layer:

```
data/indices/
├── cpi_u/
│   ├── historical.yaml      # BLS published values (authoritative)
│   └── forecast/
│       ├── cbo_2024_06.yaml  # CBO forecast from June 2024
│       └── fed_2024_09.yaml  # Fed forecast from Sept 2024
│
└── chained_cpi_u/
    ├── historical.yaml
    └── forecast/
        └── cbo_2024_06.yaml
```

```yaml
# data/indices/chained_cpi_u/historical.yaml
chained_cpi_u:
  description: Chained Consumer Price Index for All Urban Consumers
  source: "Bureau of Labor Statistics"
  source_url: "https://www.bls.gov/cpi/data.htm"

  # Annual average values (used for tax indexing)
  annual_average:
    2017: 245.120
    2018: 251.107
    2019: 255.657
    2020: 258.811
    2021: 270.970
    2022: 292.655
    2023: 304.702
    # 2024: TBD (published ~January 2025)

  # For precision: August values (used for some provisions)
  august_values:
    2017: 245.519
    2018: 252.146
    # ...
```

```yaml
# data/indices/chained_cpi_u/forecast/cbo_2024_06.yaml
forecast:
  provider: cbo
  vintage: 2024-06
  source: "CBO Economic Outlook, June 2024"
  source_url: "https://www.cbo.gov/data/budget-economic-data"

  values:
    2024: 312.5  # CBO projected
    2025: 320.1
    2026: 327.3
    2027: 334.2
    2028: 341.0
    2029: 347.8
    2030: 354.7

  uncertainty:
    # CBO provides ranges
    2025: { low: 316.2, high: 324.0 }
    2026: { low: 319.8, high: 334.8 }
```

## Precedence Tiers

When resolving a parameter value for a given year, use this precedence:

```
1. PUBLISHED     - Official government value (Rev. Proc., etc.)
2. PROJECTED     - Our calculation using statute + forecasts
3. CALCULATED    - On-the-fly from base year + latest index
```

### Implementation

```yaml
# §32/b/2/A/amounts.yaml

earned_income_amount:
  reference: "26 USC § 32(b)(2)(A)"
  indexed_by: [num_qualifying_children]
  indexing_rule: us/irc/.../§32/j/1/indexing_rule

  # Tier 1: Published values (authoritative)
  published:
    - effective_from: 2024-01-01
      source: "Rev. Proc. 2023-34"
      by_num_qualifying_children:
        0: 7840
        1: 12390
        2: 17400
        3: 17400

    - effective_from: 2025-01-01
      status: unknown
      expected_source: "Rev. Proc. 2024-XX (~November 2024)"

  # Tier 2: Projected values (our calculations)
  projected:
    - effective_from: 2025-01-01
      vintage: 2024-06
      method: indexing_rule
      forecast_used: cbo_2024_06
      by_num_qualifying_children:
        0: 8050  # Calculated: 7840 * (CPI_2024 / CPI_2015), rounded
        1: 12720
        2: 17880
        3: 17880

  # Tier 3: Base values for on-the-fly calculation
  base:
    year: 2015
    reference: "26 USC § 32(b)(2)(A) as of 2015"
    by_num_qualifying_children:
      0: 6580
      1: 9880
      2: 13870
      3: 13870
```

### Resolution Logic

```python
def get_parameter_value(param_path: str, year: int, vintage: str = None) -> Value:
    param = load_parameter(param_path)

    # Tier 1: Published
    if year in param.published and param.published[year].status != "unknown":
        return param.published[year]

    # Tier 2: Projected (for specified vintage)
    if vintage and year in param.projected.get(vintage, {}):
        return param.projected[vintage][year]

    # Tier 3: Calculate from base + index
    indexing_rule = load_indexing_rule(param.indexing_rule)
    base_value = param.base[indexing_rule.base_year]
    index_ratio = get_index_ratio(
        indexing_rule.reference_section,
        from_year=indexing_rule.base_year,
        to_year=year,
        vintage=vintage
    )

    raw_value = base_value * index_ratio
    return apply_rounding(raw_value, indexing_rule.rounding)
```

## Separation of Historical vs Forecast

**Key insight**: Historical index values (from BLS) are fundamentally different from forecasts (from CBO/Fed). They should be:

1. **Stored separately**: `historical.yaml` vs `forecast/cbo_2024_06.yaml`
2. **Versioned differently**: Historical is immutable (revisions aside), forecasts are vintage-dated
3. **Used differently**: Historical for backfills, forecasts only for projections
4. **Uncertainty tracked**: Forecasts have confidence intervals, historical doesn't

```python
class IndexStore:
    def get_index(self, index_name: str, year: int, vintage: str = None) -> float:
        """Get index value with proper source selection."""

        historical = self.historical[index_name]

        if year in historical:
            # Historical: always use authoritative BLS value
            return historical[year]

        if year > current_year():
            # Future: must use forecast
            if vintage is None:
                vintage = self.latest_vintage(index_name)

            forecast = self.forecasts[index_name][vintage]
            if year not in forecast:
                raise ValueError(f"No forecast for {index_name} year {year} in vintage {vintage}")

            return forecast[year]

        # Current year but not yet published: use forecast
        forecast = self.forecasts[index_name][vintage or self.latest_vintage(index_name)]
        return forecast.get(year, self._interpolate(historical, year))
```

## Integration with RL Training

The agent should learn to:

1. **Identify indexing provisions**: Recognize "cost-of-living adjustment under section 1(f)(3)"
2. **Locate base years**: Find the year from which indexing starts
3. **Apply correct index**: CPI-U vs C-CPI-U depending on effective date
4. **Handle rounding**: "rounded down to nearest $10"

### Training Signal

When encoding a new provision, the agent should:

```cosilico
# Good: References the indexing rule, lets system handle calculation
variable earned_income_amount {
  reference "26 USC § 32(b)(2)(A)"
  indexing_rule "26 USC § 32(j)"  # Points to where indexing is defined

  # Base values (the statute's original amounts)
  base_values {
    year: 2015
    by_num_qualifying_children:
      0: 6580
      1: 9880
      2: 13870
      3: 13870
  }
}

# Bad: Hardcodes current indexed values
variable earned_income_amount {
  values {
    2024:
      0: 7840  # Where does this come from? How to update for 2025?
      1: 12390
      # ...
  }
}
```

### Reward Function

```python
def indexing_reward(generated: str, statute: Statute) -> float:
    score = 0.0

    # Does it reference the correct indexing provision?
    if identifies_indexing_section(generated, statute):
        score += 0.3

    # Does it specify base year and base values?
    if has_base_year_and_values(generated):
        score += 0.3

    # Does it avoid hardcoding current-year values?
    if not has_hardcoded_indexed_values(generated):
        score += 0.2

    # Does it correctly identify the underlying index?
    if identifies_correct_index(generated, statute):
        score += 0.2

    return score
```

## Example: Full EITC Indexing

```
us/irc/subtitle_a/chapter_1/subchapter_a/part_iv/subpart_c/§32/
├── j/                                    # Indexing rules live at §32(j)
│   ├── 1/
│   │   ├── A/indexing_rule.yaml         # The rule: "increased by CPI adjustment"
│   │   └── B/
│   │       ├── i/earned_income_adjustment.cosilico
│   │       ├── ii/joint_return_adjustment.cosilico
│   │       └── iii/disqualified_income_adjustment.cosilico
│   └── 2/
│       ├── A/rounding_rule.yaml          # Round to nearest $10
│       └── B/disqualified_income_rounding.yaml  # Round to nearest $50
│
├── b/2/A/amounts.yaml                    # References §32(j) for indexing
└── i/1/disqualified_income_limit.yaml   # References §32(j) for indexing
```

The system:
1. Reads `amounts.yaml`, sees `indexing_rule: ../j/1/indexing_rule`
2. Loads the indexing rule, which points to `§1(f)(3)` for the underlying index
3. Loads the index (C-CPI-U for 2018+), gets ratio from base year to target year
4. Applies rounding per `§32(j)(2)(A)`
5. Returns calculated value (or published value if available and more authoritative)

## API Design

```python
# Parameter resolution with indexing
class ParameterResolver:
    def __init__(self, index_store: IndexStore):
        self.index_store = index_store

    def get(
        self,
        path: str,
        year: int,
        tier: str = "auto",  # "published", "projected", "calculated", "auto"
        vintage: str = None,  # For forecasts: which vintage to use
        **indices  # num_qualifying_children=2, filing_status="JOINT"
    ) -> ParameterValue:
        """Resolve parameter with proper precedence and indexing."""
        ...

# Usage
resolver = ParameterResolver(index_store)

# Get 2024 EITC earned income amount (will use published Rev. Proc. value)
resolver.get("us/irc/.../§32/b/2/A/earned_income_amount", 2024, num_qualifying_children=1)
# Returns: ParameterValue(value=12390, source="Rev. Proc. 2023-34", tier="published")

# Get 2025 value (not yet published, uses projected)
resolver.get("us/irc/.../§32/b/2/A/earned_income_amount", 2025, num_qualifying_children=1)
# Returns: ParameterValue(value=12720, source="Calculated via §32(j)", tier="projected", vintage="2024-06")

# Get 2030 value (forecast-based calculation)
resolver.get("us/irc/.../§32/b/2/A/earned_income_amount", 2030, num_qualifying_children=1, vintage="2024-06")
# Returns: ParameterValue(value=14200, source="Calculated via §32(j)", tier="calculated", vintage="2024-06")
```

## See Also

- {doc}`parameters` - Parameter structure overview
- {doc}`versioning` - Bi-temporal versioning
- {doc}`../dsl/variables` - How indexed parameters are used in formulas
