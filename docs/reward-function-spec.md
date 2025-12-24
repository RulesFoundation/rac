# Reward Function Technical Specification

## Overview

This document provides the technical specification for the policy encoding reward function, designed to validate AI-encoded tax and benefit parameters against authoritative oracles.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     REWARD FUNCTION SYSTEM                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐         ┌──────────────────┐             │
│  │   Test Cases    │────────>│ Oracle Manager   │             │
│  │  - Inputs       │         │  - PolicyEngine  │             │
│  │  - Expected     │         │  - TAXSIM        │             │
│  └─────────────────┘         │  - IRS Tables    │             │
│                              └──────────────────┘             │
│                                       │                        │
│                                       v                        │
│                              ┌──────────────────┐             │
│                              │  Reward Computer │             │
│                              │  - Tolerance     │             │
│                              │  - Partial Credit│             │
│                              │  - Consensus     │             │
│                              └──────────────────┘             │
│                                       │                        │
│                                       v                        │
│  ┌─────────────────────────────────────────────┐             │
│  │            RewardResult                      │             │
│  │  - reward: float (0-1)                      │             │
│  │  - accuracy: float (0-1)                    │             │
│  │  - diagnostics: dict                        │             │
│  └─────────────────────────────────────────────┘             │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## Data Structures

### Input Types

```python
# Test case structure
TestCase = {
    "inputs": {
        "earned_income": int | float,
        "filing_status": str,  # "SINGLE", "JOINT", etc.
        "eitc_qualifying_children_count": int,
        # ... other inputs
    },
    "expected": {
        "eitc": float,  # or other variable
    }
}

# Encoded parameters (from YAML or DSL)
EncodedParams = dict[str, Any]

# Structural metadata
StructuralMetadata = {
    "parses": bool,
    "uses_valid_primitives": bool,
    "has_required_metadata": bool,
    "follows_naming_conventions": bool,
    "references_valid_dependencies": bool,
}
```

### Output Types

```python
@dataclass
class RewardResult:
    reward: float              # 0-1 scalar reward
    accuracy: float            # Percentage of exact matches
    oracle_results: dict[str, list[float]]
    mean_error: float          # Mean absolute error
    max_error: float           # Maximum absolute error
    n_cases: int              # Total test cases
    n_passed: int             # Cases within tolerance
    n_failed: int             # Cases outside tolerance
    diagnostics: dict         # Detailed per-case results

@dataclass
class OracleComparison:
    test_case_id: str
    expected: float
    actual: float
    oracle_values: dict[str, float]
    match: bool
    relative_error: float
    absolute_error: float
    consensus: bool
```

## Algorithm Specifications

### 1. Encoding Reward Function

**Input**: `(encoded_params, variable, test_cases, year)`

**Output**: `RewardResult`

**Algorithm**:

```
FOR each test_case IN test_cases:
    inputs = test_case.inputs
    expected = test_case.expected[variable]

    # Query all oracles
    oracle_values = {}
    FOR each oracle IN oracles:
        IF oracle.supports(variable, year):
            value = oracle.calculate(inputs, variable, year)
            IF value is not None:
                oracle_values[oracle.name] = value

    # Use highest-priority oracle
    IF oracle_values is empty:
        actual = expected  # Fallback
    ELSE:
        actual = oracle_values[first_by_priority]

    # Check match
    match = is_within_tolerance(expected, actual)
    comparisons.append(OracleComparison(...))

# Compute aggregate metrics
accuracy = count(match=True) / len(comparisons)
mean_error = mean([c.absolute_error for c in comparisons if not c.match])
max_error = max([c.absolute_error for c in comparisons if not c.match])

# Compute reward
IF partial_credit:
    reward = sum([partial_credit_score(c) for c in comparisons]) / len(comparisons)
ELSE:
    reward = accuracy

RETURN RewardResult(reward, accuracy, ...)
```

### 2. Partial Credit Function

**Input**: `relative_error: float`

**Output**: `credit: float` (0-1)

**Algorithm**:

```python
def partial_credit(relative_error: float) -> float:
    if relative_error < 0.001:   # <0.1%
        return 1.00
    elif relative_error < 0.01:  # <1%
        return 0.95
    elif relative_error < 0.05:  # <5%
        return 0.80
    elif relative_error < 0.10:  # <10%
        return 0.60
    elif relative_error < 0.25:  # <25%
        return 0.30
    else:
        return 0.00
```

**Rationale**: Exponential decay encourages getting close first, then refining.

### 3. Tolerance Checking

**Input**: `(expected: float, actual: float)`

**Output**: `match: bool`

**Algorithm**:

```python
def is_within_tolerance(expected: float, actual: float) -> bool:
    # Special case: zero expected
    if expected == 0:
        return abs(actual) <= tolerance_absolute

    # Check both tolerances
    abs_error = abs(actual - expected)
    rel_error = abs_error / abs(expected)

    return (abs_error <= tolerance_absolute) or (rel_error <= tolerance_relative)
```

**Default values**:
- `tolerance_absolute = 1.0` (dollars)
- `tolerance_relative = 0.01` (1%)

### 4. Oracle Consensus

**Input**: `oracle_values: list[float]`

**Output**: `consensus: bool`

**Algorithm**:

```python
def check_consensus(values: list[float]) -> bool:
    if len(values) <= 1:
        return True

    # All pairs must agree within tolerance
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if not is_within_tolerance(values[i], values[j]):
                return False

    return True
```

### 5. Combined Reward (Curriculum Learning)

**Input**: `(structural_score, semantic_reward, alpha)`

**Output**: `combined_reward: float`

**Algorithm**:

```python
def combined_reward(structural: float, semantic: float, alpha: float) -> float:
    return alpha * structural + (1 - alpha) * semantic
```

**Alpha schedule**:
- Iteration 1-3: α = 0.5 (equal weight)
- Iteration 4-6: α = 0.3 (more semantic)
- Iteration 7-9: α = 0.1 (mostly semantic)
- Iteration 10+: α = 0.0 (pure semantic)

## Oracle Specifications

### Base Interface

```python
class Oracle(ABC):
    name: str
    priority: int  # 1=highest (ground truth), 2=reference, 3+=supplementary

    @abstractmethod
    def supports(self, variable: str, year: int) -> bool:
        """Check if oracle can calculate this variable/year."""

    @abstractmethod
    def calculate(self, inputs: dict, variable: str, year: int) -> float | None:
        """Calculate the variable value."""
```

### PolicyEngine Oracle

**Implementation**: Uses `policyengine-us` Python package

**Supported Variables**:
- EITC (`eitc`)
- CTC (`ctc`)
- Income tax (`income_tax`)
- State income tax (`state_income_tax`)
- SNAP (`snap`)
- Medicaid (`medicaid`)

**Supported Years**: 2015-present

**Input Mapping**:
```python
{
    "earned_income" -> "employment_income",
    "filing_status" -> household structure,
    "eitc_qualifying_children_count" -> add children to household,
    "state" -> "state_name",
}
```

**Priority**: 2 (reference)

### TAXSIM Oracle

**Implementation**: Uses TAXSIM-35 executable via subprocess

**Supported Variables**:
- Federal income tax (`fiitax`)
- State income tax (`siitax`)
- FICA (`fica`)
- AGI (`v10`)
- CTC (`v22`)
- EITC (`v25`)

**Supported Years**: 1960-2023

**Input Mapping**:
```python
{
    "earned_income" -> "pwages",
    "filing_status" -> "mstat",
    "eitc_qualifying_children_count" -> "depx",
}
```

**Priority**: 2 (reference)

**Limitations**:
- Only federal taxes (no state-specific benefits)
- Requires local executable installation

### IRS Table Oracle

**Implementation**: Lookup in YAML test case files

**Supported Variables**: Any with official IRS examples

**Supported Years**: All (year-specific test cases)

**Input Matching**: Exact match required on all inputs

**Priority**: 1 (ground truth - official IRS examples)

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Structural reward | O(1) | Metadata checks only |
| Oracle query | O(n) | n = number of oracles |
| Single test case | O(m) | m = number of oracles |
| Full evaluation | O(n × m) | n = test cases, m = oracles |

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Test cases | O(n) | n = number of cases |
| Oracle results | O(n × m) | Stored for diagnostics |
| Comparisons | O(n) | Per-case metadata |

### Typical Runtimes

| Operation | Time | Notes |
|-----------|------|-------|
| PolicyEngine query | 50-100ms | Per test case |
| TAXSIM query | 200-500ms | Subprocess overhead |
| IRS table lookup | <1ms | Dict lookup |
| Structural check | <1ms | Metadata only |
| Full evaluation (10 cases) | 1-5s | Dominated by oracles |

## Error Handling

### Oracle Failures

**Behavior**: If oracle fails, skip that oracle and use next priority

**Example**:
```python
oracle_values = {}
for oracle in oracles:
    try:
        value = oracle.calculate(inputs, variable, year)
        if value is not None:
            oracle_values[oracle.name] = value
    except Exception as e:
        log.warning(f"Oracle {oracle.name} failed: {e}")
        continue

if not oracle_values:
    # Use expected value as fallback
    actual = expected
```

### Invalid Inputs

**Behavior**: Oracle returns `None`, test case is skipped

**Diagnostics**: Logged in result metadata

### Missing Test Cases

**Behavior**: Return zero reward with appropriate error

```python
if not test_cases:
    return RewardResult(reward=0.0, accuracy=0.0, n_cases=0, ...)
```

## Validation Requirements

### Input Validation

Test cases must have:
- `inputs` dict with required fields
- `expected` dict with target variable
- Valid data types (int/float for numbers, str for enums)

### Output Validation

Oracles must return:
- Numeric value (int or float)
- `None` if unsupported
- Raise exception on error

### Tolerance Constraints

- `tolerance_absolute >= 0`
- `0 <= tolerance_relative <= 1`
- At least one tolerance must be > 0

## Testing Requirements

### Unit Tests

**Coverage**: >90% line coverage

**Test Categories**:
1. Perfect matches (reward = 1.0)
2. Complete failures (reward = 0.0)
3. Partial credit (0 < reward < 1)
4. Tolerance boundaries
5. Oracle consensus
6. Error cases (empty tests, invalid inputs)
7. Edge cases (zero values, very large/small numbers)

### Integration Tests

**Requirements**:
1. Real PolicyEngine oracle
2. Real TAXSIM oracle (if available)
3. IRS official test cases
4. Cross-oracle validation

### Performance Tests

**Benchmarks**:
- 100 test cases in < 10 seconds (with 2 oracles)
- Memory usage < 100MB for 1000 test cases
- No memory leaks over 10,000 evaluations

## Security Considerations

### Input Sanitization

- Validate all numeric inputs are finite (not inf/nan)
- Validate string inputs are from allowed enum values
- Limit test case count to prevent DoS

### Oracle Sandboxing

- TAXSIM subprocess runs with timeout
- No shell injection in TAXSIM input files
- Temporary files cleaned up after use

### Data Privacy

- No PII in test cases (use synthetic data)
- Oracle queries logged but not persisted
- Diagnostic output sanitized

## Configuration

### Environment Variables

```bash
# Optional: override default oracle paths
TAXSIM_PATH=/path/to/taxsim35-osx.exe
IRS_TEST_CASES_PATH=/path/to/irs_tests.yaml

# Optional: override default tolerances
REWARD_TOLERANCE_ABSOLUTE=1.0
REWARD_TOLERANCE_RELATIVE=0.01
```

### Config File Format

```yaml
# reward_config.yaml
oracles:
  - name: PolicyEngine
    priority: 2
    enabled: true

  - name: TAXSIM
    priority: 2
    enabled: true
    path: /usr/local/bin/taxsim35-osx.exe

  - name: IRS
    priority: 1
    enabled: true
    test_cases: /data/irs_official_tests.yaml

tolerance:
  absolute: 1.0
  relative: 0.01

partial_credit:
  enabled: true
  schedule:
    - error: 0.001
      credit: 1.00
    - error: 0.01
      credit: 0.95
    # ... etc
```

## API Reference

### Main Functions

```python
# Create reward function
reward_fn = EncodingRewardFunction(
    oracles: list[Oracle],
    tolerance_absolute: float = 1.0,
    tolerance_relative: float = 0.01,
    partial_credit: bool = True,
)

# Evaluate
result = reward_fn.evaluate(
    encoded_params: dict,
    variable: str,
    test_cases: list[dict],
    year: int = 2024,
) -> RewardResult

# Access results
result.reward           # Scalar reward (0-1)
result.accuracy         # Exact match percentage
result.diagnostics      # Detailed breakdown
```

### Helper Functions

```python
# Create oracles
oracle = PolicyEngineOracle()
oracle = TaxsimOracle(taxsim_path="/path/to/exe")
oracle = IRSTableOracle(test_cases_path="/path/to/yaml")

# Structural reward
structural_fn = StructuralRewardFunction()
score = structural_fn.evaluate(code, metadata)

# Combined reward
combined_fn = CombinedRewardFunction(structural_fn, semantic_fn, alpha=0.3)
reward, result = combined_fn.evaluate(...)

# Curriculum learning
combined_fn.set_alpha(0.1)  # Adjust over time
```

## Extension Points

### Custom Oracles

Implement `Oracle` base class:

```python
class MyCustomOracle(Oracle):
    name = "MyOracle"
    priority = 3

    def supports(self, variable: str, year: int) -> bool:
        return variable in ["eitc", "ctc"]

    def calculate(self, inputs: dict, variable: str, year: int) -> float | None:
        # Your implementation
        return calculated_value
```

### Custom Partial Credit

Override `_compute_partial_credit_reward`:

```python
class CustomRewardFunction(EncodingRewardFunction):
    def _compute_partial_credit_reward(self, comparisons):
        # Custom partial credit logic
        return custom_reward
```

### Custom Tolerance

Override `_is_match`:

```python
class CustomRewardFunction(EncodingRewardFunction):
    def _is_match(self, expected: float, actual: float) -> bool:
        # Custom tolerance logic
        return custom_match
```

## Version History

- **v1.0.0** (2024-12-23): Initial implementation
  - EncodingRewardFunction
  - PolicyEngineOracle, TaxsimOracle, IRSTableOracle
  - Partial credit system
  - Tolerance handling
  - Oracle consensus

## References

1. PolicyEngine-US Documentation: https://policyengine.github.io/policyengine-us/
2. TAXSIM-35 User Guide: https://taxsim.nber.org/taxsim35/
3. IRS Publication 596 (EITC): https://www.irs.gov/pub/irs-pdf/p596.pdf
4. Reward Shaping in RL: Ng et al., "Policy Invariance Under Reward Transformations"
