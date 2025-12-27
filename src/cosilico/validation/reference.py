"""Reference validation against PolicyEngine.

Used ONCE per variable to establish correctness, then cross-compilation
tests ensure all targets match.
"""

import json
import subprocess
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..dsl_parser import parse_dsl
from ..js_generator import generate_js
from ..oracles import PolicyEngineOracle, MockOracle


@dataclass
class ValidationResult:
    """Result of validating against a reference implementation."""

    passed: bool
    cosilico_value: float
    reference_value: float
    variable: str
    inputs: dict[str, Any]
    reference_name: str = "policyengine"
    tolerance: float = 0.01  # 1% relative tolerance by default

    @property
    def difference(self) -> float:
        """Absolute difference between Cosilico and reference."""
        return self.rac_value - self.reference_value

    @property
    def relative_error(self) -> float:
        """Relative error (0 if reference is 0)."""
        if self.reference_value == 0:
            return 0.0 if self.rac_value == 0 else float("inf")
        return abs(self.difference) / abs(self.reference_value)


@dataclass
class ValidationReport:
    """Summary report of validation results."""

    variable: str
    reference: str
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def passed_cases(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed_cases / self.total_cases if self.total_cases > 0 else 0.0

    @property
    def max_error(self) -> float:
        if not self.results:
            return 0.0
        return max(r.relative_error for r in self.results)

    @property
    def mean_error(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.relative_error for r in self.results) / len(self.results)


class ReferenceValidator:
    """Validates Cosilico output against PolicyEngine or other references."""

    def __init__(
        self,
        year: int = 2024,
        oracle: PolicyEngineOracle | MockOracle | None = None,
        tolerance: float = 0.01,
    ):
        self.year = year
        self.tolerance = tolerance

        if oracle is None:
            try:
                self.oracle = PolicyEngineOracle(year=year)
            except ImportError:
                self.oracle = MockOracle()
        else:
            self.oracle = oracle

    def validate_single(
        self,
        dsl_code: str,
        variable: str,
        inputs: dict[str, Any],
        reference_variable: str | None = None,
    ) -> ValidationResult:
        """Validate a single case against the reference.

        Args:
            dsl_code: Cosilico DSL source
            variable: Name of variable in DSL
            inputs: Input values (simplified format)
            reference_variable: PolicyEngine variable name (default: same as variable)

        Returns:
            ValidationResult with comparison
        """
        reference_variable = reference_variable or variable

        # Execute Cosilico DSL
        cosilico_value = self._execute_dsl(dsl_code, variable, inputs)

        # Execute PolicyEngine
        pe_results = self.oracle.evaluate(inputs)
        reference_value = pe_results.get(reference_variable, 0.0)

        # Compare
        if reference_value == 0:
            passed = abs(cosilico_value) <= self.tolerance
        else:
            relative_error = abs(cosilico_value - reference_value) / abs(reference_value)
            passed = relative_error <= self.tolerance

        return ValidationResult(
            passed=passed,
            cosilico_value=cosilico_value,
            reference_value=reference_value,
            variable=variable,
            inputs=inputs,
            tolerance=self.tolerance,
        )

    def validate_batch(
        self,
        dsl_code: str,
        variable: str,
        test_cases: list[dict[str, Any]],
        reference_variable: str | None = None,
    ) -> list[ValidationResult]:
        """Validate multiple test cases."""
        return [
            self.validate_single(dsl_code, variable, inputs, reference_variable)
            for inputs in test_cases
        ]

    def generate_report(
        self,
        dsl_code: str,
        variable: str,
        test_cases: list[dict[str, Any]],
        reference_variable: str | None = None,
    ) -> ValidationReport:
        """Generate a comprehensive validation report."""
        results = self.validate_batch(
            dsl_code, variable, test_cases, reference_variable
        )
        return ValidationReport(
            variable=variable,
            reference="policyengine",
            results=results,
        )

    def _execute_dsl(
        self,
        dsl_code: str,
        variable: str,
        inputs: dict[str, Any],
    ) -> float:
        """Execute DSL code using JS target."""
        module = parse_dsl(dsl_code)
        js_code = generate_js(module)

        wrapper = f"""
const inputs = {json.dumps(inputs)};
const params = {{}};

{js_code}

console.log({variable}(inputs, params));
"""
        result = subprocess.run(
            ["node", "-e", wrapper],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            raise RuntimeError(f"JS execution failed: {result.stderr}")

        output = result.stdout.strip()
        if output.lower() == "true":
            return 1.0
        if output.lower() == "false":
            return 0.0
        return float(output)


def validate_against_policyengine(
    dsl_code: str,
    variable: str,
    test_cases: list[dict[str, Any]],
    reference_variable: str | None = None,
    tolerance: float = 0.01,
    year: int = 2024,
) -> ValidationReport:
    """Convenience function to validate against PolicyEngine.

    Args:
        dsl_code: Cosilico DSL source
        variable: Variable name in DSL
        test_cases: List of input dictionaries
        reference_variable: PolicyEngine variable (default: same name)
        tolerance: Relative tolerance for passing (default: 1%)
        year: Tax year for PolicyEngine

    Returns:
        ValidationReport with all results
    """
    validator = ReferenceValidator(year=year, tolerance=tolerance)
    return validator.generate_report(
        dsl_code, variable, test_cases, reference_variable
    )
