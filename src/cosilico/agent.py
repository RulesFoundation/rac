"""Agentic training loop using Anthropic SDK with tool use.

This implements a closed-loop system where Claude:
1. Reads statutory text
2. Generates Cosilico DSL code
3. Executes it against test cases (via tool)
4. Gets structured feedback
5. Iterates until accuracy threshold is met
"""

import json
import os
from typing import Any

import anthropic

from .executor import Executor
from .oracles import MockOracle
from .scorer import FailureDiagnoser, Scorer
from .types import GeneratedCode, Statute, TestCase


# Tool definitions for Claude
TOOLS = [
    {
        "name": "execute_dsl",
        "description": """Execute Cosilico DSL code against test cases and return accuracy metrics.

Use this tool after generating DSL code to test if it correctly implements the statute.
The tool will:
1. Parse the DSL code
2. Execute it against all test cases
3. Return accuracy, pass/fail counts, and specific failure details

Call this tool with your generated DSL code to see how well it performs.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "dsl_code": {
                    "type": "string",
                    "description": "The Cosilico DSL code to execute"
                }
            },
            "required": ["dsl_code"]
        }
    },
    {
        "name": "submit_final_code",
        "description": """Submit your final DSL code when you've achieved the target accuracy or exhausted improvement options.

Only call this when:
1. You've achieved >= 95% accuracy, OR
2. You've made multiple attempts and accuracy has plateaued

Include the final code and a brief explanation of the implementation.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "dsl_code": {
                    "type": "string",
                    "description": "The final Cosilico DSL code"
                },
                "explanation": {
                    "type": "string",
                    "description": "Brief explanation of the implementation and any remaining issues"
                }
            },
            "required": ["dsl_code", "explanation"]
        }
    }
]


class AgentTrainingLoop:
    """Agentic training loop using Claude with tools."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_iterations: int = 10,
        target_accuracy: float = 0.95,
        api_key: str | None = None,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.target_accuracy = target_accuracy
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

        # Execution components
        self.executor = Executor()
        self.scorer = Scorer()
        self.diagnoser = FailureDiagnoser()

        # State
        self.test_cases: list[TestCase] = []
        self.iteration = 0
        self.best_code: str | None = None
        self.best_accuracy: float = 0.0

    def _build_system_prompt(self) -> str:
        return """You are an expert tax law encoder. Your task is to convert statutory text into executable Cosilico DSL code.

## Cosilico DSL Syntax

```
variable <name>:
  entity: TaxUnit | Person | Household
  period: Year | Month
  dtype: Money | Rate | Boolean | Integer
  label: "Description"
  citation: "Legal citation"

  references:
    <alias>: <path>  # Variable or parameter reference

  formula:
    <expression>
```

## Formula Syntax
- Arithmetic: +, -, *, /
- Comparison: ==, !=, <, <=, >, >=
- Logic: and, or, not
- Conditionals: if <cond> then <expr> else <expr>
- Functions: min(a, b), max(a, b), clip(x, lo, hi)
- Array indexing: param[index_variable]

## Example

```cosilico
variable eitc_phase_in_credit:
  entity: TaxUnit
  period: Year
  dtype: Money
  label: "EITC phase-in credit"
  citation: "26 USC § 32(a)(1)"

  references:
    earned_income: us/irs/income/earned_income
    n_qualifying_children: us/irs/eitc/n_qualifying_children
    phase_in_rate: param.irs.eitc.phase_in_rate
    earned_income_amount: param.irs.eitc.earned_income_amount

  formula:
    min(earned_income, earned_income_amount[n_qualifying_children]) * phase_in_rate[n_qualifying_children]
```

## Parameters Available

For EITC (2024 values):
- param.irs.eitc.phase_in_rate: {0: 0.0765, 1: 0.34, 2: 0.40, 3: 0.45}
- param.irs.eitc.earned_income_amount: {0: 7840, 1: 11750, 2: 16510, 3: 16510}
- param.irs.eitc.max_credit: {0: 600, 1: 3995, 2: 6604, 3: 7430}

## Your Task

1. Read the statutory text carefully
2. Generate DSL code that implements the rules
3. Use the execute_dsl tool to test your code
4. Analyze failures and iterate
5. When you reach 95%+ accuracy (or have exhausted options), use submit_final_code

Be precise with the formula - small errors in rates or thresholds cause test failures."""

    def _build_user_prompt(self, statute: Statute, test_cases: list[TestCase]) -> str:
        # Show a sample of test cases
        sample_cases = test_cases[:5]
        cases_str = "\n".join([
            f"- earned_income=${tc.inputs.get('earned_income', 0)}, "
            f"n_children={tc.inputs.get('n_children', 0)} → "
            f"expected EITC=${tc.expected.get('eitc', tc.expected.get('eitc_phase_in_credit', 0)):.2f}"
            for tc in sample_cases
        ])

        return f"""## Statutory Text to Encode

**Citation:** {statute.citation}

{statute.text}

## Test Cases (sample of {len(test_cases)} total)

{cases_str}

## Instructions

1. Analyze the statutory text
2. Write Cosilico DSL code that implements it
3. Use the `execute_dsl` tool to test your implementation
4. Iterate based on failure feedback until you reach 95%+ accuracy
5. Use `submit_final_code` when done

Start by generating your initial DSL code and testing it."""

    def _handle_tool_call(self, tool_name: str, tool_input: dict) -> str:
        """Handle a tool call from Claude."""
        if tool_name == "execute_dsl":
            return self._execute_dsl(tool_input["dsl_code"])
        elif tool_name == "submit_final_code":
            return self._submit_final(tool_input["dsl_code"], tool_input.get("explanation", ""))
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def _execute_dsl(self, dsl_code: str) -> str:
        """Execute DSL code against test cases."""
        self.iteration += 1

        # Create a GeneratedCode object
        code = GeneratedCode(
            source=dsl_code,
            citation="test",
            iteration=self.iteration
        )

        # Execute
        results = self.executor.execute(code, self.test_cases)

        # Score
        score = self.scorer.score(results)

        # Track best
        if score.accuracy > self.best_accuracy:
            self.best_accuracy = score.accuracy
            self.best_code = dsl_code

        # Diagnose failures
        failures = self.diagnoser.diagnose(results)

        # Build response
        response = {
            "iteration": self.iteration,
            "accuracy": f"{score.accuracy:.1%}",
            "passed": int(score.accuracy * score.n_cases),
            "total": score.n_cases,
            "target": f"{self.target_accuracy:.0%}",
            "mean_absolute_error": f"${score.mean_absolute_error:.2f}",
        }

        if score.accuracy >= self.target_accuracy:
            response["status"] = "SUCCESS - Target accuracy reached!"
            response["suggestion"] = "Use submit_final_code to complete."
        else:
            response["status"] = "NEEDS_IMPROVEMENT"
            response["failures"] = [
                {
                    "type": f.type,
                    "message": f.message[:100],
                    "expected": f.expected,
                    "actual": f.actual
                }
                for f in failures[:5]
            ]
            response["suggestion"] = "Analyze the failures and adjust your formula or parameters."

        return json.dumps(response, indent=2)

    def _submit_final(self, dsl_code: str, explanation: str) -> str:
        """Handle final code submission."""
        # One final execution to get accurate metrics
        code = GeneratedCode(source=dsl_code, citation="final", iteration=self.iteration)
        results = self.executor.execute(code, self.test_cases)
        score = self.scorer.score(results)

        return json.dumps({
            "status": "SUBMITTED",
            "final_accuracy": f"{score.accuracy:.1%}",
            "iterations": self.iteration,
            "explanation": explanation
        })

    def train(
        self,
        statute: Statute,
        test_cases: list[TestCase],
        verbose: bool = True
    ) -> dict[str, Any]:
        """Run the agentic training loop.

        Returns dict with:
        - success: bool
        - final_code: str
        - final_accuracy: float
        - iterations: int
        - conversation: list of messages
        """
        self.test_cases = test_cases
        self.iteration = 0
        self.best_code = None
        self.best_accuracy = 0.0

        # Initialize conversation
        messages = [
            {"role": "user", "content": self._build_user_prompt(statute, test_cases)}
        ]

        if verbose:
            print(f"Starting agentic training loop for: {statute.citation}")
            print(f"Test cases: {len(test_cases)}, Target: {self.target_accuracy:.0%}")
            print("-" * 60)

        final_result = None

        # Main agentic loop
        for turn in range(self.max_iterations * 2):  # Allow multiple tool calls per iteration
            # Call Claude
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self._build_system_prompt(),
                tools=TOOLS,
                messages=messages
            )

            # Process response
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Check for tool use
            tool_uses = [block for block in assistant_content if block.type == "tool_use"]

            if not tool_uses:
                # No tool call - Claude is done or needs prompting
                if verbose:
                    for block in assistant_content:
                        if hasattr(block, "text"):
                            print(f"\n[Claude]: {block.text[:300]}...")
                break

            # Handle each tool call
            tool_results = []
            for tool_use in tool_uses:
                if verbose:
                    print(f"\n[Tool: {tool_use.name}]")

                result = self._handle_tool_call(tool_use.name, tool_use.input)

                if verbose:
                    # Parse and display key info
                    try:
                        result_data = json.loads(result)
                        if "accuracy" in result_data:
                            print(f"  Accuracy: {result_data['accuracy']}")
                        if "status" in result_data:
                            print(f"  Status: {result_data['status']}")
                        if "failures" in result_data and result_data["failures"]:
                            print(f"  Failures ({len(result_data['failures'])}):")
                            for f in result_data["failures"][:2]:
                                print(f"    - {f['message'][:60]}...")
                    except json.JSONDecodeError:
                        print(f"  {result[:100]}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result
                })

                # Check if this was final submission
                if tool_use.name == "submit_final_code":
                    final_result = json.loads(result)
                    break

            if final_result:
                break

            # Add tool results to conversation
            messages.append({"role": "user", "content": tool_results})

            # Check iteration limit
            if self.iteration >= self.max_iterations:
                if verbose:
                    print(f"\nMax iterations ({self.max_iterations}) reached.")
                break

        # Build final result
        if final_result:
            success = "SUCCESS" in final_result.get("status", "")
            return {
                "success": self.best_accuracy >= self.target_accuracy,
                "final_code": self.best_code,
                "final_accuracy": self.best_accuracy,
                "iterations": self.iteration,
                "submitted": True
            }
        else:
            return {
                "success": self.best_accuracy >= self.target_accuracy,
                "final_code": self.best_code,
                "final_accuracy": self.best_accuracy,
                "iterations": self.iteration,
                "submitted": False
            }


def create_eitc_test_cases() -> list[TestCase]:
    """Create test cases for EITC using mock oracle."""
    oracle = MockOracle()
    cases = []

    # Test various income levels and child counts
    incomes = [0, 1000, 5000, 7840, 10000, 11750, 15000, 16510, 20000]
    for i, income in enumerate(incomes):
        for n_children in [0, 1, 2, 3]:
            inputs = {
                "earned_income": income,
                "filing_status": "SINGLE",
                "n_children": n_children,
                "n_qualifying_children": n_children,
            }
            expected = oracle.evaluate(inputs)
            cases.append(TestCase(
                id=f"case_{i}_{n_children}",
                inputs=inputs,
                expected=expected,
            ))
    return cases


# CLI entry point
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cosilico Agentic Training Loop")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model to use")
    parser.add_argument("--max-iterations", type=int, default=5, help="Max iterations")
    parser.add_argument("--target-accuracy", type=float, default=0.95, help="Target accuracy")
    args = parser.parse_args()

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        print("Set with: export ANTHROPIC_API_KEY=your_key")
        return

    # Create statute and test cases
    statute = Statute(
        citation="26 USC § 32(a)(1)",
        text="""
(a) Allowance of credit
    (1) In general
    In the case of an eligible individual, there shall be allowed as a credit
    against the tax imposed by this subtitle for the taxable year an amount
    equal to the credit percentage of so much of the taxpayer's earned income
    for the taxable year as does not exceed the earned income amount.
        """.strip(),
        jurisdiction="us",
    )

    test_cases = create_eitc_test_cases()

    print("=" * 60)
    print("Cosilico Agentic Training Loop")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Statute: {statute.citation}")
    print(f"Test cases: {len(test_cases)}")
    print(f"Target accuracy: {args.target_accuracy:.0%}")
    print("=" * 60)

    # Run training
    loop = AgentTrainingLoop(
        model=args.model,
        max_iterations=args.max_iterations,
        target_accuracy=args.target_accuracy,
    )

    result = loop.train(statute, test_cases, verbose=True)

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Success: {result['success']}")
    print(f"Final accuracy: {result['final_accuracy']:.1%}")
    print(f"Iterations: {result['iterations']}")
    print(f"Submitted: {result['submitted']}")

    if result["final_code"]:
        print("\nFinal code:")
        print("-" * 40)
        print(result["final_code"])


if __name__ == "__main__":
    main()
