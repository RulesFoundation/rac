"""Agentic training loop for Cosilico DSL generation.

This is the DSL-native version of the agent that generates proper
Cosilico DSL code instead of Python functions.
"""

import json
import os
from typing import Any

import anthropic

from .dsl_executor import DSLExecutor, get_default_parameters
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


DSL_SYSTEM_PROMPT = """You are an expert tax law encoder. Your task is to convert statutory text into Cosilico DSL code.

## Cosilico DSL Overview

Cosilico DSL is a purpose-built language for encoding tax and benefit rules. It's designed to be:
- **Safe**: Pure functional, no side effects
- **Traceable**: Every rule links to legal citations
- **AI-native**: Structured grammar that's easy to generate correctly

## DSL Syntax

### Variable Definition

```cosilico
variable <name> {
  entity <EntityType>           # Person, TaxUnit, Household
  period <PeriodType>           # Year, Month
  dtype <DataType>              # Money, Rate, Count, Bool
  reference "<legal citation>"  # Required - cite the statute

  formula {
    let <var> = <expression>
    return <expression>
  }
}
```

### Available Data Types
- `Money` - Currency amounts (e.g., $1234.56)
- `Rate` - Decimal rates (e.g., 0.0765)
- `Count` - Non-negative integers
- `Bool` - true/false

### Expressions

**Arithmetic:** `+`, `-`, `*`, `/`
**Comparison:** `==`, `!=`, `<`, `>`, `<=`, `>=`
**Logical:** `and`, `or`, `not`
**Functions:** `min(a, b)`, `max(a, b)`, `abs(x)`, `clamp(x, lo, hi)`

**Conditionals:**
```cosilico
if condition then value_if_true else value_if_false

match {
  case condition1 => value1
  case condition2 => value2
  else => default_value
}
```

### Variable References

```cosilico
variable(earned_income)                    # Reference another variable
parameter(gov.irs.eitc.phase_in_rate)     # Reference a parameter
parameter(gov.irs.eitc.rate[n_children])  # Indexed parameter
```

### Input Variables

These variables are provided as inputs (no formula needed):
- `earned_income` - Employment/self-employment income
- `n_qualifying_children` or `n_children` - Number of qualifying children
- `filing_status` - SINGLE, JOINT, HEAD_OF_HOUSEHOLD, MARRIED_FILING_SEPARATELY
- `agi` or `adjusted_gross_income` - Adjusted gross income
- Other inputs as specified in test cases

## Example: EITC Phase-In Credit

```cosilico
module us.federal.irs.credits.eitc
version "2024.1"
jurisdiction us

variable eitc_phase_in {
  entity TaxUnit
  period Year
  dtype Money
  reference "26 USC § 32(a)(1)"

  formula {
    let earned = variable(earned_income)
    let n_children = variable(n_qualifying_children)

    # Phase-in rates by number of children (2024)
    let rate = match {
      case n_children == 0 => 0.0765
      case n_children == 1 => 0.34
      case n_children == 2 => 0.40
      else => 0.45
    }

    # Earned income amounts (caps)
    let cap = match {
      case n_children == 0 => 7840
      case n_children == 1 => 11750
      else => 16510
    }

    return min(earned, cap) * rate
  }
}
```

## Important Rules

1. **Always include metadata**: module, entity, period, dtype, reference
2. **Reference inputs correctly**: Use `variable(input_name)` for inputs
3. **Use match expressions** for multi-way conditionals
4. **Cite the statute** in the reference field
5. **Return the computed value** at the end of the formula

## Your Task

1. Read the statutory text carefully
2. Generate Cosilico DSL code that implements it
3. Use the `execute_dsl` tool to test your implementation
4. Analyze failures and adjust rates/thresholds/logic
5. When you reach 95%+ accuracy, use `submit_final_code`

Be precise with parameters - small errors in rates or thresholds cause test failures."""


class DSLAgentTrainingLoop:
    """Agentic training loop for DSL generation."""

    def __init__(
        self,
        model: str = "claude-opus-4-5-20251101",
        max_iterations: int = 10,
        target_accuracy: float = 0.95,
        api_key: str | None = None,
        parameters: dict | None = None,
    ):
        self.model = model
        self.max_iterations = max_iterations
        self.target_accuracy = target_accuracy
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

        # Execution components - use DSL executor
        self.executor = DSLExecutor(parameters=parameters or get_default_parameters())
        self.scorer = Scorer()
        self.diagnoser = FailureDiagnoser()

        # State
        self.test_cases: list[TestCase] = []
        self.iteration = 0
        self.best_code: str | None = None
        self.best_accuracy: float = 0.0

        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Trajectory logging
        self.trajectory: list[dict] = []
        self.conversation_log: list[dict] = []

    def get_cost_estimate(self) -> dict:
        """Estimate API cost based on token usage."""
        if "opus" in self.model.lower():
            input_rate = 15.0 / 1_000_000
            output_rate = 75.0 / 1_000_000
        else:  # sonnet
            input_rate = 3.0 / 1_000_000
            output_rate = 15.0 / 1_000_000

        input_cost = self.total_input_tokens * input_rate
        output_cost = self.total_output_tokens * output_rate
        total_cost = input_cost + output_cost

        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": total_cost,
            "model": self.model
        }

    def _build_user_prompt(self, statute: Statute, test_cases: list[TestCase]) -> str:
        """Build the user prompt with statute and test cases."""
        sample_cases = test_cases[:5]

        # Format test cases based on available inputs
        cases_lines = []
        for tc in sample_cases:
            parts = []
            if "earned_income" in tc.inputs:
                parts.append(f"earned_income=${tc.inputs['earned_income']}")
            if "n_children" in tc.inputs:
                parts.append(f"n_children={tc.inputs['n_children']}")
            elif "n_qualifying_children" in tc.inputs:
                parts.append(f"n_children={tc.inputs['n_qualifying_children']}")
            if "filing_status" in tc.inputs:
                parts.append(f"filing_status={tc.inputs['filing_status']}")
            if "agi" in tc.inputs:
                parts.append(f"agi=${tc.inputs['agi']}")

            # Get expected value
            exp_val = list(tc.expected.values())[0] if tc.expected else 0
            exp_key = list(tc.expected.keys())[0] if tc.expected else "output"

            cases_lines.append(f"- {', '.join(parts)} → {exp_key}=${exp_val:.2f}")

        cases_str = "\n".join(cases_lines)

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

        # Execute using DSL executor
        results = self.executor.execute(dsl_code, self.test_cases)

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

        # Log to trajectory
        self.trajectory.append({
            "iteration": self.iteration,
            "code": dsl_code,
            "accuracy": score.accuracy,
            "passed": int(score.accuracy * score.n_cases),
            "total": score.n_cases,
            "mean_absolute_error": score.mean_absolute_error,
            "failures": [
                {
                    "type": f.type,
                    "message": f.message,
                    "expected": f.expected,
                    "actual": f.actual,
                }
                for f in failures
            ],
            "is_best": score.accuracy >= self.best_accuracy,
        })

        return json.dumps(response, indent=2)

    def _submit_final(self, dsl_code: str, explanation: str) -> str:
        """Handle final code submission."""
        results = self.executor.execute(dsl_code, self.test_cases)
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
        """Run the agentic training loop."""
        self.test_cases = test_cases
        self.iteration = 0
        self.best_code = None
        self.best_accuracy = 0.0
        self.trajectory = []
        self.conversation_log = []

        # Initialize conversation
        messages = [
            {"role": "user", "content": self._build_user_prompt(statute, test_cases)}
        ]

        if verbose:
            print(f"Starting DSL training loop for: {statute.citation}")
            print(f"Test cases: {len(test_cases)}, Target: {self.target_accuracy:.0%}")
            print("-" * 60)

        final_result = None

        # Main agentic loop
        for turn in range(self.max_iterations * 2):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=DSL_SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages
            )

            # Track token usage
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens

            # Process response
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # Log assistant response
            assistant_text = ""
            for block in assistant_content:
                if hasattr(block, "text"):
                    assistant_text += block.text
            self.conversation_log.append({
                "turn": turn,
                "role": "assistant",
                "text": assistant_text,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            })

            # Check for tool use
            tool_uses = [block for block in assistant_content if block.type == "tool_use"]

            if not tool_uses:
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

                self.conversation_log.append({
                    "turn": turn,
                    "role": "tool_call",
                    "tool_name": tool_use.name,
                    "tool_input": tool_use.input,
                    "tool_result": result,
                })

                if verbose:
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

                if tool_use.name == "submit_final_code":
                    final_result = json.loads(result)
                    break

            if final_result:
                break

            messages.append({"role": "user", "content": tool_results})

            if self.iteration >= self.max_iterations:
                if verbose:
                    print(f"\nMax iterations ({self.max_iterations}) reached.")
                break

        # Build final result
        cost = self.get_cost_estimate()

        return {
            "success": self.best_accuracy >= self.target_accuracy,
            "final_code": self.best_code,
            "final_accuracy": self.best_accuracy,
            "iterations": self.iteration,
            "submitted": final_result is not None,
            "cost": cost,
            "trajectory": self.trajectory,
            "conversation": self.conversation_log,
        }
