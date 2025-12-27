# Preregistration: AI-Driven Rules-as-Code Encoding

**Study Title:** Reinforcement Learning from Implementation Feedback for Automated Statutory Encoding

**Authors:** Cosilico AI

**Date:** December 2024

**Registration:** This document serves as the preregistration for our empirical study on AI-driven rules-as-code encoding. The hypotheses, methods, and analysis plans below were specified before data collection.

---

## 1. Research Questions

### Primary Question
Can large language models (LLMs) automatically encode statutory provisions into executable code by using existing implementations as oracles for reinforcement learning?

### Secondary Questions
1. How does accuracy vary across provision complexity (simple formulas vs. multi-branch logic)?
2. Does transfer learning occur—do later provisions encode faster than earlier ones?
3. What is the cost-effectiveness compared to manual encoding?
4. Where do LLMs systematically fail, and what human oversight is required?

---

## 2. Hypotheses

### H1: Convergence Hypothesis
**Statement:** For well-defined statutory provisions with dense oracle coverage, the agentic loop will achieve ≥95% accuracy within 10 iterations.

**Operationalization:**
- "Well-defined": Provisions with explicit formulas or clear conditional logic (e.g., EITC phase-in, standard deduction)
- "Dense oracle coverage": ≥100 test cases from PolicyEngine or TAXSIM
- "Accuracy": Proportion of test cases where generated code output matches oracle output within $1

**Expected effect:** 80% of well-defined provisions will converge within 10 iterations.

### H2: Complexity Scaling Hypothesis
**Statement:** The number of iterations required scales with provision complexity, measured by:
- Number of conditional branches
- Number of parameters referenced
- Dependency depth (how many other provisions it references)

**Operationalization:**
- Simple (complexity score 1-3): ≤3 iterations to 95% accuracy
- Medium (complexity score 4-6): 4-7 iterations
- Complex (complexity score 7+): 8+ iterations or failure

**Complexity scoring:**
- +1 per conditional branch
- +1 per parameter
- +1 per external reference
- +2 per nested conditional

### H3: Transfer Learning Hypothesis
**Statement:** Encoding performance improves over the course of the study—provisions encoded later require fewer iterations than earlier provisions of similar complexity.

**Operationalization:**
- Compare mean iterations for first 5 provisions vs. last 5 provisions (matched by complexity)
- Expect ≥20% reduction in iterations

**Mechanism:** Few-shot examples accumulate; model learns statutory patterns.

### H4: Cost Efficiency Hypothesis
**Statement:** The cost per provision (API tokens × rate) decreases over time and is competitive with estimated manual encoding costs.

**Operationalization:**
- Track total tokens (input + output) per provision
- Estimate manual cost at $100-500 per provision (based on engineer hours)
- Expect AI cost <$10 per simple provision, <$50 per complex provision

### H5: Failure Mode Hypothesis
**Statement:** Systematic failures will cluster around specific categories:
1. Ambiguous statutory language
2. Missing parameters in oracle
3. Oracle disagreements (PE vs. TAXSIM)
4. Temporal logic (effective dates, phase-ins)

**Operationalization:**
- Categorize all failures that don't reach 90% accuracy
- Expect ≥50% of failures to fall into the above categories

---

## 3. Methods

### 3.1 Provisions to Encode

We will encode the following provisions from the Internal Revenue Code (IRC):

**Phase 1: Simple Provisions (complexity 1-3)**
1. EITC phase-in credit (26 USC § 32(a)(1))
2. Standard deduction (26 USC § 63(c))
3. Child tax credit base amount (26 USC § 24(a))
4. Personal exemption (26 USC § 151) [historical]
5. SALT deduction cap (26 USC § 164(b)(6))

**Phase 2: Medium Provisions (complexity 4-6)**
6. Full EITC with phase-out (26 USC § 32)
7. Child and dependent care credit (26 USC § 21)
8. Premium tax credit (26 USC § 36B)
9. Saver's credit (26 USC § 25B)
10. Adoption credit (26 USC § 23)

**Phase 3: Complex Provisions (complexity 7+)**
11. Alternative Minimum Tax (26 USC § 55-59)
12. Net Investment Income Tax (26 USC § 1411)
13. Social Security benefit taxation (26 USC § 86)
14. Qualified business income deduction (26 USC § 199A)
15. Foreign tax credit (26 USC § 27)

### 3.2 Oracle Setup

**Primary Oracle:** PolicyEngine-US (v1.x)
- Provides comprehensive federal tax calculations
- API access for automated testing

**Secondary Oracle:** TAXSIM (NBER)
- Independent validation
- Used to flag disagreements

**Test Case Generation:**
- 100-500 test cases per provision
- Stratified sampling across income levels, filing statuses, family sizes
- Boundary cases at phase-in/phase-out thresholds
- Adversarial cases designed to expose edge cases

### 3.3 Agentic Loop Configuration

**Model:** Claude Sonnet 4 (claude-sonnet-4-20250514)
**Max iterations:** 10 per provision
**Target accuracy:** 95%
**Tools available:**
- `execute_dsl`: Run code against test cases
- `submit_final_code`: Complete encoding

**Prompt structure:**
1. System prompt with DSL specification
2. Statutory text (verbatim from USC)
3. Sample test cases (5 examples)
4. Parameter values (from IRS publications)

### 3.4 Metrics

**Primary Metrics:**
- Accuracy: % test cases passing (within $1 tolerance)
- Iterations: Number of generate-test cycles
- Tokens: Total input + output tokens
- Cost: Tokens × rate (Sonnet: $3/M in, $15/M out)

**Secondary Metrics:**
- Time to convergence (wall clock)
- Error categories (syntax, runtime, value mismatch)
- Human interventions required

### 3.5 Analysis Plan

**H1 Analysis:**
- Calculate convergence rate (% provisions reaching 95% in ≤10 iterations)
- 95% CI via bootstrap (1000 resamples)
- Success threshold: lower bound of CI > 60%

**H2 Analysis:**
- Fit regression: iterations ~ complexity_score
- Expect positive coefficient with p < 0.05
- R² threshold: > 0.3 indicates meaningful relationship

**H3 Analysis:**
- Paired comparison: first 5 vs. last 5 provisions (matched complexity)
- Wilcoxon signed-rank test
- Effect size: Cohen's d

**H4 Analysis:**
- Plot cost per provision over sequence
- Fit exponential decay: cost_i = a * exp(-b*i) + c
- Compare to manual cost estimate ($100-500)

**H5 Analysis:**
- Categorize failures manually
- Chi-square test for non-uniform distribution across categories
- Report category proportions with 95% CI

---

## 4. Data Collection Protocol

### 4.1 Procedure
1. For each provision in order (Phase 1 → Phase 2 → Phase 3):
   a. Generate test cases from oracle
   b. Run agentic loop with logging
   c. Record all metrics
   d. If failure (no convergence), categorize failure mode
   e. Optional: manual fix and re-run with fix as context

### 4.2 Logging
All runs will log:
- Full conversation history (prompts and responses)
- Token counts per turn
- Test results per iteration
- Final code
- Timestamps

Logs stored in: `paper/data/runs/`

### 4.3 Stopping Rules
- Stop iteration if accuracy ≥95% (success)
- Stop iteration if accuracy unchanged for 3 consecutive iterations (plateau)
- Stop iteration at 10 iterations (max)
- Stop study early if 5 consecutive complex provisions fail (systemic issue)

---

## 5. Deviations and Amendments

Any deviations from this preregistration will be documented in:
`paper/amendments.md`

Types of acceptable deviations:
- Adding provisions (extending scope)
- Adjusting complexity scores based on initial results
- Bug fixes in tooling (not affecting results)

Types requiring justification:
- Changing target accuracy threshold
- Changing max iterations
- Excluding provisions post-hoc

---

## 6. Timeline

- **Week 1:** Phase 1 provisions (simple)
- **Week 2:** Phase 2 provisions (medium)
- **Week 3:** Phase 3 provisions (complex)
- **Week 4:** Analysis and writeup

---

## 7. Code and Data Availability

All code, data, and analysis will be available at:
- Repository: https://github.com/CosilicoAI/cosilico-engine
- Paper: https://docs.rac.ai/paper/

---

## Appendix A: Complexity Scoring Examples

### EITC Phase-In (Complexity: 2)
- 0 conditionals (single formula)
- 2 parameters (phase_in_rate, earned_income_amount)
- 0 external references
- Score: 2

### Full EITC (Complexity: 6)
- 3 conditionals (phase-in, plateau, phase-out regions)
- 5 parameters (rate, amount, max, po_start, po_rate)
- 1 external reference (earned_income)
- Score: 6

### AMT (Complexity: 12)
- 5+ conditionals (exemption phase-out, rate brackets, preferences)
- 8+ parameters
- 5+ external references (regular tax, various deductions)
- 2 nested conditionals
- Score: 12+

---

## Appendix B: Test Case Stratification

For each provision, test cases will be stratified across:

**Income levels:** $0, $10K, $25K, $50K, $75K, $100K, $150K, $200K, $500K, $1M
**Filing status:** Single, MFJ, MFS, HoH, QW
**Dependents:** 0, 1, 2, 3, 4+
**Age:** <65, 65+
**State:** CA (for state tax provisions)

Plus boundary cases at:
- Exactly at threshold values
- $1 below/above thresholds
- Zero income edge cases
- Maximum value edge cases
