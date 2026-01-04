"""Run a policy reform analysis.

Usage:
    python examples/run_reform.py examples/uk_tax_benefit.rac examples/reform.rac

Compares baseline vs reform on synthetic microdata and shows distributional impact.
"""

import sys
import time
from datetime import date

import numpy as np

from rac import compile, compile_to_binary, parse


def load_microdata(n: int = 100_000) -> dict[str, np.ndarray]:
    """Generate synthetic UK population microdata."""
    np.random.seed(42)

    # Income distribution (log-normal, roughly UK-like)
    gross_income = np.random.lognormal(mean=10.3, sigma=0.8, size=n)
    gross_income = np.clip(gross_income, 0, 500_000)

    return {
        "person": np.column_stack([
            gross_income,
            np.full(n, 40.0),  # hours_worked
            np.ones(n),        # is_adult
            np.zeros(n),       # is_child
        ])
    }


def run_analysis(baseline_path: str, reform_path: str):
    """Run baseline vs reform comparison."""
    as_of = date(2024, 6, 1)

    # Parse
    print(f"Parsing {baseline_path}...")
    baseline_module = parse(open(baseline_path).read())

    print(f"Parsing {reform_path}...")
    reform_module = parse(open(reform_path).read())

    # Compile baseline
    print("Compiling baseline...")
    baseline_ir = compile([baseline_module], as_of=as_of)
    baseline_binary = compile_to_binary(baseline_ir)

    # Compile reform (amendments applied on top of baseline)
    print("Compiling reform...")
    reform_ir = compile([baseline_module, reform_module], as_of=as_of)
    reform_binary = compile_to_binary(reform_ir)

    # Load microdata
    print("Loading microdata...")
    data = load_microdata(n=1_000_000)
    n = len(data["person"])
    income = data["person"][:, 0]

    # Run baseline
    print("Running baseline...")
    start = time.perf_counter()
    baseline_result = baseline_binary.run(data)
    baseline_time = time.perf_counter() - start

    # Run reform
    print("Running reform...")
    start = time.perf_counter()
    reform_result = reform_binary.run(data)
    reform_time = time.perf_counter() - start

    # Extract net income (last column)
    baseline_net = baseline_result["person"][:, -1]
    reform_net = reform_result["person"][:, -1]
    gain = reform_net - baseline_net

    # Analysis
    print(f"\n{'='*60}")
    print("REFORM IMPACT ANALYSIS")
    print(f"{'='*60}")
    print(f"Population: {n:,}")
    print(f"Baseline run: {baseline_time:.3f}s ({n/baseline_time/1e6:.1f}M/sec)")
    print(f"Reform run:   {reform_time:.3f}s ({n/reform_time/1e6:.1f}M/sec)")

    # Aggregate stats
    total_cost = gain.sum() * 12  # Annual
    avg_gain = gain.mean()
    winners = (gain > 1).sum()
    losers = (gain < -1).sum()

    print(f"\nAggregate impact:")
    print(f"  Total cost: £{total_cost/1e9:.2f}bn/year")
    print(f"  Avg gain:   £{avg_gain:.0f}/month")
    print(f"  Winners:    {winners:,} ({100*winners/n:.1f}%)")
    print(f"  Losers:     {losers:,} ({100*losers/n:.1f}%)")

    # Distributional analysis by income decile
    print(f"\nBy income decile:")
    print(f"{'Decile':>8} {'Avg Income':>12} {'Avg Gain':>10} {'% Winners':>10}")
    print("-" * 44)

    deciles = np.percentile(income, np.arange(10, 101, 10))
    decile_idx = np.digitize(income, deciles)

    for d in range(10):
        mask = decile_idx == d
        if mask.sum() == 0:
            continue
        avg_inc = income[mask].mean()
        avg_g = gain[mask].mean()
        pct_win = 100 * (gain[mask] > 1).sum() / mask.sum()
        print(f"{d+1:>8} £{avg_inc:>10,.0f} £{avg_g:>9,.0f} {pct_win:>9.0f}%")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_reform.py <baseline.rac> <reform.rac>")
        sys.exit(1)

    run_analysis(sys.argv[1], sys.argv[2])
