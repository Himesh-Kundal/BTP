"""Unit test for Mutation Suite v2 (Task UC.13).

Enforces 100% kill rate across synthetic RTL fault models M10 through M30:
- Every mutant must be detected and killed by the standard test suite.
- Kill matrix is verified and saved to `results/mutation_matrix_v2.json`.
"""

from __future__ import annotations

from pathlib import Path

from verification.mutate2 import MUTANTS_V2, run_mutation_campaign

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_mutation_suite_v2_100_percent_kill(tmp_path: Path):
    """Verify that every single mutant M10-M30 is caught and killed by tests."""
    out_matrix = tmp_path / "mutation_matrix_v2.json"
    campaign = run_mutation_campaign(MUTANTS_V2, output_path=out_matrix)

    assert campaign["total_mutants"] == 21, f"Expected 21 mutants, got {campaign['total_mutants']}"
    assert campaign["survived_mutants"] == 0, (
        f"Mutants survived! {campaign['survived_mutants']} survived out of {campaign['total_mutants']}: "
        f"{[r['id'] for r in campaign['results'] if r['status'] == 'SURVIVED']}"
    )
    assert campaign["kill_rate_pct"] == 100.0, f"Expected 100.0% kill rate, got {campaign['kill_rate_pct']}%"

    # Also persist to repo results/ directory as required by contract
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "mutation_matrix_v2.json", "w", encoding="utf-8") as f:
        import json
        json.dump(campaign, f, indent=2)
