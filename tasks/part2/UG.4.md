# Task UG.4 — Whole-Project Report (Part I + Part II) and Traceability Table

## Objective
Author the comprehensive combined Part I + Part II thesis report (`docs/whole_project_report.md`) following the 12-section outline in `docs/part2/07_checklists.md` §7, including the revised Part I arithmetic share derivation ($\approx 78.9\%$ of linear MACs / $74.1\%$ of total decode with the LM head, $>90\%$ only for prefill), labeled estimates, and an exhaustive numerical traceability table (`docs/report_traceability.md`) where every number traces to a `results/` artifact, producing command, and git commit.

## Implementation
- **`docs/whole_project_report.md`**: Complete 12-section + 4-appendix combined thesis report covering Part I spatial linear compilation (`E1`–`E5`, Claim `C1`) and Part II full 30-layer SmolLM2-135M ROM-backed decoder hardware compilation (`numerics/`, `ADR-2-1`, `ADR-2-2`, `ir/graph_v2.py`, `rtl/lib/`, verification methodology, `UE.1`–`UE.7` Claims `C1'` & `C2'`, and `EA1`–`EA5` synthesis & 28nm ASIC projections).
- **`docs/report_traceability.md`**: Complete Part I (`M-01`..`E5-02`) and Part II (`P2-M01`..`UG3-01`) traceability table with zero unsourced numbers and explicit `[estimate]` labels on all extrapolations/projections.

## Evidence
- Verified that every number in `docs/whole_project_report.md` matches `docs/report_traceability.md` and the underlying JSON/CSV artifacts in `results/` and `results/part2/`.

## Status
Status: complete
