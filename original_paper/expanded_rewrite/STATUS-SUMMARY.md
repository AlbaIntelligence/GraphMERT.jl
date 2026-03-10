# STATUS (Summary)

Short status view for agents and implementers. For full details, see `STATUS.md` and `PROGRESS.md`.

---

## 1. Documentation status

- All 15 spec documents are **written and stable**:
  - Master: `00-INDEX.md`, `00-IMPLEMENTATION-ROADMAP.md`.
  - Critical path: leafy chain, MNM, seed injection, triple extraction.
  - High priority: architecture, attention, data structures, gaps analysis.
  - Supporting: evaluation metrics, implementation mapping, progress/status.
- Specs are intentionally verbose; use this summary + the `*-SUMMARY.md` files when possible to save tokens.

---

## 2. Implementation status (high level)

- **Done / solid**
  - RoBERTa encoder and H-GAT modules.
  - MLM training.
  - Core types (`types.jl`) and most architecture plumbing.
  - Leafy chain graphs: structure, helpers, and graph→sequence mapping implemented and exercised in tests.
  - Triple extraction: 5-stage pipeline (`discover_head_entities`, relation matching, optional tail prediction/formation/filtering) runs end-to-end with domains and fallbacks.
  - MNM helpers and joint loss: masking, loss pieces, and `forward_pass_mnm` (graph→sequence→RoBERTa→LM head) are implemented.

- **Partially done**
  - Seed injection: data structures, selection, and injection API are implemented and used in the training pipeline for small-scale runs; large-corpus tuning remains future work.
  - Domain-level `extract_entities` / `extract_relations` for biomedical/Wikipedia: basic heuristics exist; richer patterns/LLM support are still in progress.
  - Model persistence and reload: JSON-based checkpoints exist and are exercised by API tests; weight-level save/load is still a placeholder.

- **Needs work**
  - Evaluation metrics and visualization examples wired into CI in a stable way.

---

## 3. Practical guidance

- Use `reports/CODE_REVIEW.md` for the **current prioritized task list** (P0/P1/P2).
- Use `AGENTS.md` for:
  - Onboarding order (what to read first).
  - Token-efficiency rules when editing specs and code.
  - A concise snapshot of current implementation gaps.
- Use the full spec docs only when you need:
  - Exact algorithmic details.
  - Formal definitions, equations, or complete examples.

Keep this file short. When adding new status information, prefer updating `reports/CODE_REVIEW.md` and only reflect **the top-level outcome** here.

