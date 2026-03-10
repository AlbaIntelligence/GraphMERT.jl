# 13: Gaps Analysis (Short)

Short list of key implementation gaps versus the full GraphMERT spec. For the exhaustive, historical analysis, see `13-gaps-analysis-FULL.md` and `reports/CODE_REVIEW.md`.

---

## 1. Model and training

- MNM forward pass still stubbed; needs real integration with `GraphMERTModel`.
- Seed injection pipeline incomplete (selection and integration with training).
- Training pipeline orchestration (`training/pipeline.jl`) only partially wired.

---

## 2. Extraction and domains

- Domain-level `extract_entities` / `extract_relations` for biomedical/Wikipedia not fully implemented.
- Extraction API often relies on heuristic fallbacks instead of domain logic.
- Some API tests (`test_api.jl`) expect legacy fields (`head_entity_id`, etc.) that are not part of the current type design.

---

## 3. Evaluation and visualization

- Metrics are present but not consistently wired into CI or example workflows.
- Visualization stack is implemented but still new and may lack coverage in tests and docs.

---

## 4. Documentation debt

- Full specs in `expanded_rewrite/*-FULL.md` are intentionally long; summaries plus `reports/CODE_REVIEW.md` should be preferred by agents.
- Some tests and docs still reference older concepts (e.g. biomedical-only APIs) and need migration to the domain system.

When adding new gaps or resolving old ones, update:

- This short file (high-level bullets only).
- `13-gaps-analysis-FULL.md` (detailed reasoning).
- `reports/CODE_REVIEW.md` (prioritized, actionable tasks).

