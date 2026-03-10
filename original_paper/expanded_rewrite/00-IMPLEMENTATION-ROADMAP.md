# 00-IMPLEMENTATION-ROADMAP (Short)

This is the **short, token-efficient** roadmap. For the full, historical and highly detailed version, see:

- `00-IMPLEMENTATION-ROADMAP-FULL.md`
- `STATUS-SUMMARY.md` and `STATUS-FULL.md`
- `reports/CODE_REVIEW.md` (current prioritized actions)

---

## 1. Foundation (already implemented or mostly done)

- **RoBERTa encoder** (`architectures/roberta.jl`)
- **H-GAT component** (`architectures/hgat.jl`)
- **MLM training** (`training/mlm.jl`)
- **Core types** (`src/types.jl`)

Use the full roadmap only if you need legacy context, effort estimates, or historical notes. For current work, treat these as _done_ unless CODE_REVIEW says otherwise.

---

## 2. Critical path (P0)

These are required to get a robust, paper-aligned end-to-end system:

1. **Leafy chain graphs**
   - Ensure `LeafyChainGraph` and helpers in `src/graphs/leafy_chain.jl` match the spec.
   - Validate fixed-size layout, padding, graph→sequence, and injection points with tests.

2. **MNM forward pass**
   - Replace the MNM stub in `training/mnm.jl` with a real model path using `GraphMERTModel`.
   - Wire MNM loss into joint MLM+MNM training.

3. **Seed KG injection**
   - Implement and test the injection pipeline in `training/seed_injection.jl` and `src/seed_injection.jl`.

4. **Triple extraction**
   - Make the 5-stage pipeline in `api/extraction.jl` use domains by default and fall back to heuristics only when domains are missing/broken.

---

## 3. Near-term phases (A/B/C)

### Phase A – Unblock extraction and domains

- Align `extract_knowledge_graph` and tests (`test_api.jl`, `test_extraction.jl`) on one canonical flow.
- Implement minimal, correct `DomainProvider` methods for biomedical and Wikipedia.

### Phase B – Training

- Implement MNM forward + joint training.
- Connect seed injection to the training pipeline.

### Phase C – Evaluation and visualization

- Stabilize evaluation APIs and visualization examples.
- Ensure one clear, documented path for metrics and plots.

---

## 4. How to extend this document

- Keep this file under **~400 lines** with high-level tasks and links.
- Put detailed plans, tables, and history into `00-IMPLEMENTATION-ROADMAP-FULL.md` and update `reports/CODE_REVIEW.md` for day-to-day tracking.
