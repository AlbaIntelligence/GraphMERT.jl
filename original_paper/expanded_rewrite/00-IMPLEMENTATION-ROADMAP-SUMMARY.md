# 00-IMPLEMENTATION-ROADMAP (Summary)

Short, implementation-focused roadmap. For full detail, see `00-IMPLEMENTATION-ROADMAP.md`.

---

## 1. Current foundation

- RoBERTa encoder (`architectures/roberta.jl`) and H-GAT (`architectures/hgat.jl`) are in good shape.
- MLM training (`training/mlm.jl`) and core types (`src/types.jl`) are implemented and tested.
- Leafy chain graphs, MNM, seed injection, and triple extraction are partially implemented but structurally present.

---

## 2. Critical missing pieces (functional parity)

Focus on these before polishing:

1. **Leafy Chain Graph**
   - Ensure `LeafyChainGraph` + helpers in `graphs/leafy_chain.jl` fully match the spec (fixed size, padding, graph→sequence).
   - Add/keep unit tests validating construction, encoding, and injection behavior.

2. **MNM forward pass**
   - Replace the MNM forward stub in `training/mnm.jl` with a real path:
     - `LeafyChainGraph` → `graph_to_sequence` → model embeddings → H-GAT → LM head.
   - Wire MNM loss into the joint MLM+MNM training step.

3. **Seed KG injection**
   - Implement the 4-stage injection pipeline in `training/seed_injection.jl` and `seed_injection.jl`:
     - Entity linking, triple selection, bucket/sampling, injection into graphs.

4. **Triple extraction pipeline**
   - Make the 5-stage extraction API (`api/extraction.jl`) use domains by default and only fall back to heuristics when needed.
   - Ensure `extract_knowledge_graph` (API + tests) runs end-to-end with at least one working domain.

---

## 3. Short phase plan

### Phase A – Unblock end-to-end extraction

- Implement robust `discover_head_entities` / `match_relations_for_entities` using `DomainProvider`.
- Fix `KnowledgeGraph` construction from `Entity` / `Relation` so tests and examples agree on one representation.
- Make sure `test_extraction.jl` and `test_api.jl` both pass or encode the intended spec.

### Phase B – Training pipeline

- Implement MNM forward and joint MLM+MNM training as described in `07-training-mnm.md`.
- Hook up seed injection so training can consume injected triples.
- Add small, fast tests that exercise a minimal training step.

### Phase C – Evaluation and visualization

- Stabilize FActScore / Validity / GraphRAG APIs.
- Use the new visualization module (`src/visualization/*.jl`) and examples as the canonical plotting path.

---

## 4. How to use this summary

- Start here to understand **what to build next**.
- Jump to the full `00-IMPLEMENTATION-ROADMAP.md` only when you need detailed rationale, time estimates, or historical notes.
- When editing or adding roadmap content, keep this summary to **1–2 screens** and push long explanations into the full document.
