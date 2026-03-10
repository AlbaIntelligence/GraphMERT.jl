# 07: MNM Training (Short)

Short specification of the **Masked Node Modeling (MNM)** objective. For full math, examples, and derivations, see `07-training-mnm-FULL.md`.

---

## 1. Goal

MNM trains the model to reconstruct **semantic leaf tokens** (from injected triples) given:

- The surrounding **leafy chain graph** (roots + leaves).
- The model’s hidden representations after H-GAT and transformer layers.

It complements MLM (on roots) and encourages the model to internalize KG structure.

---

## 2. Masking strategy (conceptual)

- Choose a subset of leaf spans to mask:
  - Probability ≈ 0.15 (configurable).
  - Optionally mask entire leaf spans instead of single tokens.
- Apply standard 80/10/10 masking:
  - 80% replaced by `[MASK]` token ID.
  - 10% replaced by random leaf token ID.
  - 10% left unchanged (for stabilization).

Implementation entry points:

- Config: `MNMConfig` in `src/types.jl`.
- Helpers: `training/mnm.jl` (mask selection, batch construction).

---

## 3. Forward pass (high level)

For each batch:

1. Build masked leafy chain graphs (roots + leaves).
2. Convert graphs to sequences (`graph_to_sequence`, attention masks).
3. Run through `GraphMERTModel` to obtain logits over the vocabulary.
4. Compute cross-entropy loss on masked leaf positions only.

The current code still uses a **stubbed MNM forward**; see `training/mnm.jl` and `reports/CODE_REVIEW.md` for TODOs.

---

## 4. Loss and training

- MNM loss is combined with MLM loss using a configurable weight.
- Joint objective encourages:
  - Language understanding (MLM).
  - Graph/semantic understanding (MNM).

Training scaffolding lives in:

- `training/mnm.jl`
- `training/mlm.jl`
- `training/pipeline.jl`

Keep this document short; add new detailed sections to `07-training-mnm-FULL.md` as needed.
