# 06: MLM Training (Short)

Short description of **Masked Language Modeling (MLM)** in GraphMERT. For full details, see `06-training-mlm-FULL.md`.

---

## 1. Objective

Standard MLM over **root tokens** (text) in the leafy chain graph:

- Mask a subset of root positions.
- Predict the original tokens given context.

This provides the usual language-modeling signal for the encoder.

---

## 2. Masking behavior

- Probability ≈ 0.15 of masking each eligible token.
- 80/10/10 rule:
  - 80% → `[MASK]`
  - 10% → random token
  - 10% → unchanged

Implementation reference: `GraphMERT/src/training/mlm.jl`.

---

## 3. Loss and usage

- Cross-entropy over masked positions only.
- Combined with MNM loss during joint training.
- Training helpers (batch creation, masking, boundary loss, total loss) are implemented in `training/mlm.jl`.

---

## 4. Notes

- This doc intentionally omits full pseudocode; see the full spec and tests for that.
- When changing MLM behavior, keep this file short and update `06-training-mlm-FULL.md` with detailed rationale.
