# 03: RoBERTa Encoder (Short)

Short description of the RoBERTa-based encoder used in GraphMERT. For full architecture details and hyperparameters, see `03-roberta-encoder-FULL.md`.

---

## 1. Role in GraphMERT

- Provides **token-level contextual embeddings** for the input sequence derived from the leafy chain graph.
- Acts as the **base transformer** on top of which H-GAT operates.

---

## 2. Inputs and outputs

- Inputs:
  - `input_ids::Matrix{Int}` (batch × seq_len).
  - `attention_mask::Matrix{Float32}`.
  - Optional position and token-type IDs.
- Outputs:
  - Sequence embeddings (per token).
  - Pooled embedding (for global tasks, if needed).

Implementation reference: `GraphMERT/src/architectures/roberta.jl`.

---

## 3. Configuration

- Based on a standard base-size RoBERTa:
  - Hidden size, number of layers/heads, vocab size, etc.
- Exact defaults and tunables are defined in the `RoBERTaConfig` type and documented in the full spec.

---

## 4. Usage notes

- In GraphMERT, RoBERTa is **not used in isolation**; it is always part of `GraphMERTModel` together with H-GAT and classification heads.
- See `models/graphmert.jl` and the architecture overview for how it composes with the rest of the system.
