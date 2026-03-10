# 04: H-GAT Component (Short)

Short description of the **Hierarchical Graph Attention (H-GAT)** module. For full equations, ablations, and proofs, see `04-hgat-component-FULL.md`.

---

## 1. Purpose

H-GAT operates on the leafy chain graph to:

- Inject **spatial and structural bias** into token representations.
- Aggregate information along root–leaf–leaf paths.
- Provide graph-aware embeddings for entity/relation prediction and MNM.

---

## 2. Inputs and outputs

- Inputs:
  - Node representations from RoBERTa.
  - Graph structure (adjacency, shortest paths, attention decay masks).
- Outputs:
  - Updated node embeddings reflecting graph connectivity and relation hints.

Implementation reference: `GraphMERT/src/architectures/hgat.jl`.

---

## 3. Attention structure

- Uses **multi-head attention** constrained by:
  - Graph connectivity (which nodes can attend).
  - Distance-based decay (from `SpatialAttentionConfig`).
- Can be configured to emphasize:
  - Local neighborhoods.
  - Relation-specific paths.

---

## 4. Practical notes

- H-GAT is called from `GraphMERTModel` after RoBERTa.
- Exact formulas and update rules are in the full spec and should be consulted when changing implementation details.
