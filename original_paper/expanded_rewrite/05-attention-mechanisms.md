# 05: Attention Mechanisms (Short)

Short overview of spatial attention and decay mechanisms used in GraphMERT. For detailed formulas and derivations, see `05-attention-mechanisms-FULL.md`.

---

## 1. Spatial decay mask

- Based on **shortest-path distances** in the leafy chain graph.
- Encodes the intuition that:
  - Close nodes (in graph distance) attend strongly.
  - Distant nodes are down-weighted or masked.

Key config: `SpatialAttentionConfig` in `src/types.jl`.

---

## 2. Graph attention mask

- Combines:
  - Base attention mask (padding / sequence length).
  - Distance-based decay.
  - Optional learned thresholds.
- Used by H-GAT to restrict which pairs of nodes can exchange information.

Implementation anchors:

- `GraphMERT/src/architectures/attention.jl`
- `GraphMERT/src/graphs/leafy_chain.jl` (for paths/distances)

---

## 3. Practical guidance

- When modifying attention behavior:
  - Keep decay simple and monotonic unless the spec explicitly requires more complex shapes.
  - Update or add small tests that verify mask shapes and basic monotonicity.
- Put any new math-heavy sections into `05-attention-mechanisms-FULL.md` rather than here.

