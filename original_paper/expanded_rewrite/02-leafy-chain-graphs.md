# 02: Leafy Chain Graphs (Short)

This is the **short spec** for the leafy chain graph. For full mathematical detail, proofs, and extended examples, see:

- `02-leafy-chain-graphs-FULL.md`
- `11-data-structures-FULL.md` (type definitions)

---

## 1. Purpose

Leafy chain graphs provide a fixed-size representation that unifies:

- **Syntactic space**: token sequence from text.
- **Semantic space**: injected KG triples (heads, relations, tails).

They are the backbone for:

- Joint MLM (on roots) + MNM (on leaves).
- Graph-based attention and spatial bias.

---

## 2. Shape and invariants

- Fixed maximum size:
  - `num_roots = 128`
  - `num_leaves_per_root = 7`
  - `max_sequence_length = 1024`
- Every node is either:
  - A **root** token (text) or
  - A **leaf** token (semantic triple-related).
- Padding is explicit; all sequences are exactly 1024 positions.

Implementation reference:

- Type `LeafyChainGraph` and helpers in `GraphMERT/src/graphs/leafy_chain.jl`.

---

## 3. Key operations (high level)

1. **Construction from text**
   - Tokenize text.
   - Create 128 root tokens (truncation/padding as needed).
   - Initialize leaves as padding.

2. **Triple injection**
   - Map semantic triples to leaf positions under appropriate roots.
   - Track how many leaves are actually injected.

3. **Graph→sequence encoding**
   - Flatten nodes into a 1D sequence of length 1024.
   - Produce adjacency / shortest-path matrices for spatial attention.

---

## 4. Implementation notes

- The file `GraphMERT/src/graphs/leafy_chain.jl` is the authority for:
  - Concrete field names and types.
  - Adjacency, shortest paths, and attention masks.
  - Helper functions like `create_leafy_chain_from_text`.
- Use tests in `GraphMERT/test/unit/test_extraction.jl` and any dedicated leafy-chain tests as the executable spec.

When extending the design, keep this file short and push derivations, detailed layouts, and long examples into `02-leafy-chain-graphs-FULL.md`.
