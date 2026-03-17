# Stream C Refinement Plan

## Context
Streams A, B, D, E, F, and G are largely complete. Stream C (External Integrations) has functional stubs but lacks persistence and robustness for real-world usage.

## Goal
Harden the external integration layer, specifically adding persistence to the UMLS client and fleshing out the SapBERT linker structure.

## Tasks

### C2.4 — Persistent UMLS Cache
- [ ] Add `SQLite` dependency to `Project.toml`.
- [ ] Define `SQLiteUMLSCache` struct implementing the cache interface.
- [ ] Create tables: `concepts` (cui, data, timestamp), `relations` (cui, data, timestamp).
- [ ] Implement `get` and `set` methods using SQLite.
- [ ] Update `create_umls_client` to use `SQLiteUMLSCache` by default if a path is provided.
- [ ] Add unit tests for cache persistence (write, close, open, read).

### C3.2 — SapBERT Linker Refinement
- [ ] Update `SapBERTLinker` to support loading a pre-computed embedding matrix from a file (e.g., `.npy` or `.jld2`) if ONNX is too heavy for now.
- [ ] Implement `cosine_similarity` search over this matrix.
- [ ] Add `save_index` and `load_index` methods to persist the ANN index (or simple matrix).
- [ ] Document the expected model format.

## Next Steps
1. Execute C2.4 (SQLite Cache).
2. Execute C3.2 (SapBERT Index Persistence).
