# 01: Architecture Overview (Short)

High-level summary of the GraphMERT architecture. For detailed diagrams, equations, and historical notes, see `01-architecture-overview-FULL.md`.

---

## 1. Main components

- **RoBERTa encoder**: contextual token representations.
- **Leafy Chain Graph**: fixed-size graph representation of text + injected triples.
- **H-GAT**: graph attention over the leafy chain.
- **Heads**:
  - Entity classifier.
  - Relation classifier.
  - LM head for MLM/MNM.

---

## 2. Data flow (inference)

1. Input text → tokenize → leafy chain graph.
2. Graph → sequence + masks → RoBERTa.
3. RoBERTa outputs + graph structure → H-GAT.
4. H-GAT outputs → entity & relation classifiers, LM head.
5. Outputs → entities, relations, and candidate tails.

Implementation anchors:

- `GraphMERT/src/GraphMERT.jl`
- `GraphMERT/src/architectures/roberta.jl`
- `GraphMERT/src/architectures/hgat.jl`
- `GraphMERT/src/graphs/leafy_chain.jl`

---

## 3. Training objectives

- **MLM** on roots (text tokens).
- **MNM** on leaves (semantic tokens).
- Optional auxiliary losses (e.g. boundary, relation-specific).

See `06-training-mlm-FULL.md` and `07-training-mnm-FULL.md` for full details.

---

## 4. Practical usage

- For conceptual understanding, this file + the short training and triple-extraction specs should be enough.
- For implementation details or research work, jump to the `*-FULL.md` specs and the implementation mapping in `12-implementation-mapping-FULL.md`.

