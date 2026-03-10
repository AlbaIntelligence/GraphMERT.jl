# 10: Evaluation Metrics (Short)

Short overview of the evaluation metrics used for GraphMERT. For rigorous definitions and derivations, see `10-evaluation-metrics-FULL.md`.

---

## 1. FActScore\*

- Measures **factual accuracy** of extracted knowledge graphs.
- Compares predicted triples against a reference set with:
  - Precision, recall, F1.
  - Coverage over ground-truth facts.

Implementation anchors:

- `GraphMERT/src/evaluation/factscore.jl`
- Type: `FActScore` in `src/types.jl`.

---

## 2. ValidityScore

- Measures **logical/ontological validity** of triples:
  - Type correctness.
  - Constraint satisfaction.
- Often domain-specific (e.g. UMLS constraints).

Type: `ValidityScore` in `src/types.jl` and helpers in evaluation modules.

---

## 3. GraphRAG

- High-level score capturing:
  - Retrieval quality.
  - Generation quality (when used with an LLM).
  - Overall end-to-end performance.

Type: `GraphRAG` in `src/types.jl`, with implementations in `evaluation/graphrag.jl`.

---

## 4. Usage

- Evaluation entry points are exported from `GraphMERT` (e.g. `evaluate_factscore`, `evaluate_graphrag`).
- When adding or changing metrics, keep this document brief and move detailed math into `10-evaluation-metrics-FULL.md`.
