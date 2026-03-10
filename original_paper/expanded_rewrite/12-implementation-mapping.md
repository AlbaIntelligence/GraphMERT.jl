# 12: Implementation Mapping (Short)

Short guide mapping major spec sections to code locations. For exhaustive tables and commentary, see `12-implementation-mapping-FULL.md`.

---

## 1. Core model

- **RoBERTa encoder**
  - Spec: `03-roberta-encoder-FULL.md`
  - Code: `src/architectures/roberta.jl`

- **H-GAT**
  - Spec: `04-hgat-component-FULL.md`
  - Code: `src/architectures/hgat.jl`

---

## 2. Graph and attention

- **Leafy chain graph**
  - Spec: `02-leafy-chain-graphs-FULL.md`
  - Code: `src/graphs/leafy_chain.jl`

- **Attention mechanisms**
  - Spec: `05-attention-mechanisms-FULL.md`
  - Code: `src/architectures/attention.jl`

---

## 3. Training

- **MLM**
  - Spec: `06-training-mlm-FULL.md`
  - Code: `src/training/mlm.jl`

- **MNM**
  - Spec: `07-training-mnm-FULL.md`
  - Code: `src/training/mnm.jl`

- **Seed injection**
  - Spec: `08-seed-kg-injection-FULL.md`
  - Code: `src/training/seed_injection.jl`, `src/seed_injection.jl`

---

## 4. Extraction and domains

- **Triple extraction**
  - Spec: `09-triple-extraction-FULL.md`
  - Code: `src/api/extraction.jl`, `test/unit/test_extraction.jl`

- **Domain abstraction**
  - Spec: domain-related sections in `GENERALIZATION_PLAN` and domain docs in `reports/`.
  - Code: `src/domains/interface.jl`, `src/domains/registry.jl`, domain modules.

---

## 5. Evaluation and visualization

- **Metrics**
  - Spec: `10-evaluation-metrics-FULL.md`
  - Code: `src/evaluation/*.jl`, `src/types.jl` for score types.

- **Visualization**
  - Spec: `reports/GRAPH_VISUALIZATION_RESEARCH.md`
  - Code: `src/visualization/*.jl`, visualization examples.

Use this file when you need a **quick mapping**; jump to the full implementation mapping for detailed status and line-level notes.

