# Reference Sources and Encoder Alignment

This document summarizes the contextual information and reference implementations added to the repo, and clarifies how the **encoder** (RoBERTa vs alternatives) is used across specs, reference code, and this Julia implementation.

---

## 1. Contextual information (`Contextual_information.md`)

**Location:** repo root `Contextual_information.md`.

**Contents (condensed):**

- **Related work:** GraphMERT (arXiv:2510.09580) is cited as an ~80M-parameter graphical encoder-only transformer; follow-on work (LinguGKD, DRAG, low-resource KGC, RSC/CEUR KG-from-LLM papers) is summarized. No explicit “GraphMERT v2” extensions; GraphMERT is one exemplar in neurosymbolic LLM–graph distillation.
- **How GraphMERT extracts KGs with a small model:** Domain narrowing + ontology grounding, graphical encoder-only architecture, leaf-structured text–graph representation, distillation from curated text + seed KG (not only from one LLM), symbolic constraints and post-processing, provenance (sentence-level extraction). Pipeline: corpus + seed KG → leaf-structured representation → 80M encoder → triple prediction → symbolic validation → iterative improvement.
- **Reliability and provenance:** FActScore (factuality), ValidityScore (ontology consistency), sentence-level provenance, KG cleaning, human-in-the-loop and neurosymbolic stack.

**Relevance for GraphMERT.jl:** The doc describes “compact transformer encoder” and “80M encoder”; it does not name RoBERTa explicitly. Our **expanded spec** (`original_paper/expanded_rewrite/01-architecture-overview.md`) and **PROJECT_STATUS** do name **RoBERTa** as the encoder. So the high-level narrative in `Contextual_information.md` is aligned; the concrete choice of RoBERTa comes from the paper/spec and our architecture.

---

## 2. Notebook (`original_paper/graphMERT.ipynb`)

**Location:** `original_paper/graphMERT.ipynb`.

**Setup:**

- **Text encoding:** **SentenceTransformer `all-MiniLM-L6-v2`** (SBERT), then **PCA** to a fixed dimension (e.g. 10).
- **Graph construction:** Sentences → SBERT embeddings → PCA → chunked into fixed-size sequences (e.g. `seq_len=5`) → each chunk is one “chain graph” with `features` (reduced embeddings) and `labels`.
- **Model:** **GraphMERTEncoder** = `Linear(input_dim → embed_dim*num_heads)` + **H-GAT layers** + **`nn.TransformerEncoder`** (standard PyTorch). Training is a classification-style loss over node labels.

**Encoder in the notebook:** There is **no RoBERTa (and no BERT)**. The text encoder is **MiniLM (SBERT)**; the “GraphMERT” part is a linear projection + H-GAT + standard transformer encoder on top of precomputed embeddings. So the notebook is a **simplified, encoder-light** variant: external SBERT does the language understanding; the rest is graph + transformer on fixed features.

**Takeaway:** Useful for understanding H-GAT + chain-graph data shape and training loop, but **not** an implementation of the paper’s “RoBERTa + H-GAT” stack.

---

## 3. Cloned Python repo (`original_paper/GraphMert/`)

**Location:** `original_paper/GraphMert/` (Python, PyTorch, HuggingFace `transformers`).

**Architecture (from `src/models/graphmert.py` and `src/config.py`):**

- **Embeddings:** Custom **GraphMERTEmbeddings** (BERT-style: word + position + token type; optional injection of H-GAT relation embeddings at `[REL]` positions). **No** use of pretrained `BertModel` or `BertEncoder` in the forward path; only `BertModel, BertConfig` (and related classes) are imported but the model uses custom layers.
- **Transformer:** Custom **GraphMERTLayer** / **GraphMERTAttention** with **graph-aware attention decay** (distance matrix). So: BERT-like embedding layout, but **custom** encoder layers with decay mask, not vanilla BERT/RoBERTa.
- **Heads:** MLM head, MNM head; H-GAT encoder for relation embeddings.
- **Config:** 12 layers, 8 heads, 512 hidden, vocab 28895 (BioMedBERT), ~80M parameters.

**Encoder in the clone:** **BERT-style** (word+position+token_type) + **custom graph-aware transformer**. No RoBERTa; no loading of pretrained BERT/RoBERTa weights in the provided code. So the reference implementation is “BERT-like” in shape only; the actual encoder is custom and graph-conditioned.

---

## 4. This Julia project: RoBERTa and “not using RoBERTa”

**What the Julia codebase has:**

- **RoBERTa:** Full encoder in `GraphMERT/src/architectures/roberta.jl` (RoBERTaConfig, RoBERTaModel, embeddings, layers, etc.). Used in **training** (e.g. `src/training/mnm.jl` calls `model.roberta(...)`; `src/training/pipeline.jl` builds a config with `RoBERTaConfig`).
- **Spec:** `original_paper/expanded_rewrite/01-architecture-overview.md` and `reports/PROJECT_STATUS.md` state that the encoder is **RoBERTa**.
- **Persistence / extraction:** `load_model(model_path)` delegates to the persistence layer; `reports/PROJECT_STATUS.md` marks **model persistence as a stub**. So the **extraction path** may not currently load or run the Julia RoBERTa at inference time (e.g. if the “model” is an ONNX stub or missing).

**Interpretation of “our project is not using RoBERTa (and should maybe)”:**

1. **At inference/extraction:** If `load_model` does not return a model that includes the RoBERTa encoder (e.g. persistence not wired), then the **extraction pipeline is not using RoBERTa** in practice.
2. **Pretrained weights:** Even if the RoBERTa **architecture** is used in training, we may not be loading **pretrained** RoBERTa weights (only random init or placeholder), so “using RoBERTa” in the paper’s sense (compact pretrained encoder) might not be true yet.
3. **“Should maybe”:** To align with the paper and spec, the project **should**:
   - Use the **RoBERTa encoder** in the **extraction path** (wire persistence so that `load_model` provides a model that runs RoBERTa + H-GAT), and/or
   - Support loading **pretrained RoBERTa** (or compatible) weights into the Julia RoBERTa module so that the 80M encoder is not randomly initialized.

---

## 5. Summary table

| Source                    | Text / token encoder              | Graph / structure        | Uses RoBERTa? |
|---------------------------|-----------------------------------|---------------------------|---------------|
| Paper / Contextual info   | “Compact transformer encoder”     | Leaf-structured, H-GAT    | Implied (80M) |
| Expanded spec (this repo) | RoBERTa                           | Leafy chain, H-GAT       | Yes (spec)    |
| Notebook                  | MiniLM (SBERT) + PCA              | H-GAT + TransformerEnc   | No            |
| Python clone              | Custom BERT-style + graph layers  | H-GAT, attention decay   | No            |
| GraphMERT.jl              | RoBERTa (implemented)             | H-GAT, leafy chain       | In training; extraction/persistence TBD |

---

## 6. Reliability narrative and mapping (Contextual_information.md → capabilities)

**Contextual_information.md** emphasizes: provenance (sentence-level), FActScore (factuality), ValidityScore (ontology consistency), KG cleaning, human-in-the-loop, neurosymbolic stack. Mapping to **GraphMERT.jl**:

| Contextual_information.md | GraphMERT.jl capability | Status |
|---------------------------|--------------------------|--------|
| Sentence-level provenance | `enable_provenance_tracking`, `get_provenance(kg, relation)` | ✅ Implemented |
| ValidityScore (ontology) | `validate_kg(kg, domain)` → ValidityReport; graceful degradation when ontology missing | ✅ Implemented |
| FActScore (factuality) | `evaluate_factscore(kg, reference)` → FactualityScore | ✅ Implemented |
| KG cleaning | `clean_kg(kg; policy)` → cleaned KnowledgeGraph | ✅ Implemented |
| Encoder in path | `load_model(path)` returns full model; extraction uses encoder when model is GraphMERTModel | ⚠️ Partial (config load; weight load TBD) |
| Iterative seed (cleaned KG as seed) | Documented path: clean → export → seed config; seed_injection accepts triples | ⚠️ Documented; full orchestration partial |
| Human-in-the-loop | Provenance and validity reports support audit and correction workflows | ✅ Supported by API |

**Model storage:** The RoBERTa/GraphMERT model is stored on disk at a **user-provided path** (no fixed repo location). Use `load_model(path)` and `save_model(model, path)`; the checkpoint file is JSON (config and metadata). Loading pretrained weights from the checkpoint file is post-MVP (see `specs/003-align-contextual-description/spec.md` Clarifications).

**Gaps:** Full checkpoint weight loading (pretrained RoBERTa) is not yet wired; iterative seed at scale is partial. See `specs/003-align-contextual-description/` and `reports/PROJECT_STATUS.md` for roadmap.

---

## 7. Recommended next steps

- **Document:** Keep `Contextual_information.md` at repo root as high-level context; use this file (`reports/REFERENCE_SOURCES_AND_ENCODER.md`) as the single place that ties together the three sources and the encoder story.
- **Encoder alignment:** Treat the **expanded spec** and **paper** as requiring a RoBERTa (or RoBERTa-like) encoder. The notebook and Python clone are **reference implementations** that use different encoders (SBERT+PCA, or custom BERT-style); we should not change our spec to match them; instead we should **wire and use** the Julia RoBERTa for extraction and, where applicable, pretrained weights.
- **Persistence and extraction:** As per `reports/PROJECT_STATUS.md`, wire full weight loading in `load_model` so that the extraction pipeline can run pretrained RoBERTa + H-GAT when a full checkpoint is provided.
