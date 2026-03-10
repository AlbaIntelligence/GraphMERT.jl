# GraphMERT.jl — Full Code Review

This document is a code review of the GraphMERT.jl replication project against the original paper: _GraphMERT: Efficient and Scalable Distillation of Reliable Knowledge Graphs from Unstructured Data_ (arXiv 2510.09580). The implementation mapping in `original_paper/expanded_rewrite/12-implementation-mapping.md` is used as a reference; this review updates and extends that analysis.

---

## 1. Executive Summary

| Aspect              | Assessment                                                                                                                                                        |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Paper alignment** | Architecture (RoBERTa, H-GAT, leafy chain, MLM/MNM) is well reflected in code structure and specs.                                                                |
| **Completeness**    | Core building blocks (RoBERTa, H-GAT, MLM, leafy chain, MNM helpers) are largely in place; extraction and training now run end-to-end, with remaining work in data prep, advanced injection, and evaluation wiring. |
| **Code quality**    | Strong in architectures and types; API surface, tests, and documentation have been aligned with the current domain system and public types but still need ongoing consolidation.                                 |
| **Risks**           | Historical API mismatches (e.g. `discover_head_entities`), Entity vs KnowledgeEntity confusion, and the MNM forward stub have been addressed; residual risk is in less-tested training/evaluation paths.       |

**Recommendation:** Focus ongoing work on (1) tightening training data preparation and seed injection for larger corpora, (2) extending test coverage for joint MLM+MNM and persistence, and (3) adding small, stable replication checks (e.g. FActScore on a fixed snippet) to guard regressions.

---

## 2. Paper vs Implementation

### 2.1 What the paper describes

- **Goal:** Reliable, domain-specific KGs from text: factual (provenance), valid (ontology-consistent), automatable, scalable, domain-agnostic, globally integrated.
- **Model:** Encoder-only transformer (RoBERTa-style) + H-GAT; input is a _leafy chain graph_ (roots = text tokens, leaves = semantic triple tokens).
- **Training:** Joint MLM (syntactic, on roots) + MNM (semantic, on leaves); seed KG injection into training data.
- **Extraction:** Head discovery → relation matching → tail prediction (model) → tail formation → filtering/deduplication.
- **Evaluation:** FActScore\*, ValidityScore, GraphRAG.

### 2.2 What the codebase provides

- **RoBERTa + H-GAT:** Implemented in `architectures/roberta.jl` and `architectures/hgat.jl`; structure and forward passes align with the paper.
- **Leafy chain:** `graphs/leafy_chain.jl` implements chain+star structure, adjacency, Floyd–Warshall, `inject_triple!`, `graph_to_sequence`, attention mask (beyond the “7% complete” claim in the implementation mapping).
- **MLM:** Implemented in `training/mlm.jl` (span masking, boundary loss, metrics).
- **MNM:** Helpers in `training/mnm.jl` (mask selection, mask application, loss, joint step); `forward_pass_mnm` is now implemented using RoBERTa + LM head over the leafy-chain sequence and returns real logits of the correct shape.
- **Seed injection:** `training/seed_injection.jl` has domain-agnostic structure (e.g. `link_entity_sapbert` delegating to `link_entity(domain, ...)`) and a working selection algorithm; integration into large-scale training remains a future optimisation.
- **Extraction API:** `api/extraction.jl` implements the full 5-stage pipeline with `discover_head_entities(text, domain_provider, options)` and a convenience `discover_head_entities(text)` overload; when a `GraphMERTModel` is provided, stages 3–5 (tail prediction, formation, filtering) are enabled, otherwise the pipeline runs in domain-only mode.
- **Domains:** Pluggable `DomainProvider` in `domains/interface.jl`; Wikipedia and biomedical domains exist with entity/relation types and optional KB linking.
- **Evaluation:** FActScore, validity, GraphRAG modules exist; FActScore references `HelperLLMClient` (defined in `llm/helper.jl`), but some evaluation code paths may not be wired for all domains.


### 2.3 KnowledgeGraphRecipe vs core visualization

`KnowledgeGraphRecipe` is a small, separate package under `KnowledgeGraphRecipe/` that predates the richer `GraphMERT.Visualization` module. It provides:
- Thin wrappers around GraphRecipes/Plots (`plot_knowledge_graph`, `plot_knowledge_graph_filtered`, `export_graph`).
- Very simple filters (`filter_by_confidence`, `filter_by_entity_type`) that operate directly on `KnowledgeGraph`.
- A basic `kg_to_graph` helper that builds an adjacency matrix from entity *texts* rather than IDs and discards most metadata.
- Styling helpers keyed off `entity.entity_type` and relation strings.

By contrast, the main visualization stack in `GraphMERT/src/visualization/`:
- Converts `KnowledgeGraph` into a `MetaGraph` (`kg_to_graphs_format`) and preserves full entity/relation metadata (IDs, attributes, confidence, timestamps).
- Implements static/interactive plots that work over `Graphs` + `MetaGraphs` and reuse more of the graph ecosystem.
- Provides higher‑level helpers for filtering, simplification, clustering, and summary statistics.

**Implications:**
- There is duplicated functionality: both trees implement `filter_by_confidence`, `filter_by_entity_type`, `export_graph`, and their own `kg_to_graph*` helpers, but only the `GraphMERT.Visualization` path preserves IDs and attributes and is used consistently by the library.
- `KnowledgeGraphRecipe`’s layout story is minimal: `custom_layout` is effectively a stub that just returns `:spring`, whereas the main visualization module already has hooks for more sophisticated layouts (and can be extended with NetworkLayout.jl when available).

**Recommendation:**
- Treat `KnowledgeGraphRecipe` as an experimental/legacy visualization wrapper and avoid using it in new code. Prefer the `GraphMERT.Visualization` entry points (`kg_to_graphs_format`, `visualize_graph`, `plot_knowledge_graph`, filters in `visualization/utils.jl`).
- If we want a public, package‑separate plotting story, consider *wrapping* the `GraphMERT.Visualization` API from `KnowledgeGraphRecipe` instead of re‑implementing helpers there, or retire the separate package once downstream users are migrated.


---

## 3. Critical Issues

### 3.1 `discover_head_entities` and extraction API (resolved)

- **Location:** `api/extraction.jl` calls `discover_head_entities(text, domain_provider, options)` and exposes a convenience `discover_head_entities(text)` overload.
- **Status:** Implemented and wired to `extract_entities(domain, text, options)` with robust fallbacks when domains are missing; tests use the same public signatures.

### 3.2 Entity/Relation vs KnowledgeEntity/KnowledgeRelation (resolved)

- **Location:** `api/extraction.jl` builds `GraphMERT.KnowledgeGraph(entities, relations, metadata)` where `entities` and `relations` come from domain extraction.
- **Status:** A convenience `KnowledgeGraph(entities::Vector{Entity}, relations::Vector{Relation}, ...)` constructor in `types.jl` converts generic entities/relations into `KnowledgeEntity`/`KnowledgeRelation` and stores relation indices (`head_entity_id`/`tail_entity_id`) in attributes; tests and visualization use this unified representation.

### 3.3 MNM forward pass

- **Location:** `training/mnm.jl`, `forward_pass_mnm`.
- **Status:** The forward path now runs `graph_to_sequence(masked_graph)` → RoBERTa encoder → LM head to produce real logits with shape `[batch, seq_len, vocab_size]`. This matches the paper’s requirement for a semantic prediction head over the leafy-chain sequence; H-GAT remains available for classification-style heads.

### 3.4 GraphMERTModel call signature and sequence length

- **Location:** `models/graphmert.jl`, `api/extraction.jl` (e.g. `model(reshape(input_ids, 1, :), reshape(attention_mask, 1, :))`).
- **Problem:** The main `GraphMERTModel` combines RoBERTa, H-GAT, and classifiers; it is unclear whether `(input_ids, attention_mask)` is the intended external interface and whether the model’s internal RoBERTa/H-GAT and leafy chain length (e.g. 1024) are aligned. `create_graphmert_model` uses `GraphMERTConfig` from `models/graphmert.jl` with `max_sequence_length::Int = 512`, while the leafy chain can be 128 roots + 896 leaves = 1024.
- **Impact:** Risk of dimension mismatches or silent truncation when moving from graph to model.
- **Recommendation:** Document the expected input: batch of token IDs and mask, and max length. Unify `max_sequence_length` with the leafy chain size (or explicitly define how the graph is truncated/padded). Add a small test that runs one forward pass from `LeafyChainGraph` → `graph_to_sequence` → model and checks output shape.

---

## 4. High-Priority Issues

### 4.1 Test suite inconsistencies

- **Problem:** Many integration tests call `discover_head_entities(text)` and expect `BiomedicalEntity` or similar. Unit tests mix `BiomedicalEntity`/`BiomedicalRelation` with `KnowledgeEntity`/`KnowledgeRelation` and different `KnowledgeGraph` constructor forms (e.g. 3 args vs 4 args with metadata).
- **Recommendation:** Standardise on one “public” entity/relation type for the extraction API (prefer `Entity`/`Relation` with conversion to `KnowledgeEntity`/`KnowledgeRelation` for storage, or the opposite). Update tests to use the same signatures as the main API (e.g. domain + options where applicable) and to construct `KnowledgeGraph` via the same constructor.

### 4.2 Evaluation modules and HelperLLMClient

- **Location:** `evaluation/factscore.jl`, `evaluation/validity.jl`, `evaluation/graphrag.jl` use `Union{HelperLLMClient,Nothing}`.
- **Observation:** `HelperLLMClient` is defined in `llm/helper.jl`; the evaluation code is in the same module, so the type is available. Ensure all evaluation entry points accept `nothing` and skip LLM-dependent steps when no client is passed, and that docstrings/export list match.

### 4.3 Deprecated types still in use

- **Location:** `types.jl` marks `BiomedicalEntity` and `BiomedicalRelation` as deprecated in favor of `Entity`/`Relation` with `domain="biomedical"`.
- **Problem:** Tests and examples still use `BiomedicalEntity`/`BiomedicalRelation` and `KnowledgeGraph(..., BiomedicalRelation[], ...)`.
- **Recommendation:** Add a small migration path in tests (e.g. a test helper that builds `Entity`/`Relation` from legacy types) and gradually replace usages; then remove deprecation warnings once call sites are updated.

### 4.4 Leafy chain: two `create_empty_chain_graph` variants

- **Location:** `graphs/leafy_chain.jl` has (1) `create_empty_chain_graph(tokens, token_texts, config)` (with roots from tokens) and (2) `create_empty_chain_graph(config)` (fully empty).
- **Observation:** Both are valid; the second is documented as the “empty” constructor. `create_leafy_chain_from_text` (used in extraction) should be verified to use the intended variant and tokenization so that sequence length and padding match the model.

---

## 5. Positive Findings

- **RoBERTa and H-GAT:** Clear structure, documented, and aligned with the paper; good base for extension.
- **MLM:** Span masking and boundary loss are implemented and testable.
- **Leafy chain:** Structure, adjacency, Floyd–Warshall, injection, `graph_to_sequence`, and attention mask are implemented and validated; the implementation mapping understated this.
- **Domain abstraction:** `DomainProvider` and Wikipedia/biomedical domains give a clean way to add domains and to separate entity/relation logic from the core pipeline.
- **Seed injection design:** Domain-agnostic delegation to `link_entity(domain, ...)` and caching are in place; remaining work is mainly triple selection and injection orchestration.
- **MNM helpers:** Mask selection (roots vs leaves), 80/10/10 masking, and joint MLM+MNM step structure are in place; only the forward pass is missing.
- **Documentation:** `DOMAIN_DEVELOPER_GUIDE.md`, implementation mapping, and expanded paper rewrite under `original_paper/expanded_rewrite/` are useful for onboarding and alignment.

---

## 6. Recommendations Summary

| Priority | Action                                                                                                                                                                 |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P0       | [32mDONE[0m Implement `discover_head_entities(text, domain_provider, options)` and optional `discover_head_entities(text)`; ensure extraction path runs end-to-end.                |
| P0       | [32mDONE[0m Resolve Entity/Relation vs KnowledgeEntity/KnowledgeRelation: add conversion or an outer `KnowledgeGraph` constructor and use it consistently in extraction and tests. |
| P0       | [32mDONE[0m Replace `forward_pass_mnm` stub with real model forward (graph → sequence → embeddings → transformer → LM head).                                               |
| P1       | [32mDONE[0m Unify `GraphMERTModel` interface and max sequence length with leafy chain (1024 vs 512); add a single forward-pass test from graph to logits.                          |
| P1       | Align tests with the chosen API (domain + options, single KnowledgeGraph constructor) and reduce use of deprecated Biomedical\* types.                                 |
| P2       | Seed injection and training pipeline are implemented for small-scale runs; future work is to harden them for large corpora and multiple domains.                      |
| P2       | Re-run full test suite and fix any failures; add one small replication check (e.g. FActScore on a fixed snippet) to guard regressions.                                 |

---

## 7. Library usage and consolidation

### 7.1 Graph and visualization stack

- **Current usage**
  - Core visualization (`GraphMERT/src/visualization/`) uses **Graphs.jl** + **MetaGraphs.jl** as the canonical representation and **GraphRecipes.jl** + **Plots.jl** (when available) to render static/interactive plots.
  - `kg_to_graphs_format` converts `KnowledgeGraph` into a `MetaGraph` and preserves full metadata; helpers in `visualization/utils.jl` work on this representation for filtering, simplification, clustering, and statistics.
  - `KnowledgeGraphRecipe` uses **Graphs.jl** directly and its own `kg_to_graph` helper that builds an adjacency matrix from entity texts, dropping IDs and most attributes.

- **Observations**
  - The MetaGraph-based path is strictly more expressive (it preserves attributes and supports richer analysis) and is already the one integrated into the main API.
  - There is duplicated functionality (filters, `export_graph`, simple adjacency construction) between `KnowledgeGraphRecipe` and `GraphMERT.Visualization`.

- **Recommendations**
  - Standardise on **Graphs.jl + MetaGraphs.jl** as the internal graph representation and treat `kg_to_graphs_format` as the single entry point for visualization/analysis.
  - Keep **GraphRecipes.jl + Plots.jl** usage confined to the `Visualization` module; if `KnowledgeGraphRecipe` is kept around, have it *wrap* `GraphMERT.Visualization` rather than re‑implement helpers.

### 7.2 ML / NLP and API libraries

- **Current usage**
  - **Flux.jl** underpins all model and training code (RoBERTa, H-GAT, MLM/MNM, training loops).
  - **HTTP.jl** + **JSON/JSON3.jl** are used for API surfaces: configuration loading/serialization, LLM helper requests, and UMLS/Wikidata-style integrations.
  - Tokenisation and text preprocessing are implemented directly in `GraphMERT/src/text/tokenizer.jl` and domain modules, rather than via a generic Transformers.jl/TextAnalysis.jl stack.

- **Observations**
  - For the scope of this replication, Flux covers the paper’s needs; there is no obvious benefit to swapping the NN engine.
  - HTTP/JSON usage is already factored into helpers (`api/helpers.jl`, `llm/helper.jl`), keeping the rest of the codebase insulated from protocol details.
  - There are opportunities to lean more on existing text/transformer tooling in the future (e.g. tokenizers, schedulers), but doing so is not required to match the paper.

- **Recommendations**
  - Continue to use **Flux.jl** as the primary NN/optimizer stack for GraphMERT; avoid ad‑hoc training loops that bypass its ecosystem.
  - Keep **HTTP/JSON** usages confined to helper modules and LLM/UMLS/Wikidata clients, so the core pipeline remains framework‑agnostic.
  - When extending tokenization or adding new models, prefer reusing existing components from **Transformers.jl** or similar libraries where it does not conflict with the paper’s architecture or training objectives.

---

## 8. References

- Paper: `original_paper/2510.09580 - GraphMERT - Efficient and Scalable Distillation of Reliable Knowledge Graphs from Unstructured Data/paper.tex`
- Implementation mapping: `original_paper/expanded_rewrite/12-implementation-mapping.md`
- Domain guide: `DOMAIN_DEVELOPER_GUIDE.md`
