# 09: Triple Extraction Pipeline (Short)

Short specification of the **5-stage triple extraction pipeline**. For full detail, see `09-triple-extraction-FULL.md`.

---

## 1. Goal

Given input text and a `GraphMERTModel`, extract a **knowledge graph** with:

- Entities (heads and tails).
- Relations.
- Provenance and basic metadata.

The high-level stages:

1. Head discovery
2. Relation matching
3. Tail prediction (model)
4. Tail formation (text)
5. Filtering and deduplication

---

## 2. Stage summaries

### 2.1 Head discovery

- Use a `DomainProvider` to extract entities from text:
  - `extract_entities(domain, text, options) -> Vector{Entity}`
- Fallback: heuristic entity recognition if the domain fails or is missing.

### 2.2 Relation matching

- Use `extract_relations(domain, entities, text, options)` to propose relations between heads.
- Fallback: simple co-occurrence-based relations and heuristics over the raw text.

### 2.3 Tail prediction

- Use the model to predict likely tail tokens given (head, relation, context).
- Implementation: `predict_tail_tokens` in `api/extraction.jl`.

### 2.4 Tail formation

- Turn token-level predictions into textual tail candidates.
- Implementation: `form_tail_from_tokens`.

### 2.5 Filtering and deduplication

- Use similarity and key-based deduplication to prune redundant/low-quality triples.
- Implementation: `filter_and_deduplicate_triples`.

---

## 3. Implementation notes

- The main API lives in:
  - `GraphMERT/src/api/extraction.jl`
  - `GraphMERT/src/GraphMERT.jl` (public wrappers).
- Use `GraphMERT/test/unit/test_extraction.jl` and `GraphMERT/test/unit/test_api.jl` as the **executable spec** for expected behavior and edge cases.

Keep this document short and defer extended examples, prompts, and error-handling details to `09-triple-extraction-FULL.md`.
