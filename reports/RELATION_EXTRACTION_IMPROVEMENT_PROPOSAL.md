# Proposal: Improve Relation Extraction

**Context**: On the 3 Wikipedia fixture articles, extraction currently yields **2 relations total** (both on the first article); the other two articles get **0 relations**. This document proposes concrete changes to improve relation extraction.

---

## Implementation status (done)

All four proposed improvements have been implemented:

- **§2.1** **Generalized Wikipedia pattern-based extraction**: `domains/wikipedia/relations.jl` now uses entity-agnostic regex patterns (SPOUSE_OF, PARENT_OF, CHILD_OF, SUCCESSOR_OF, PREDECESSOR_OF, BORN_IN, DIED_IN, REIGNED_*, MEMBER_OF_DYNASTY, RULED) and resolves captures against the extracted entity list. Deduplication by (head, relation_type, tail).
- **§2.4** **Stronger fallback**: `api/extraction.jl` — when domain is missing or relation extraction fails, `_fallback_cooccurrence_relations` runs sentence-level co-occurrence (entity pairs only within the same sentence), deduplicates by (head, relation_type, tail), and uses `String(sentence)` for provenance. `domain === nothing` skips domain call and uses fallback directly.
- **§2.2** **LLM wired into relation matching**: `match_relations_for_entities` now accepts and passes `llm_client` to `extract_relations(..., llm_client=...)`. Wikipedia `extract_relations` accepts `; llm_client=nothing` and, when `llm_client !== nothing` and `config.use_local`, calls `create_prompt(domain, :relation_matching, context)`, runs `LocalLLM.generate`, and parses "Entity1 | Relation | Entity2" lines into relations. Interface and biomedical `extract_relations` accept optional `llm_client` for compatibility. `LocalLLM.generate(client, prompt)` added and exported.
- **§2.3** **Optional Wikidata enrichment**: When `domain.wikidata_client !== nothing`, `extract_wikipedia_relations` calls `_enrich_relations_from_wikidata`: for each entity with `attributes["wikidata_qid"]`, fetches relations via `get_wikidata_relations`, maps properties to relation types, resolves tail to an entity in the list (or via `get_wikidata_label`), and appends relations with confidence 0.85. Include order in `domain.jl` set to wikidata before relations.

---

## 1. Current state (pre-implementation)

### 1.1 Wikipedia domain (`domains/wikipedia/relations.jl`)

- **Hardcoded regexes only**: A small set of literal patterns for specific entity pairs, e.g.:
  - `(Louis XIV)\s+married\s+(Maria Theresa of Spain)`
  - `(Marie Antoinette)\s+married\s+(Louis XVI)`
  - `(Louis XIV)\s+.*(?:father|parent)\s+of\s+(Louis XV)`
  - etc.
- Relations are found only when the exact phrase appears. Slight wording changes (“son of”, “was married to”, “succeeded”) or different entities (Henry IV, Louis XIII) often match nothing.
- **No LLM**: The Wikipedia `create_relation_matching_prompt` exists in `prompts.jl` but is **never used** in the extraction pipeline.

### 1.2 Extraction pipeline (`api/extraction.jl`)

- `match_relations_for_entities` calls `domain.extract_relations(domain, entities, text, options)` only.
- **No `llm_client` is passed** to relation extraction, so domains cannot use the helper LLM for relation matching even if they support it.

### 1.3 Interface default (`domains/interface.jl`)

- Default `extract_relations` uses sentence splitting and regexes with `\w+` (single-word captures), so multi-word names (e.g. “Louis XIV”, “Maria Theresa of Spain”) are not matched. It also only considers the first two entities per sentence.

### 1.4 Spec (`09-triple-extraction-FULL.md`)

- Stage 2 is specified as **“Relation Matching (Helper LLM)”** with an LLM that, given entity and context, returns applicable relations. The current code path does not implement this for Wikipedia.

---

## 2. Proposed improvements (in order)

### 2.1 Generalize Wikipedia pattern-based extraction (quick win)

**Goal**: Replace the few literal regexes with **entity-agnostic**, **relation-type-first** patterns so many more (head, relation, tail) triples are found from text.

**Changes**:

- In `domains/wikipedia/relations.jl`, define a **list of (regex, relation_type)** where the regex uses **named or positional capture groups** that can be matched against the **actual entity list** (not literal “Louis XIV”):
  - Examples:
    - `(X)\s+married\s+(Y)` → SPOUSE_OF (then check X, Y in entity list)
    - `(X)\s+(?:was\s+)?(?:the\s+)?(?:father|mother|son|daughter|parent)\s+of\s+(Y)` → PARENT_OF
    - `(X)\s+(?:reigned|ruled|succeeded)\s+(?:from|after|following)\s+(Y)` or “X succeeded Y” → SUCCESSOR_OF / REIGNED_AFTER
    - `(X)\s+(?:was\s+)?born\s+in\s+(Y)`, `(X)\s+(?:was\s+)?died\s+in\s+(Y)` → BORN_IN, DIED_IN
    - `(X)\s+reigned\s+from\s+(\d+)` → reigned_from (with date as tail)
- For each sentence (or paragraph), find all entities that appear in it; for each pattern, run the regex and resolve captures to entity IDs using the existing `entity_lookup` / `find_entity` (with multi-word names). Add a `Relation` only when both head and tail resolve to entities in the list.
- Align relation type names with `reference_facts.jl` (e.g. spouse, parent_of, predecessor, successor, reigned_from, born_in, died_in) so tests and evaluation can compare.

**Outcome**: More relations per article without new dependencies; works offline.

---

### 2.2 Wire the helper LLM into relation matching (spec-aligned)

**Goal**: Use the existing **Wikipedia relation-matching prompt** so that when a local LLM is configured, relation extraction can use it and return “Entity1 | Relation | Entity2” style triples.

**Changes**:

- **API**: Extend the relation-extraction call so the domain can optionally receive an LLM client:
  - Option A: Add an overload or optional kwarg, e.g. `extract_relations(domain, entities, text, options; llm_client=nothing)`. In `match_relations_for_entities`, pass through `llm_client` from the extraction pipeline (which already has it).
  - Option B: Add a separate step in the pipeline: “if options.use_local && llm_client !== nothing, call domain.create_prompt(domain, :relation_matching, context) and domain method to parse LLM output into Relation[]; merge with pattern-based relations.”
- **Wikipedia**:
  - In `extract_wikipedia_relations` (or a new helper), when `llm_client` is provided and `options.use_local` is true:
    - Build context `Dict("entities" => [e.text for e in entities], "text" => text)`.
    - Call `create_prompt(domain, :relation_matching, context)` to get the prompt.
    - Call the LLM (reuse the same mechanism used for entity discovery in `entities.jl`), then parse the response: split lines, look for “Entity1 | Relation | Entity2”, resolve Entity1/Entity2 to entities via `find_entity`, and append `Relation` objects with a suitable confidence (e.g. 0.75).
  - Deduplicate with pattern-based relations (e.g. by (head_id, relation_type, tail_id)).
- **Interface**: In `domains/interface.jl`, extend the default `extract_relations` signature to accept an optional `llm_client` (default `nothing`) so other domains can adopt the same pattern later.

**Outcome**: Relation extraction matches the spec (Helper LLM for relation matching) and improves recall and relation type diversity when a local LLM is available.

---

### 2.3 Optional Wikidata enrichment

**Goal**: For entities that have a Wikidata QID, optionally **merge in relations from Wikidata** (e.g. from `get_wikidata_relations`) so the KG includes structured relations even when the article text does not state them explicitly.

**Changes**:

- In the Wikipedia domain, after pattern-based (and optionally LLM) relation extraction:
  - For each entity that has a QID (e.g. from `link_entity` or from entity metadata), call `get_wikidata_relations(qid, domain.wikidata_client)` if `wikidata_client !== nothing`.
  - Map property IDs to relation types via `map_wikidata_property_to_relation_type`; convert to `Relation` with head/tail resolved to entity IDs where the tail entity is in the extracted entity list (or add a “tail” entity for the value if desired).
  - Merge into the relation list with a source tag (e.g. metadata `"source" => "wikidata"`) and optional confidence discount (e.g. 0.85) so evaluation can distinguish text-based vs Wikidata relations.
- Make this **optional** (e.g. `options.wikidata_enrich_relations = true` or a domain config flag) and **guarded** (no network in tests unless explicitly enabled).

**Outcome**: Richer graphs for well-linked entities; evaluation can still focus on text-based extraction if needed.

---

### 2.4 Stronger fallback in `match_relations_for_entities`

**Goal**: When domain relation extraction fails or returns empty, the fallback should still produce **co-occurrence-based relations** that are **entity-aware** (use actual entity IDs and optional relation hints from text).

**Changes**:

- In `api/extraction.jl`, in the fallback branch:
  - Split text into sentences; for each sentence, collect entities that **occur in** that sentence (substring or normalized match).
  - For each pair (head, tail) of entities in the same sentence, add a relation with type “ASSOCIATED_WITH” (or keep simple keyword heuristics like “treat” → TREATS, “cause” → CAUSES for biomedical fallback).
  - Build `Relation` objects with the same structure as today (head.id, tail.id, relation_type, confidence, …) and **deduplicate** by (head, tail) or (head, relation_type, tail).
- This avoids the current fallback that creates a relation for **every** pair of entities in the whole text (O(n²)) and often overwhelms the graph; sentence-scoped co-occurrence is more meaningful.

**Outcome**: Fewer empty relation sets when the domain fails; better behavior for unknown or future domains.

---

## 3. Suggested order of work

| Priority | Item                               | Effort  | Impact |
|----------|------------------------------------|---------|--------|
| 1        | §2.1 Generalize Wikipedia patterns | Medium  | High   |
| 2        | §2.4 Stronger fallback             | Small   | Medium |
| 3        | §2.2 Wire LLM into relation match  | Medium  | High   |
| 4        | §2.3 Wikidata enrichment           | Medium  | Medium (optional) |

Implementing **§2.1** and **§2.4** first gives immediate gains and no new dependencies; **§2.2** aligns with the spec and enables LLM-based relation extraction; **§2.3** is optional and useful for evaluation/coverage.

---

## 4. Success criteria

- **Fixtures**: On the 3 Wikipedia test articles, total relations ≥ 10 (with pattern-based + optional LLM), and relation types include at least SPOUSE_OF, PARENT_OF, BORN_IN/DIED_IN or REIGNED/SUCCESSOR where the text supports them.
- **Reference facts**: A subset of `reference_facts.jl` triples that appear in the fixture text should be recoverable as (head, relation_type, tail) in the extracted KG (allowing for naming normalization).
- **No regression**: Existing unit tests in `test_extraction.jl` and Wikipedia domain tests still pass; extraction pipeline still runs with `model = nothing` (discovery-only) without errors.

---

## 5. Files to touch

- **Pattern-based + LLM (Wikipedia)**: `GraphMERT/src/domains/wikipedia/relations.jl`, optionally `domain.jl` (to pass llm_client if signature changes).
- **API / pipeline**: `GraphMERT/src/api/extraction.jl` (`match_relations_for_entities`, and where it is called to pass `llm_client`).
- **Interface**: `GraphMERT/src/domains/interface.jl` (`extract_relations` signature and fallback behavior if desired).
- **Tests**: `GraphMERT/test/unit/test_extraction.jl`, `GraphMERT/test/wikipedia/` (e.g. relation count and type checks), and any integration test that checks relation extraction.
- **Optional**: `reports/WIKIPEDIA_DOMAIN_IMPLEMENTATION.md` or similar to document the new relation patterns and LLM option.

This proposal keeps the domain-agnostic core intact, pushes relation logic into the Wikipedia domain and the shared fallback, and aligns with `09-triple-extraction.md` and the existing prompts.
