# 11: Data Structures (Short)

Short summary of the core data structures used in GraphMERT. For full field lists and invariants, see `11-data-structures-FULL.md`.

---

## 1. Text and positions

- `TextPosition`: simple struct for character/line/column positions.

---

## 2. Knowledge graph layer

- `KnowledgeEntity`:
  - ID, text, label, confidence, `TextPosition`, attributes, timestamp.
- `KnowledgeRelation`:
  - Head/tail IDs, relation type, confidence, attributes, timestamp.
- `KnowledgeGraph`:
  - Vectors of entities/relations, metadata dict, created_at.

Convenience constructors allow building a `KnowledgeGraph` from generic `Entity`/`Relation` vectors.

---

## 3. Generic entity/relation layer

- `Entity`:
  - Generic, domain-agnostic entity with `entity_type`, `domain`, attributes, and provenance.
- `Relation`:
  - Generic relation between entities with `relation_type`, `domain`, and provenance/evidence.

These are the primary types for domain providers and extraction.

---

## 4. Configuration and processing

- `GraphMERTConfig`: model-level configuration (encoder + H-GAT + attention).
- `ProcessingOptions`: extraction/runtime configuration (domain, lengths, thresholds, etc.).

---

## 5. Graph and training types

- Leafy chain types:
  - `ChainGraphNode`, `ChainGraphConfig`, `LeafyChainGraph`.
- MNM and seed injection:
  - `MNMConfig`, `MNMBatch`, `SemanticTriple`, `SeedInjectionConfig`.
- LLM/evaluation:
  - `LLMRequest`, `LLMResponse`, `EntityLinkingResult`, and evaluation score types.

Refer to `src/types.jl` for exact field definitions and to `*_FULL.md` specs for diagrams and extended discussion.

