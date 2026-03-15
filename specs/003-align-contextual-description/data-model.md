# Data Model: Reliability Pipeline (003-align-contextual-description)

**Status**: Design for implementation  
**Last Updated**: 2026-03-15  
**Feature**: [spec.md](spec.md)

---

## Overview

This document defines the entities and relationships added or extended for the reliability pipeline: provenance, validity, factuality, KG cleaning, and iterative seed. Existing types (`KnowledgeGraph`, `KnowledgeEntity`, `KnowledgeRelation`, `Entity`, `Relation`) remain; we add structured provenance and result types for validation and cleaning.

---

## 1. ProvenanceRecord (new or extended)

Attached to each triple so users can trace it to source text.

| Field | Type | Description |
|-------|------|-------------|
| document_id | String | Identifier of the source document (e.g., corpus id, path, or hash) |
| segment_id | String or Int | Sentence or segment identifier within the document |
| span_start | Optional Int | Character start offset in document (optional) |
| span_end | Optional Int | Character end offset in document (optional) |
| context | Optional String | Short snippet or context (optional) |

**Validation**: document_id must be non-empty when provenance is required. segment_id or (span_start, span_end) must be present to satisfy “traceable to source span.”

**Relationship**: One-to-one with a triple (Relation / KnowledgeRelation). Existing `provenance::String` may remain for backward compatibility; structured ProvenanceRecord is the canonical source for document_id and segment_id.

---

## 2. Triple (existing Relation / KnowledgeRelation, extended)

A (head, relation_type, tail) fact. Must carry provenance when the reliability pipeline is enabled.

**Existing fields** (unchanged): head, tail, relation_type, confidence, attributes, etc.

**Extension**: Each relation has an associated **ProvenanceRecord** (or equivalent structured data). Extraction pipeline MUST populate it at creation time (FR-001).

**State**: No state transitions; immutable after creation. Cleaning produces new triples (or removes them), not in-place mutation.

---

## 3. Seed ontology / seed KG (domain-provided)

Domain schema and optional known triples used for validation and training.

| Concept | Description |
|---------|-------------|
| Relation types | Allowed relation type names (e.g., TREATS, ASSOCIATED_WITH) |
| Entity types | Allowed entity type names or labels |
| Domain/range | For each relation type, allowed head/tail entity types (optional) |

**Source**: DomainProvider (e.g., biomedical, Wikipedia) exposes ontology or schema. When missing or incomplete, validation is skipped or relaxed with clear indication (FR-008).

**Relationship**: Input to validation (ValidityScore); input to seed injection and iterative improvement.

---

## 4. ValidityReport / ValidityScore

Result of checking triples against the seed ontology.

| Field | Type | Description |
|-------|------|-------------|
| score | Float | Proportion (0–1) or normalized score of triples that pass ontology checks |
| total_triples | Int | Number of triples checked |
| valid_count | Int | Number of triples that satisfy ontology |
| per_triple | Optional | Per-triple valid/invalid and reason (optional) |
| ontology_id | Optional String | Identifier of ontology used |

**Validation**: score = valid_count / total_triples when total_triples > 0; else undefined or 0. Produced by validation function (FR-002).

---

## 5. FactualityScore

Result of comparing triples to reference or ground-truth data.

| Field | Type | Description |
|-------|------|-------------|
| score | Float | Proportion of triples deemed factually correct (e.g., match reference) |
| total_triples | Int | Number of triples evaluated |
| correct_count | Int | Number of triples matching reference |
| reference_id | Optional String | Identifier of reference dataset used |

**Validation**: Produced only when reference data is available (FR-003). When reference is absent, no FactualityScore is produced.

---

## 6. CleaningPolicy (configuration)

Configurable rules for the KG cleaning step.

| Concept | Description |
|---------|-------------|
| min_confidence | Float | Triples below this confidence are removed or flagged |
| require_provenance | Bool | Remove triples without valid provenance |
| contradiction_handling | Enum or strategy | How to handle contradicted triples (remove, flag, etc.) |
| output_mode | Optional | Drop vs. flag only; optional rectification rules |

**Relationship**: Input to KG cleaning; output is a cleaned KG (new instance).

---

## 7. Cleaned KG

A knowledge graph produced by applying the cleaning step to an extracted KG.

**Representation**: Same type as KnowledgeGraph (same entities/relations structure). Difference is semantic: only triples that pass the cleaning policy remain. May carry metadata indicating it is the output of cleaning (e.g., cleaning_policy_applied, before/after counts).

**Relationship**: Can be used as augmented seed for a subsequent run (FR-006, iterative improvement).

---

## 8. State and lifecycle (summary)

- **Extraction**: Text → (model + options) → KnowledgeGraph with triples; each triple has ProvenanceRecord.
- **Validation**: KnowledgeGraph + ontology → ValidityReport (ValidityScore).
- **Factuality**: KnowledgeGraph + reference_data → FactualityScore (when reference exists).
- **Cleaning**: KnowledgeGraph + CleaningPolicy → KnowledgeGraph (cleaned).
- **Iterative seed**: Cleaned (or curated) KnowledgeGraph → configured as seed → next extraction or training run.

No new persistent storage entities; all in-memory or file export as already supported.
