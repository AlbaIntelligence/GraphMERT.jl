# Test Contract: Wikipedia Domain Extraction

**Date**: 2026-03-10

## Contract Overview

This document defines the expected behavior of the Wikipedia domain extraction for testing purposes.

## Entity Extraction Contract

### Input
- `text`: String containing Wikipedia article content
- `domain`: WikipediaDomain instance
- `options`: ProcessingOptions with domain="wikipedia"

### Expected Output
- `Vector{Entity}`: Extracted entities with:
  - `id`: Unique identifier
`: Entity  - `text text
  - `entity_type`: "PERSON", "LOCATION", "TITLE", "ORGANIZATION", "DATE"
  - `confidence`: Float64 between 0 and 1
  - `domain`: "wikipedia"

### Contract Rules

1. **Entity Presence**: MUST extract at least one entity from valid Wikipedia text
2. **Type Classification**: MUST classify persons, locations, and titles correctly for French monarchy text
3. **Confidence**: MUST provide confidence scores between 0 and 1
4. **No Duplicates**: MUST NOT return duplicate entities with same text and type

## Relation Extraction Contract

### Input
- `entities`: Vector{Entity} from entity extraction
- `text`: Original Wikipedia text
- `domain`: WikipediaDomain instance

### Expected Output
- `Vector{Relation}`: Extracted relations with:
  - `head`: Source entity ID
  - `tail`: Target entity ID
  - `relation_type`: "PARENT_OF", "SPOUSE_OF", "REIGNED_AFTER", "BORN_IN", "DIED_IN", etc.
  - `confidence`: Float64 between 0 and 1

### Contract Rules

1. **Valid Relations**: MUST only produce valid relation types
2. **Entity References**: All head/tail must reference existing entities
3. **Bidirectional**: Family relations should be extractable in both directions when context supports

## Knowledge Graph Contract

### Input
- Text or entities+relations

### Expected Output
- `KnowledgeGraph` with:
  - `entities`: Vector{KnowledgeEntity}
  - `relations`: Vector{KnowledgeRelation}
  - `metadata`: Dict with extraction details

### Contract Rules

1. **Consistency**: All relation references must point to existing entities
2. **Graph Validity**: No self-loops in relations
3. **Export**: Must support JSON and CSV export formats

## Quality Metrics Contract

### Input
- Extracted KnowledgeGraph
- Reference facts vector

### Expected Output
- `QualityMetrics` with:
  - `entity_precision`: Float64
  - `entity_recall`: Float64  
  - `relation_precision`: Float64
  - `relation_recall`: Float64
  - `fact_capture_rate`: Float64

### Contract Rules

1. **Metrics Range**: All metrics must be between 0 and 1
2. **Baseline**: On French monarchy articles, MUST achieve:
   - Entity precision >= 0.70
   - Relation precision >= 0.60
3. **Confidence Correlation**: High-confidence extractions should have higher precision (AUC > 0.7)
