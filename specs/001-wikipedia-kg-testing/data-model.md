# Data Model: Wikipedia Knowledge Graph Testing

**Date**: 2026-03-10
**Feature**: Wikipedia KG Testing (French Monarchy)

## Test Data Entities

### TestDataset

A collection of Wikipedia articles used for testing.

| Field | Type | Description |
|-------|------|-------------|
| articles | Vector{String} | Raw Wikipedia article text |
| metadata | Dict | Article titles, URLs, word counts |
| reference_facts | Vector{ReferenceFact} | Known facts for validation |

### ReferenceFact

A known fact about French monarchy for validation.

| Field | Type | Description |
|-------|------|-------------|
| subject | String | Entity name (e.g., "Louis XIV") |
| predicate | String | Relation type (e.g., "reigned_after") |
| object | String | Target entity (e.g., "Louis XIII") |
| source | String | Wikipedia article source |
| verified | Bool | Whether fact is historically verified |

### QualityMetrics

Evaluation results for extracted knowledge graphs.

| Field | Type | Description |
|-------|------|-------------|
| entity_precision | Float64 | Precision of entity extraction |
| entity_recall | Float64 | Recall of entity extraction |
| relation_precision | Float64 | Precision of relation extraction |
| relation_recall | Float64 | Recall of relation extraction |
| fact_capture_rate | Float64 | Percentage of reference facts captured |

## Test Article Selection

Selected Wikipedia articles for French monarchy testing:

1. **Louis XIV of France** - Longest-reigning monarch, complex relations
2. **Henry IV of France** - First Bourbon king
3. **Louis IX of France** - Saint Louis, Crusades
4. **French Revolution** - End of monarchy context
5. **Marie Antoinette** - Consort relations
6. **Capetian dynasty** - Foundational dynasty
7. **Bourbon dynasty** - Major dynasty
8. **Louis XV of France** - Transition period
9. **Louis XVI of France** - Last absolute king
10. **List of French monarchs** - Comprehensive reference

## Validation Rules

1. Entity extraction must classify persons, locations, titles correctly
2. Relation extraction must identify family (parent, spouse) and temporal (reigned) relations
3. Confidence scores must correlate with extraction quality (AUC > 0.7)
4. Extracted facts must match reference facts within tolerance

## State Transitions

```
Test Dataset → Entity Extraction → Relation Extraction → KG Construction → Quality Assessment
                                                                      ↓
                                                            Export (JSON/CSV)
```
