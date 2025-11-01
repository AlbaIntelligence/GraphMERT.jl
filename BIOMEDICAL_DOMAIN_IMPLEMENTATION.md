# Biomedical Domain Module Implementation

This document outlines the biomedical domain module implementation.

## Files to Create

### 1. `domain.jl` - Main Domain Provider

Implements `DomainProvider` for biomedical domain. Acts as the entry point
and coordinates all biomedical-specific functionality.

**Key Responsibilities**:
- Implement all required DomainProvider methods
- Coordinate between biomedical submodules
- Initialize biomedical entity and relation types
- Provide domain name and configuration

### 2. `entities.jl` - Entity Types and Extraction

**Moved from**: `src/biomedical/entities.jl`

**Key Functions**:
- `register_biomedical_entity_types()` - Register DISEASE, DRUG, PROTEIN, etc.
- `extract_biomedical_entities(text, config)` - Extract entities using patterns
- `validate_biomedical_entity(text, type)` - Validate entity against type
- `calculate_biomedical_entity_confidence(text, type, context)` - Calculate confidence

**Changes Needed**:
- Remove dependency on global registry, use domain-specific registry
- Make functions work with domain interface
- Keep all biomedical-specific patterns and validation logic

### 3. `relations.jl` - Relation Types and Extraction

**Moved from**: `src/biomedical/relations.jl`

**Key Functions**:
- `register_biomedical_relation_types()` - Register TREATS, CAUSES, etc.
- `extract_biomedical_relations(entities, text, config)` - Extract relations
- `validate_biomedical_relation(head, relation, tail)` - Validate relation
- `calculate_biomedical_relation_confidence(head, relation, tail, context)` - Calculate confidence

**Changes Needed**:
- Remove dependency on global registry
- Make functions work with domain interface
- Keep all biomedical-specific relation patterns

### 4. `validation.jl` - Validation Rules

**New file** - Extract validation logic from entities.jl and relations.jl

**Key Functions**:
- `validate_entity_type(entity_text, entity_type)` - Type-specific validation
- `validate_relation_type(head_type, relation, tail_type)` - Relation compatibility
- `validate_entity_patterns(entity_text, entity_type)` - Pattern matching

### 5. `confidence.jl` - Confidence Calculation

**New file** - Extract confidence calculation logic

**Key Functions**:
- `calculate_base_confidence(entity_text, entity_type)` - Base confidence
- `calculate_pattern_confidence(entity_text, patterns)` - Pattern-based confidence
- `calculate_umls_confidence(entity_text, umls_result)` - UMLS-enhanced confidence

### 6. `prompts.jl` - LLM Prompts

**New file** - Extract LLM prompt generation from `llm/helper.jl`

**Key Functions**:
- `create_entity_discovery_prompt(text, domain)` - Biomedical entity discovery
- `create_relation_matching_prompt(entities, text, domain)` - Relation matching
- `create_tail_formation_prompt(tokens, text, domain)` - Tail entity formation

**Content**: Biomedical-specific prompts referencing medical terminology, UMLS, etc.

### 7. `umls.jl` - UMLS Integration

**Moved from**: `src/biomedical/umls.jl`

**Key Functions**:
- `link_entity_to_umls(entity_text, config)` - Link entity to UMLS CUI
- `get_umls_concept_details(cui)` - Get concept details
- `search_umls_concepts(query)` - Search UMLS

**No changes needed** - Keep as-is

### 8. `evaluation.jl` - Domain Evaluation

**Moved from**: `src/evaluation/diabetes.jl`

**Key Functions**:
- `evaluate_icd_benchmark(kg, benchmark_data)` - ICD-Bench evaluation
- `evaluate_medmcqa(kg, benchmark_data)` - MedMCQA evaluation
- `run_diabetes_benchmark_evaluation(kg)` - Combined evaluation

**No changes needed** - Keep as-is

### 9. `graph.jl` - Biomedical Graph Structures

**Moved from**: `src/graphs/biomedical.jl`

**Key Functions**:
- `build_biomedical_graph(entities, relations, umls_client)` - Build graph with UMLS
- `analyze_biomedical_graph(graph)` - Analyze graph metrics
- `filter_by_biomedical_criteria(graph, criteria)` - Domain-specific filtering

**No changes needed** - Keep as-is

### 10. `pubmed.jl` - PubMed Processing

**Moved from**: `src/text/pubmed.jl`

**Key Functions**:
- `create_pubmed_client(api_key)` - Create PubMed client
- `search_pubmed(query, client)` - Search PubMed
- `process_pubmed_article(article)` - Process article text

**No changes needed** - Keep as-is

## Integration Points

The biomedical domain module must integrate with:

1. **Domain Interface**: Implement all required DomainProvider methods
2. **Domain Registry**: Register as "biomedical" domain
3. **Core Modules**: Used by extraction, training, evaluation modules
4. **UMLS**: External API integration for entity linking
5. **Helper LLM**: Domain-specific prompt generation

## Usage Example

```julia
using GraphMERT
using GraphMERT.Domains.Biomedical

# Register domain
biomedical_domain = BiomedicalDomain()
register_domain!("biomedical", biomedical_domain)

# Use domain
text = "Diabetes is treated with metformin."
domain = get_domain("biomedical")
entities = extract_entities(domain, text, ProcessingOptions())
relations = extract_relations(domain, entities, text, ProcessingOptions())
```
