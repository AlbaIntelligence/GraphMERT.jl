# GraphMERT Generalization Plan

## Executive Summary

This document outlines a comprehensive plan to generalize GraphMERT.jl to be completely domain-independent, with domain-specific functionality moved to separate, pluggable modules.

## Current State Analysis

### Domain-Specific Dependencies Identified

#### 1. **Core Module Dependencies** (`GraphMERT.jl`)
- ❌ Hardcoded biomedical exports
- ❌ Direct includes of biomedical modules
- ❌ Biomedical-specific fallback functions

#### 2. **Type System** (`types.jl`)
- ✅ Generic `Entity` and `Relation` types (GOOD)
- ❌ `BiomedicalEntity` and `BiomedicalRelation` wrappers (should be domain-specific)
- ✅ Generic `KnowledgeGraph` (GOOD)
- ❌ Domain-specific entity type registry initialization

#### 3. **API/Extraction** (`api/extraction.jl`)
- ❌ Hardcoded biomedical patterns (`extract_biomedical_terms`)
- ❌ Biomedical-specific entity discovery
- ❌ Domain-specific relation matching
- ❌ Domain-specific confidence calculations

#### 4. **Helper LLM** (`llm/helper.jl`)
- ❌ Hardcoded biomedical prompts
- ❌ Domain-specific entity discovery prompts
- ❌ Domain-specific relation matching prompts

#### 5. **Seed Injection** (`training/seed_injection.jl`)
- ❌ UMLS-specific entity linking
- ❌ Hardcoded UMLS concepts
- ❌ Domain-specific triple selection

#### 6. **Graph Structures** (`graphs/biomedical.jl`)
- ❌ Entirely biomedical-specific
- ❌ UMLS-specific mappings

#### 7. **Text Processing** (`text/pubmed.jl`)
- ❌ Entirely PubMed-specific

#### 8. **Evaluation** (`evaluation/diabetes.jl`)
- ❌ Entirely diabetes-domain specific

#### 9. **Utils** (`utils.jl`)
- ✅ Mostly generic (GOOD)
- ❌ Domain-specific patterns in some functions

## Generalization Strategy

### Phase 1: Domain Abstraction Layer

Create a pluggable domain interface that allows domains to register:
1. Entity types and their validation rules
2. Relation types and their validation rules
3. Entity extraction patterns
4. Relation extraction patterns
5. Confidence calculation strategies
6. Entity linking mechanisms (if domain-specific)
7. Seed KG formats and injection strategies
8. Domain-specific evaluation metrics

### Phase 2: Core Refactoring

1. Remove all hardcoded domain logic from core modules
2. Replace with domain interface calls
3. Make domain a required parameter to all extraction functions
4. Create domain registry system

### Phase 3: Domain Module Creation

1. Create `GraphMERT.domains.biomedical` module
2. Create `GraphMERT.domains.wikipedia` module
3. Move all domain-specific code to respective modules

### Phase 4: Testing & Validation

1. Ensure biomedical functionality still works
2. Test Wikipedia domain functionality
3. Verify no domain-specific code in core

## Detailed Task Breakdown

### Task 1: Create Domain Interface (`src/domains/interface.jl`)

**Purpose**: Define abstract interface that all domains must implement

**Key Components**:
- `DomainConfig` - Domain configuration structure
- `DomainProvider` - Abstract type for domain implementations
- Required methods:
  - `register_entity_types(domain)`
  - `register_relation_types(domain)`
  - `extract_entities(domain, text, config)`
  - `extract_relations(domain, entities, text, config)`
  - `validate_entity(domain, entity_text, entity_type)`
  - `validate_relation(domain, head, relation, tail)`
  - `calculate_entity_confidence(domain, entity_text, entity_type, context)`
  - `calculate_relation_confidence(domain, head, relation, tail, context)`
  - `link_entity(domain, entity_text, config)` [optional]
  - `create_seed_triples(domain, entity_text, config)` [optional]
  - `create_evaluation_metrics(domain, kg)` [optional]

**Files to Create**:
- `src/domains/interface.jl`

### Task 2: Create Domain Registry (`src/domains/registry.jl`)

**Purpose**: Central registry for domain implementations

**Key Components**:
- `DOMAIN_REGISTRY` - Global registry
- `register_domain!(domain_name, provider)`
- `get_domain(domain_name)`
- `list_domains()`
- `set_default_domain(domain_name)`

**Files to Create**:
- `src/domains/registry.jl`

### Task 3: Refactor Core Types (`src/types.jl`)

**Changes**:
- Remove `BiomedicalEntity` and `BiomedicalRelation` specializations
- Keep only generic `Entity` and `Relation`
- Add `domain::String` field to `Entity` and `Relation`
- Update `GraphMERTConfig` to require `domain::String`
- Update `ProcessingOptions` to require `domain::String`

**Files to Modify**:
- `src/types.jl`

### Task 4: Refactor API Extraction (`src/api/extraction.jl`)

**Changes**:
- Remove `extract_biomedical_terms` function
- Replace with `extract_entities(domain, text, config)`
- Remove hardcoded biomedical patterns
- Use domain provider for entity extraction
- Use domain provider for relation extraction
- Use domain provider for confidence calculations

**Files to Modify**:
- `src/api/extraction.jl`

### Task 5: Refactor Helper LLM (`src/llm/helper.jl`)

**Changes**:
- Remove hardcoded biomedical prompts
- Add `create_prompt(domain, task_type, context)` method to domain interface
- Use domain provider for prompt generation

**Files to Modify**:
- `src/llm/helper.jl`

### Task 6: Refactor Seed Injection (`src/training/seed_injection.jl`)

**Changes**:
- Remove UMLS-specific code
- Add `link_entity(domain, entity_text, config)` to domain interface
- Add `create_seed_triples(domain, entity_text, config)` to domain interface
- Use domain provider for entity linking
- Use domain provider for triple selection

**Files to Modify**:
- `src/training/seed_injection.jl`

### Task 7: Refactor Main Module (`src/GraphMERT.jl`)

**Changes**:
- Remove biomedical-specific exports
- Remove biomedical module includes
- Add domain module includes
- Add domain registry initialization
- Update main API to require domain parameter

**Files to Modify**:
- `src/GraphMERT.jl`

### Task 8: Create Biomedical Domain Module

**Structure**:
```
src/domains/biomedical/
├── domain.jl           # Main domain provider
├── entities.jl         # Entity types and extraction
├── relations.jl       # Relation types and extraction
├── umls.jl            # UMLS integration (moved from src/biomedical/)
├── validation.jl      # Domain-specific validation
├── confidence.jl      # Confidence calculation
└── prompts.jl         # LLM prompts for biomedical domain
```

**Files to Create**:
- `src/domains/biomedical/domain.jl`
- `src/domains/biomedical/entities.jl`
- `src/domains/biomedical/relations.jl`
- `src/domains/biomedical/umls.jl` (move from `src/biomedical/umls.jl`)
- `src/domains/biomedical/validation.jl`
- `src/domains/biomedical/confidence.jl`
- `src/domains/biomedical/prompts.jl`

**Files to Move**:
- `src/biomedical/entities.jl` → `src/domains/biomedical/entities.jl`
- `src/biomedical/relations.jl` → `src/domains/biomedical/relations.jl`
- `src/biomedical/umls.jl` → `src/domains/biomedical/umls.jl`

### Task 9: Create Wikipedia Domain Module

**Structure**:
```
src/domains/wikipedia/
├── domain.jl           # Main domain provider
├── entities.jl         # Entity types and extraction
├── relations.jl        # Relation types and extraction
├── validation.jl       # Domain-specific validation
├── confidence.jl       # Confidence calculation
└── prompts.jl          # LLM prompts for Wikipedia domain
```

**Files to Create**:
- `src/domains/wikipedia/domain.jl`
- `src/domains/wikipedia/entities.jl`
- `src/domains/wikipedia/relations.jl`
- `src/domains/wikipedia/validation.jl`
- `src/domains/wikipedia/confidence.jl`
- `src/domains/wikipedia/prompts.jl`

### Task 10: Move Domain-Specific Evaluation

**Changes**:
- Move `evaluation/diabetes.jl` to `domains/biomedical/evaluation.jl`
- Create generic evaluation interface
- Domain-specific evaluation metrics register with domain

**Files to Move**:
- `src/evaluation/diabetes.jl` → `src/domains/biomedical/evaluation.jl`

**Files to Modify**:
- `src/evaluation/factscore.jl` (keep generic)
- `src/evaluation/validity.jl` (keep generic)

### Task 11: Move Domain-Specific Graph Structures

**Changes**:
- Move `graphs/biomedical.jl` to `domains/biomedical/graph.jl`
- Keep `graphs/leafy_chain.jl` (generic)

**Files to Move**:
- `src/graphs/biomedical.jl` → `src/domains/biomedical/graph.jl`

### Task 12: Move Domain-Specific Text Processing

**Changes**:
- Move `text/pubmed.jl` to `domains/biomedical/pubmed.jl`
- Keep `text/tokenizer.jl` (generic)

**Files to Move**:
- `src/text/pubmed.jl` → `src/domains/biomedical/pubmed.jl`

### Task 13: Update Examples

**Changes**:
- Update biomedical examples to use `domain="biomedical"`
- Update Wikipedia examples to use `domain="wikipedia"`
- Ensure examples work with new domain system

**Files to Modify**:
- `examples/biomedical/*.jl`
- `examples/wikipedia/*.jl`

### Task 14: Update Documentation

**Changes**:
- Update README.md
- Update API documentation
- Create domain development guide
- Update examples documentation

**Files to Modify**:
- `README.md`
- `EXAMPLES_README.md`
- Create `docs/domains/development_guide.md`

## Implementation Order

### Week 1: Foundation
1. Task 1: Create Domain Interface
2. Task 2: Create Domain Registry
3. Task 3: Refactor Core Types

### Week 2: Core Refactoring
4. Task 4: Refactor API Extraction
5. Task 5: Refactor Helper LLM
6. Task 6: Refactor Seed Injection
7. Task 7: Refactor Main Module

### Week 3: Domain Modules
8. Task 8: Create Biomedical Domain Module
9. Task 9: Create Wikipedia Domain Module

### Week 4: Cleanup & Testing
10. Task 10: Move Domain-Specific Evaluation
11. Task 11: Move Domain-Specific Graph Structures
12. Task 12: Move Domain-Specific Text Processing
13. Task 13: Update Examples
14. Task 14: Update Documentation

## Migration Strategy

### Backward Compatibility

To maintain backward compatibility during migration:

1. Create a compatibility layer that defaults to "biomedical" domain
2. Deprecate old biomedical-specific functions
3. Provide migration guide for users

### Testing Strategy

1. Create test suite for domain interface
2. Ensure all biomedical tests still pass
3. Create new tests for Wikipedia domain
4. Integration tests for domain switching

## Success Criteria

✅ No hardcoded domain-specific logic in core modules
✅ Domain can be switched at runtime
✅ New domains can be added without modifying core
✅ All existing biomedical functionality preserved
✅ Wikipedia domain fully functional
✅ Examples work with both domains
✅ Documentation updated

## Risk Mitigation

- **Risk**: Breaking existing biomedical functionality
  - **Mitigation**: Comprehensive test suite, gradual migration, backward compatibility layer

- **Risk**: Performance degradation
  - **Mitigation**: Benchmark before/after, optimize domain interface calls

- **Risk**: API changes break user code
  - **Mitigation**: Deprecation warnings, migration guide, version bump

## Future Enhancements

- Support for multiple domains simultaneously
- Domain-specific model fine-tuning
- Domain-specific preprocessing pipelines
- Domain-specific evaluation metrics
- Domain marketplace/plugin system
