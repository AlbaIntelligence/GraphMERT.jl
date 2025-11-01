# GraphMERT Generalization - Complete Plan & File List

## Summary

This document provides a complete plan for generalizing GraphMERT.jl to be domain-independent,
with separate pluggable modules for biomedical and Wikipedia domains.

## Immediate Action Plan

### Phase 1: Foundation (Week 1)
**Goal**: Set up domain abstraction layer

1. ✅ Create domain interface (`src/domains/interface.jl`)
2. ✅ Create domain registry (`src/domains/registry.jl`)
3. ⏳ Refactor `types.jl` to add domain field
4. ⏳ Refactor `config.jl` to support domain configuration

### Phase 2: Core Refactoring (Week 2)
**Goal**: Remove all hardcoded domain logic from core

5. ⏳ Refactor `api/extraction.jl` to use domain providers
6. ⏳ Refactor `llm/helper.jl` to use domain prompts
7. ⏳ Refactor `training/seed_injection.jl` to use domain entity linking
8. ⏳ Refactor `GraphMERT.jl` main module to use domain system

### Phase 3: Domain Modules (Week 3)
**Goal**: Create domain-specific modules

9. ⏳ Create biomedical domain module (`src/domains/biomedical/`)
10. ⏳ Create Wikipedia domain module (`src/domains/wikipedia/`)

### Phase 4: Cleanup & Testing (Week 4)
**Goal**: Move files, update examples, test

11. ⏳ Move domain-specific files to appropriate modules
12. ⏳ Update examples to use domain system
13. ⏳ Update documentation
14. ⏳ Run full test suite

## Files Created

### Domain Interface Layer
- ✅ `GraphMERT/src/domains/interface.jl` - Domain provider interface
- ✅ `GraphMERT/src/domains/registry.jl` - Domain registry

### Documentation
- ✅ `GENERALIZATION_PLAN.md` - Comprehensive generalization plan
- ✅ `IMPLEMENTATION_FILES.md` - File structure and organization
- ✅ `BIOMEDICAL_DOMAIN_IMPLEMENTATION.md` - Biomedical domain details
- ✅ `WIKIPEDIA_DOMAIN_IMPLEMENTATION.md` - Wikipedia domain details

## Files to Create (Next Steps)

### Biomedical Domain Module
1. `GraphMERT/src/domains/biomedical/domain.jl` - Main provider
2. `GraphMERT/src/domains/biomedical/entities.jl` - Entity extraction
3. `GraphMERT/src/domains/biomedical/relations.jl` - Relation extraction
4. `GraphMERT/src/domains/biomedical/validation.jl` - Validation rules
5. `GraphMERT/src/domains/biomedical/confidence.jl` - Confidence calculation
6. `GraphMERT/src/domains/biomedical/prompts.jl` - LLM prompts
7. `GraphMERT/src/domains/biomedical/umls.jl` - UMLS integration (move)
8. `GraphMERT/src/domains/biomedical/evaluation.jl` - Evaluation (move)
9. `GraphMERT/src/domains/biomedical/graph.jl` - Graph structures (move)
10. `GraphMERT/src/domains/biomedical/pubmed.jl` - PubMed processing (move)

### Wikipedia Domain Module
11. `GraphMERT/src/domains/wikipedia/domain.jl` - Main provider
12. `GraphMERT/src/domains/wikipedia/entities.jl` - Entity extraction
13. `GraphMERT/src/domains/wikipedia/relations.jl` - Relation extraction
14. `GraphMERT/src/domains/wikipedia/validation.jl` - Validation rules
15. `GraphMERT/src/domains/wikipedia/confidence.jl` - Confidence calculation
16. `GraphMERT/src/domains/wikipedia/prompts.jl` - LLM prompts
17. `GraphMERT/src/domains/wikipedia/wikidata.jl` - Wikidata integration (optional)

## Files to Modify

### Core Modules
- `GraphMERT/src/GraphMERT.jl` - Update to use domain system
- `GraphMERT/src/types.jl` - Add domain field, remove biomedical specializations
- `GraphMERT/src/api/extraction.jl` - Use domain providers
- `GraphMERT/src/llm/helper.jl` - Use domain prompts
- `GraphMERT/src/training/seed_injection.jl` - Use domain entity linking
- `GraphMERT/src/config.jl` - Add domain configuration

## Files to Move

- `GraphMERT/src/biomedical/entities.jl` → `GraphMERT/src/domains/biomedical/entities.jl`
- `GraphMERT/src/biomedical/relations.jl` → `GraphMERT/src/domains/biomedical/relations.jl`
- `GraphMERT/src/biomedical/umls.jl` → `GraphMERT/src/domains/biomedical/umls.jl`
- `GraphMERT/src/graphs/biomedical.jl` → `GraphMERT/src/domains/biomedical/graph.jl`
- `GraphMERT/src/evaluation/diabetes.jl` → `GraphMERT/src/domains/biomedical/evaluation.jl`
- `GraphMERT/src/text/pubmed.jl` → `GraphMERT/src/domains/biomedical/pubmed.jl`

## Key Design Decisions

### 1. Domain Provider Interface
- All domains must implement `DomainProvider` abstract type
- Required methods: entity/relation extraction, validation, confidence
- Optional methods: entity linking, seed triples, evaluation metrics, prompts

### 2. Domain Registry
- Central registry for all domains
- Default domain can be set
- Domains registered at module load time

### 3. Backward Compatibility
- Default to "biomedical" domain if not specified
- Provide compatibility layer for old API
- Deprecation warnings for old functions

### 4. Separation of Concerns
- Core modules: Domain-agnostic algorithms
- Domain modules: Domain-specific logic
- Clear boundaries between core and domain

## Testing Strategy

1. **Unit Tests**: Test domain interface and registry
2. **Integration Tests**: Test domain providers with core modules
3. **Biomedical Tests**: Ensure all biomedical functionality still works
4. **Wikipedia Tests**: Test new Wikipedia domain functionality
5. **Cross-Domain Tests**: Test switching between domains

## Migration Path

### For Existing Users

1. **Immediate**: Code continues to work with default "biomedical" domain
2. **Transition**: Update code to explicitly specify domain
3. **Future**: Use new domain-specific modules directly

### Example Migration

**Before**:
```julia
using GraphMERT
graph = extract_knowledge_graph(text)  # Uses biomedical implicitly
```

**After**:
```julia
using GraphMERT
using GraphMERT.Domains.Biomedical

# Register domain
biomedical_domain = BiomedicalDomain()
register_domain!("biomedical", biomedical_domain)

# Use with explicit domain
graph = extract_knowledge_graph(text, domain="biomedical")
```

## Success Criteria

✅ No hardcoded domain logic in core modules
✅ Domain can be switched at runtime
✅ New domains can be added without modifying core
✅ All existing biomedical functionality preserved
✅ Wikipedia domain fully functional
✅ Examples work with both domains
✅ Documentation updated

## Next Immediate Steps

1. **Review** this plan and implementation files
2. **Start** with Phase 1: Refactor `types.jl` and `config.jl`
3. **Create** biomedical domain module structure
4. **Create** Wikipedia domain module structure
5. **Test** each phase before moving to next

## Questions to Resolve

1. Should domain be a required parameter or optional (default to biomedical)?
   - **Recommendation**: Optional with default "biomedical" for backward compatibility

2. Should domains be loaded automatically or manually registered?
   - **Recommendation**: Manual registration for explicit control

3. Should domain-specific models be supported?
   - **Recommendation**: Yes, via domain configuration

4. How to handle domain-specific evaluation metrics?
   - **Recommendation**: Via optional `create_evaluation_metrics` method

## Contact & Support

For questions about this generalization plan, refer to:
- `GENERALIZATION_PLAN.md` - Detailed plan
- `BIOMEDICAL_DOMAIN_IMPLEMENTATION.md` - Biomedical specifics
- `WIKIPEDIA_DOMAIN_IMPLEMENTATION.md` - Wikipedia specifics
