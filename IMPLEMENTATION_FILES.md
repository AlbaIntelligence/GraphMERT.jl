# GraphMERT Generalization Implementation Files

## File Structure

### Created Files

#### Domain Interface Layer
1. ✅ `GraphMERT/src/domains/interface.jl` - Domain provider interface
2. ✅ `GraphMERT/src/domains/registry.jl` - Domain registry

#### Biomedical Domain Module (to be created)
3. `GraphMERT/src/domains/biomedical/domain.jl` - Main biomedical domain provider
4. `GraphMERT/src/domains/biomedical/entities.jl` - Biomedical entity types and extraction
5. `GraphMERT/src/domains/biomedical/relations.jl` - Biomedical relation types and extraction
6. `GraphMERT/src/domains/biomedical/validation.jl` - Biomedical validation rules
7. `GraphMERT/src/domains/biomedical/confidence.jl` - Biomedical confidence calculation
8. `GraphMERT/src/domains/biomedical/prompts.jl` - LLM prompts for biomedical domain
9. `GraphMERT/src/domains/biomedical/umls.jl` - UMLS integration (move from src/biomedical/)
10. `GraphMERT/src/domains/biomedical/evaluation.jl` - Diabetes evaluation (move from src/evaluation/)
11. `GraphMERT/src/domains/biomedical/graph.jl` - Biomedical graph structures (move from src/graphs/)
12. `GraphMERT/src/domains/biomedical/pubmed.jl` - PubMed processing (move from src/text/)

#### Wikipedia Domain Module (to be created)
13. `GraphMERT/src/domains/wikipedia/domain.jl` - Main Wikipedia domain provider
14. `GraphMERT/src/domains/wikipedia/entities.jl` - Wikipedia entity types and extraction
15. `GraphMERT/src/domains/wikipedia/relations.jl` - Wikipedia relation types and extraction
16. `GraphMERT/src/domains/wikipedia/validation.jl` - Wikipedia validation rules
17. `GraphMERT/src/domains/wikipedia/confidence.jl` - Wikipedia confidence calculation
18. `GraphMERT/src/domains/wikipedia/prompts.jl` - LLM prompts for Wikipedia domain
19. `GraphMERT/src/domains/wikipedia/wikidata.jl` - Wikidata integration (optional)

### Files to Modify

#### Core Modules
- `GraphMERT/src/GraphMERT.jl` - Update to use domain system
- `GraphMERT/src/types.jl` - Add domain field, remove biomedical specializations
- `GraphMERT/src/api/extraction.jl` - Use domain providers instead of hardcoded logic
- `GraphMERT/src/llm/helper.jl` - Use domain providers for prompts
- `GraphMERT/src/training/seed_injection.jl` - Use domain providers for entity linking
- `GraphMERT/src/config.jl` - Add domain configuration

### Files to Move (Refactor)

- `GraphMERT/src/biomedical/entities.jl` → `GraphMERT/src/domains/biomedical/entities.jl`
- `GraphMERT/src/biomedical/relations.jl` → `GraphMERT/src/domains/biomedical/relations.jl`
- `GraphMERT/src/biomedical/umls.jl` → `GraphMERT/src/domains/biomedical/umls.jl`
- `GraphMERT/src/graphs/biomedical.jl` → `GraphMERT/src/domains/biomedical/graph.jl`
- `GraphMERT/src/evaluation/diabetes.jl` → `GraphMERT/src/domains/biomedical/evaluation.jl`
- `GraphMERT/src/text/pubmed.jl` → `GraphMERT/src/domains/biomedical/pubmed.jl`

### Files to Delete (After Migration)

- `GraphMERT/src/biomedical/` directory (after moving files)
- References to biomedical-specific code in core modules

## Implementation Priority

### Phase 1: Foundation (HIGH PRIORITY)
1. Domain interface implementation ✅
2. Domain registry implementation ✅
3. Refactor types.jl to be domain-agnostic

### Phase 2: Core Refactoring (HIGH PRIORITY)
4. Refactor api/extraction.jl
5. Refactor llm/helper.jl
6. Refactor training/seed_injection.jl
7. Refactor GraphMERT.jl main module

### Phase 3: Domain Modules (MEDIUM PRIORITY)
8. Create biomedical domain module
9. Create Wikipedia domain module

### Phase 4: Cleanup (LOW PRIORITY)
10. Move domain-specific files
11. Update examples
12. Update documentation
