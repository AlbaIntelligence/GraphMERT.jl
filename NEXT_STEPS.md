# GraphMERT Generalization - Next Steps

## ✅ Completed Work

### Phase 1: Foundation
- ✅ Domain interface (`domains/interface.jl`)
- ✅ Domain registry (`domains/registry.jl`)
- ✅ Types refactored with domain field
- ✅ Config updated for domain support

### Phase 2: Core Refactoring
- ✅ API extraction refactored to use domain providers
- ✅ LLM helper refactored to use domain prompts
- ✅ Seed injection refactored to use domain entity linking
- ✅ Main GraphMERT.jl module refactored

### Phase 3: Domain Modules
- ✅ Biomedical domain module created
- ✅ Wikipedia domain module created
- ✅ Both domains load and register successfully

## 🔄 Next Steps (Priority Order)

### 1. **Testing & Validation** (HIGH PRIORITY)

**Goal**: Ensure the domain system works end-to-end

- [ ] **Test biomedical domain extraction**
  - Test entity extraction with biomedical text
  - Test relation extraction
  - Verify UMLS integration works
  - Test with existing biomedical examples

- [ ] **Test Wikipedia domain extraction**
  - Test entity extraction with Wikipedia-style text
  - Test relation extraction
  - Verify both domains can be used simultaneously

- [ ] **Integration testing**
  - Test `extract_knowledge_graph` with both domains
  - Test domain switching at runtime
  - Test backward compatibility (default domain behavior)

- [ ] **Create test files**
  - `test/domains/test_biomedical_domain.jl`
  - `test/domains/test_wikipedia_domain.jl`
  - `test/domains/test_domain_registry.jl`

### 2. **Improve Biomedical Domain Module** (HIGH PRIORITY)

**Goal**: Remove dependency on old biomedical files

**Status**: ✅ **COMPLETED**

**Completed Tasks**:
- [x] Refactor `domains/biomedical/entities.jl` to be self-contained
  - ✅ Moved all entity extraction logic, enums, patterns, validation, and confidence calculation from `biomedical/entities.jl`
  - ✅ Removed dependency on old `biomedical/entities.jl` file
  - ✅ Includes `BiomedicalEntityType` enum for backward compatibility
  - ✅ All biomedical-specific patterns and validation preserved

- [x] Refactor `domains/biomedical/relations.jl` to be self-contained
  - ✅ Moved all relation extraction logic, enums, classification, validation, and confidence calculation from `biomedical/relations.jl`
  - ✅ Removed dependency on old `biomedical/relations.jl` file
  - ✅ Includes `BiomedicalRelationType` enum for backward compatibility
  - ✅ All biomedical-specific patterns preserved
  - ✅ Made string-based validation more lenient for domain interface

- [x] Verified `domains/biomedical/domain.jl` works correctly
  - ✅ All imports work correctly
  - ✅ All 55 biomedical domain tests passing
  - ✅ Domain creation and registration verified

### 3. **Move Domain-Specific Files** (MEDIUM PRIORITY)

**Goal**: Complete the migration of domain-specific code

**Status**: ✅ **COMPLETED**

**Completed Tasks**:
- [x] Move `graphs/biomedical.jl` → `domains/biomedical/graph.jl`
  - ✅ File moved to biomedical domain directory
  - ✅ Added MetaGraphs import for compatibility
  - ✅ Available for inclusion when needed

- [x] Move `evaluation/diabetes.jl` → `domains/biomedical/evaluation.jl`
  - ✅ File moved to biomedical domain directory
  - ✅ Available for inclusion when needed

- [x] Move `text/pubmed.jl` → `domains/biomedical/pubmed.jl`
  - ✅ File moved to biomedical domain directory
  - ✅ Available for inclusion when needed

- [x] Update `domains/biomedical.jl` loader
  - ✅ Added comments documenting the new module locations
  - ✅ Modules can be included on-demand when needed

**Note**: These modules are domain-specific and loaded on-demand rather than automatically. They can be included explicitly when needed:
- `include("GraphMERT/src/domains/biomedical/graph.jl")`
- `include("GraphMERT/src/domains/biomedical/evaluation.jl")`
- `include("GraphMERT/src/domains/biomedical/pubmed.jl")`

### 4. **Update Examples** (MEDIUM PRIORITY)

**Goal**: Update examples to use the domain system

**Status**: ✅ **PARTIALLY COMPLETED**

**Completed Tasks**:
- [x] **Updated biomedical example** (`examples/biomedical/01_basic_entity_extraction.jl`)
  - ✅ Updated to load and register biomedical domain
  - ✅ Uses domain system for entity extraction
  - ✅ Demonstrates domain provider usage
  - ✅ Shows confidence scoring and entity statistics

- [x] **Updated Wikipedia example** (`examples/wikipedia/01_wikipedia_entity_extraction.jl`)
  - ✅ Updated to use Wikipedia domain
  - ✅ Uses domain system for entity and relation extraction
  - ✅ Demonstrates domain provider usage
  - ✅ Shows entity and relation type distributions

- [x] **Created domain switching example** (`examples/00_domain_switching_demo.jl`)
  - ✅ Shows how to use multiple domains simultaneously
  - ✅ Demonstrates domain switching
  - ✅ Compares extraction results across domains
  - ✅ Shows default domain behavior

**Remaining Tasks**:
- [ ] Update remaining biomedical examples (`examples/biomedical/02_*.jl` through `14_*.jl`)
  - Update to use domain system where applicable
  - Test that they still work
  - Update documentation

### 5. **Enhanced Domain Features** (MEDIUM PRIORITY)

**Goal**: Complete optional domain features

- [ ] **Complete UMLS integration in biomedical domain**
  - Ensure `link_entity` returns proper format
  - Implement `create_seed_triples` fully
  - Test with real UMLS data (if available)

- [ ] **Implement Wikidata integration for Wikipedia domain**
  - Create `wikidata.jl` module
  - Implement `link_entity` for Wikidata
  - Implement `create_seed_triples` for Wikidata

- [ ] **Domain-specific evaluation metrics**
  - Implement `create_evaluation_metrics` for biomedical
  - Implement `create_evaluation_metrics` for Wikipedia
  - Update evaluation modules to use domain metrics

### 6. **Documentation & Examples** (LOW PRIORITY)

**Goal**: Improve usability and documentation

- [ ] **Update main README.md**
  - Add domain system overview
  - Update quick start examples
  - Add domain switching examples

- [ ] **Create domain developer guide**
  - How to create a new domain
  - Interface requirements
  - Best practices
  - Common patterns

- [ ] **Update API documentation**
  - Document domain-related functions
  - Add examples for each domain method
  - Document domain configuration options

### 7. **Backward Compatibility** (LOW PRIORITY)

**Goal**: Ensure existing code still works

- [ ] **Default domain behavior**
  - Ensure `default_processing_options()` defaults to "biomedical"
  - Auto-register biomedical domain if available (optional)
  - Provide clear error messages when domain not registered

- [ ] **Deprecation warnings**
  - Add deprecation warnings for old biomedical-specific functions
  - Guide users to domain system
  - Update migration guide

### 8. **Performance & Optimization** (LOW PRIORITY)

**Goal**: Ensure domain system doesn't add overhead

- [ ] **Profile domain switching**
  - Measure overhead of domain provider calls
  - Optimize if needed
  - Cache domain lookups if beneficial

- [ ] **Domain caching**
  - Consider caching domain providers
  - Cache domain-specific patterns/rules
  - Optimize entity/relation extraction

## 🎯 Immediate Next Actions (This Week)

1. **Test the domain system** with real examples
   - Create simple test script
   - Verify biomedical extraction works
   - Verify Wikipedia extraction works

2. **Fix biomedical domain dependencies**
   - Make entities.jl and relations.jl self-contained
   - Remove circular dependencies
   - Test domain loading

3. **Create integration tests**
   - Test full extraction pipeline with domains
   - Test domain switching
   - Verify backward compatibility

## 📝 Notes

- The domain system is **functional** but needs testing
- Biomedical domain currently has dependencies on old files (acceptable for now, but should be fixed)
- Wikipedia domain is **fully self-contained**
- Both domains can be registered and used successfully
- Core system is **completely domain-agnostic**

## 🚀 Quick Start for Testing

```julia
using GraphMERT

# Load domains
include("GraphMERT/src/domains/biomedical.jl")
include("GraphMERT/src/domains/wikipedia.jl")

# Register domains
bio = load_biomedical_domain()
wiki = load_wikipedia_domain()
register_domain!("biomedical", bio)
register_domain!("wikipedia", wiki)

# Test biomedical extraction
text_bio = "Diabetes is treated with metformin."
options_bio = ProcessingOptions(domain="biomedical")
model = create_graphmert_model(GraphMERTConfig())
graph_bio = extract_knowledge_graph(text_bio, model; options=options_bio)
println("Biomedical entities: ", length(graph_bio.entities))

# Test Wikipedia extraction
text_wiki = "Leonardo da Vinci was born in Vinci, Italy."
options_wiki = ProcessingOptions(domain="wikipedia")
graph_wiki = extract_knowledge_graph(text_wiki, model; options=options_wiki)
println("Wikipedia entities: ", length(graph_wiki.entities))
```
