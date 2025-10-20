# GraphMERT Algorithm Replication

## Overview

**Objective:** Replicate the GraphMERT algorithm in Julia to construct knowledge graphs using the exact RoBERTa-based architecture with H-GAT (Hierarchical Graph Attention), maintaining fidelity to the original paper while leveraging Julia's scientific computing capabilities and achieving extremely elegant code.

**Scope:** Implement the complete GraphMERT pipeline including RoBERTa-based architecture with H-GAT, biomedical domain processing with UMLS integration, seed KG injection, MLM+MNM training, and knowledge graph construction with performance validation using the diabetes dataset from the original paper.

**Timeline:** 3 months for complete implementation and validation

## User Scenarios & Testing

### Primary User Flow

1. **Input:** User provides biomedical text corpus or documents
2. **Processing:** System applies GraphMERT algorithm with RoBERTa+H-GAT architecture, UMLS integration, and helper LLM for entity discovery
3. **Output:** User receives structured biomedical knowledge graph with entities, relations, and confidence scores
4. **Validation:** User can compare results with original paper benchmarks on the diabetes dataset

### Secondary User Flow

1. **Benchmarking:** Researcher runs performance comparisons against original implementation
2. **Customization:** Developer adapts algorithm for specific domain or language
3. **Integration:** Data scientist incorporates into larger NLP pipeline

## Functional Requirements

### Core Algorithm Implementation

- **REQ-001:** Implement RoBERTa-based encoder architecture with H-GAT (Hierarchical Graph Attention)
- **REQ-002:** Implement leafy chain graph structure for text representation
- **REQ-003:** Integrate UMLS biomedical knowledge base for entity linking and validation
  - **REQ-003a:** Connect to UMLS REST API (<https://uts-ws.nlm.nih.gov/rest>) with authentication
  - **REQ-003b:** Support rate limiting (100 requests/minute) and retry logic with exponential backoff
  - **REQ-003c:** Map biomedical entities to UMLS CUI (Concept Unique Identifier) codes
  - **REQ-003d:** Cache UMLS mappings locally to reduce API calls and improve performance
  - **REQ-003e:** Handle UMLS API errors gracefully with fallback to local entity recognition
- **REQ-004:** Implement seed KG injection algorithm for training data preparation
  - **REQ-004a:** Extract relevant triples from external KG (UMLS) based on text content similarity
  - **REQ-004b:** Filter triples by confidence threshold (>0.7) and biomedical domain relevance
  - **REQ-004c:** Inject triples as leaf nodes in leafy chain graph structure
  - **REQ-004d:** Create semantic connections between text tokens and injected KG triples
  - **REQ-004e:** Validate injected triples against original text for semantic consistency
  - **REQ-004f:** Support configurable injection ratio (10-30% of training examples)
- **REQ-005:** Implement MLM + MNM (Masked Language Modeling + Masked Node Modeling) training objectives
- **REQ-006:** Implement helper LLM integration for entity discovery and relation matching
  - **REQ-006a:** Support OpenAI GPT-4 API integration with configurable model parameters
  - **REQ-006b:** Implement structured prompts for entity discovery and relation extraction
  - **REQ-006c:** Handle API rate limits (10,000 tokens/minute) with request queuing
  - **REQ-006d:** Parse JSON responses from helper LLM and validate entity/relation formats
  - **REQ-006e:** Implement fallback to local entity recognition when LLM API is unavailable
  - **REQ-006f:** Cache LLM responses to reduce API costs and improve performance
- **REQ-007:** Generate biomedical knowledge graph structure with nodes (entities) and edges (relations)
- **REQ-008:** Provide confidence scores for extracted entities and relations
- **REQ-009:** Support biomedical text formats (PubMed abstracts, medical documents)
- **REQ-010:** Optimize for laptop deployment with 80M parameter model

### Performance & Validation

- **REQ-011:** Achieve FActScore within 5% of original paper results (69.8% target) on diabetes dataset using exact same evaluation methodology as original paper with 95% confidence intervals
- **REQ-012:** Achieve ValidityScore within 5% of original paper results (68.8% target) on diabetes dataset using exact same evaluation methodology as original paper with 95% confidence intervals
- **REQ-013:** Support biomedical datasets up to 124.7M tokens for training
- **REQ-014:** Provide memory usage optimization for 80M parameter model (target: <4GB RAM)
- **REQ-015:** Generate reproducible results with documented random seeds
- **REQ-016:** Implement GraphRAG evaluation methodology for KG quality assessment

### API Design

- **REQ-017:** Expose simple API: `extract_knowledge_graph(text, model, options)`
- **REQ-018:** Support batch processing for multiple biomedical documents
- **REQ-019:** Provide configuration options for RoBERTa+H-GAT model parameters
- **REQ-020:** Enable biomedical entity and relation types from UMLS
- **REQ-021:** Support helper LLM integration for entity discovery
- **REQ-022:** Implement seed KG injection and training pipeline APIs

### Code Elegance

- **REQ-023:** Implement extremely elegant, idiomatic Julia code that serves as a reference implementation
  - **REQ-023a:** Maintain cyclomatic complexity ≤ 10 for all functions
  - **REQ-023b:** Keep function length ≤ 50 lines with single responsibility
  - **REQ-023c:** Achieve 100% type coverage for all public APIs
  - **REQ-023d:** Use descriptive function names following Julia naming conventions
  - **REQ-023e:** Implement comprehensive docstrings with examples for all exported functions
  - **REQ-023f:** Maintain ≥80% test coverage for all public APIs (constitution compliance)
  - **REQ-023g:** Ensure all code passes Julia's built-in linting and formatting standards
- **REQ-024:** Use Julia's multiple dispatch system effectively for clean, extensible design
- **REQ-025:** Leverage Julia's type system for type-safe, performant code
- **REQ-026:** Write self-documenting code with clear, expressive function and variable names
- **REQ-027:** Follow functional programming principles where appropriate
- **REQ-028:** Use Julia's broadcasting and vectorization capabilities for elegant numerical operations

## Non-Functional Requirements

### Performance

- **NFR-001:** Process 5,000 tokens per second on laptop hardware (Intel i7-10750H or AMD Ryzen 7 4800H, 16GB RAM, no GPU acceleration)
- **NFR-002:** Memory usage should not exceed 4GB for datasets up to 100K tokens (tested on 16GB RAM systems)
- **NFR-003:** Algorithm should scale linearly with input size
- **NFR-004:** ONNX model loading should complete within 30 seconds

### Reliability

- **NFR-005:** Handle malformed input gracefully with appropriate error messages
- **NFR-006:** Provide deterministic results for identical inputs
- **NFR-007:** Maintain 99.9% uptime for processing operations
- **NFR-008:** ONNX model inference should be stable across different Julia versions

### Usability

- **NFR-009:** API should be intuitive for Julia users familiar with scientific computing
- **NFR-010:** Documentation should include complete examples and tutorials
- **NFR-011:** Error messages should be informative and actionable
- **NFR-012:** ONNX model integration should be transparent to end users

### Code Quality

- **NFR-013:** Code must be extremely elegant and serve as a reference implementation for Julia scientific computing
- **NFR-014:** All functions must have clear, single responsibilities following the single responsibility principle
- **NFR-015:** Use Julia's type system to create expressive, type-safe interfaces
- **NFR-016:** Implement clean abstractions that hide complexity while remaining extensible
- **NFR-017:** Code must be readable by both domain experts and Julia newcomers
- **NFR-018:** Follow Julia community best practices and style guidelines

## Success Criteria

### Quantitative Metrics

- **SC-001:** Achieve F1 score within 5% of original paper results on the most commonly cited dataset using identical evaluation methodology with statistical significance testing (p < 0.05)
- **SC-002:** Processing speed improvement of at least 20% over Python reference implementation (benchmarked on Intel i7-10750H, 16GB RAM)
- **SC-003:** Memory efficiency improvement of at least 30% over reference implementation
- **SC-004:** Test coverage of at least 90% for all public APIs
- **SC-005:** Laptop deployment successful on systems with 8GB RAM or less
- **SC-006:** ONNX model loading and inference works across Julia 1.8+ versions

### Qualitative Measures

- **SC-007:** Code quality meets Julia community standards (passes all linter checks)
- **SC-008:** Documentation completeness enables new users to run examples within 30 minutes
- **SC-009:** Integration with existing Julia ecosystem (compatible with major ML packages)
- **SC-010:** Scientific reproducibility verified by independent researchers
- **SC-011:** Code elegance recognized by Julia community as exemplary implementation
- **SC-012:** Code serves as educational reference for Julia scientific computing best practices

## Key Entities

### Data Structures

- **KnowledgeGraph:** Main output structure containing biomedical entities, relations, and metadata
- **Entity:** Represents extracted biomedical entities with confidence scores and UMLS CUI mapping
- **Relation:** Represents biomedical relationships between entities with confidence scores
- **ProcessingOptions:** Configuration parameters for GraphMERT algorithm execution
- **GraphMERTConfig:** Configuration for RoBERTa+H-GAT model parameters
- **LeafyChainGraph:** Graph structure with root nodes (text tokens) and leaf nodes (semantic triples)
- **H-GAT:** Hierarchical Graph Attention component for semantic relation encoding
- **SeedKG:** Seed knowledge graph for training data preparation
- **UMLSIntegration:** Biomedical knowledge base integration for entity linking

### Input/Output

- **BiomedicalTextCorpus:** Input biomedical text data (PubMed abstracts, medical documents)
- **GraphMERTModel:** RoBERTa+H-GAT model for biomedical knowledge extraction
- **BenchmarkResults:** Performance metrics and comparison data for diabetes dataset
- **UMLSData:** Biomedical knowledge base for entity linking and validation
- **SeedKGData:** Seed knowledge graph for training data preparation

## Assumptions

- **A-001:** Users have basic familiarity with Julia and scientific computing concepts
- **A-002:** Input text is biomedical English (PubMed abstracts, medical documents)
- **A-003:** Laptop hardware configuration (8GB RAM, modern CPU) for 80M parameter model
- **A-004:** Julia ML ecosystem supports RoBERTa architecture implementation
- **A-005:** Original paper provides sufficient algorithmic detail for implementation
- **A-006:** 80M parameter GraphMERT model provides acceptable accuracy for biomedical knowledge graph construction
- **A-007:** Diabetes dataset from original paper is sufficient for initial validation
- **A-008:** UMLS biomedical knowledge base is accessible for entity linking
- **A-009:** Helper LLM integration is feasible in Julia ecosystem
- **A-010:** Elegant code design will enhance maintainability and community adoption

## Dependencies

### External Libraries

- **ML Framework:** Flux.jl, Transformers.jl for RoBERTa implementation
- **Text Processing:** TextAnalysis.jl, Languages.jl for biomedical text processing
- **Graph Processing:** LightGraphs.jl, MetaGraphs.jl for leafy chain graphs
- **Data Manipulation:** DataFrames.jl, CSV.jl for data handling
- **Biomedical Integration:** UMLS access libraries, helper LLM integration
- **HTTP Client:** HTTP.jl for UMLS REST API integration
- **JSON Processing:** JSON.jl for API response parsing

### External Data

- Diabetes dataset from original paper (350k abstracts, 124.7M tokens)
- UMLS biomedical knowledge base (SNOMED CT, Gene Ontology)
- Seed KG data for training preparation
- Pre-trained RoBERTa models for GraphMERT architecture

## Constraints

- **C-001:** Must use existing Julia libraries to maximum extent possible
- **C-002:** Implementation must be scientifically accurate to original GraphMERT algorithm
- **C-003:** Code must follow Julia package development best practices
- **C-004:** Performance must be competitive with or exceed original implementation
- **C-005:** Documentation must be comprehensive and accessible to researchers
- **C-006:** Must be deployable on laptop hardware with 80M parameter model
- **C-007:** Must complete within 3-month timeline
- **C-008:** Biomedical domain focus must be maintained for scientific accuracy
- **C-009:** Code must achieve extremely elegant design that serves as a reference implementation
- **C-010:** UMLS integration must be stable and well-documented

## Implementation Notes

### Code Elegance Principles

- **Multiple Dispatch:** Leverage Julia's multiple dispatch for clean, extensible interfaces
- **Type System:** Use abstract types and concrete types effectively for type safety
- **Functional Style:** Prefer pure functions and immutable data structures where possible
- **Broadcasting:** Use Julia's broadcasting syntax for elegant vectorized operations
- **Composability:** Design functions that compose well together
- **Error Handling:** Use Julia's exception system elegantly with informative error messages
- **Documentation:** Write self-documenting code with clear, expressive names

### Design Patterns

- **Strategy Pattern:** For different biomedical text preprocessing approaches
- **Builder Pattern:** For constructing leafy chain graphs and knowledge graphs
- **Observer Pattern:** For progress monitoring during MLM+MNM training
- **Factory Pattern:** For creating different GraphMERT model configurations
- **Adapter Pattern:** For UMLS integration and helper LLM communication
- **Template Method Pattern:** For GraphMERT training pipeline steps
