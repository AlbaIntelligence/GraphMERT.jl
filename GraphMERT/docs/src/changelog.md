# Changelog

All notable changes to GraphMERT.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Relation extraction improvements** (see `reports/RELATION_EXTRACTION_IMPROVEMENT_PROPOSAL.md`): (1) Wikipedia relation extraction now uses entity-agnostic regex patterns (SPOUSE_OF, PARENT_OF, BORN_IN, DIED_IN, SUCCESSOR_OF, etc.) and resolves captures against the entity list. (2) When domain is missing or relation extraction fails, fallback uses sentence-level co-occurrence and deduplication instead of document-wide pairs. (3) LLM is wired into relation matching: `match_relations_for_entities(..., llm_client=...)` and domain `extract_relations(..., llm_client=nothing)`; Wikipedia uses `create_prompt(domain, :relation_matching, context)` and `LocalLLM.generate` when `use_local` is set. (4) Optional Wikidata enrichment: when `domain.wikidata_client !== nothing`, relations from Wikidata are merged for entities with `attributes["wikidata_qid"]`. `LocalLLM.generate(client, prompt)` added for custom prompts.
- **Default encoder path:** `load_model()` with no arguments loads the RoBERTa encoder from `~/.cache/llama-cpp/models/encoders/roberta-base` (directory with `checkpoint.json` or Hugging Face `config.json`). Override root with `GRAPHMERT_ENCODER_ROOT`. `default_encoder_path()` returns the path; persistence accepts a directory and looks for `checkpoint.json` or builds a default model when only `config.json` exists. `extract_knowledge_graph(text; options)` uses the default model when available.
- **Reliability pipeline** (spec 003-align-contextual-description): Provenance tracking and `get_provenance(kg, relation_or_index)` for traceable triples; `validate_kg(kg, domain)` returning ValidityReport with graceful degradation when ontology is missing; `evaluate_factscore(kg, reference)` returning FactualityScore; `clean_kg(kg; policy)` with CleaningPolicy (min_confidence, require_provenance, contradiction_handling). New types: ProvenanceRecord, ValidityReport, FactualityScore, CleaningPolicy. Extraction populates provenance when `enable_provenance_tracking=true`; empty corpus returns empty KG without phantom provenance. Encoder-in-path: `load_model` returns full GraphMERTModel; extraction uses encoder when model is GraphMERTModel. Documented path for augmented seed (cleaned KG as seed). See `reports/REFERENCE_SOURCES_AND_ENCODER.md`, `reports/PROJECT_STATUS.md`, and `specs/003-align-contextual-description/`.
- Unit tests: `test_provenance.jl`, `test_cleaning.jl`. Integration tests: `test_extraction_provenance.jl`, `test_validity.jl`, `test_factscore_cleaning.jl`, `test_encoder_in_path.jl`.
- Comprehensive documentation using Documenter.jl
- API reference with complete function documentation
- User guides and tutorials
- Scientific background documentation
- Performance optimization guides

## [0.1.0] - 2024-12-19

### Added
- Initial release of GraphMERT.jl
- Core knowledge graph extraction functionality
- Biomedical entity recognition and classification
- Relation extraction and matching
- UMLS integration for entity linking
- Helper LLM integration for entity discovery
- Leafy chain graph construction
- RoBERTa-based encoder with H-GAT components
- Training pipeline with MLM and MNM objectives
- Seed KG injection for prior knowledge integration
- Comprehensive evaluation metrics (FActScore, ValidityScore, GraphRAG)
- Batch processing for large document corpora
- Memory monitoring and optimization
- Performance benchmarking and profiling
- Multiple export formats (JSON, CSV, RDF, Turtle)
- Utility functions for graph manipulation
- Comprehensive test suite with 50+ test files
- Working examples and demonstrations
- Scientific validation against original paper benchmarks

### Features
- **Entity Recognition**: Extract biomedical entities with confidence scoring
- **Relation Extraction**: Identify semantic relationships between entities
- **UMLS Integration**: Link entities to Unified Medical Language System
- **LLM Assistance**: Use external LLMs for entity discovery and relation matching
- **Batch Processing**: Efficient processing of multiple documents
- **Performance Monitoring**: Real-time metrics and optimization
- **Export Formats**: Multiple output formats for different use cases
- **Evaluation Metrics**: Comprehensive quality assessment tools

### Performance
- Process 5,000+ tokens per second on laptop hardware
- Memory usage <4GB for datasets up to 124.7M tokens
- FActScore within 5% of original paper results (69.8% target)
- ValidityScore within 5% of original paper results (68.8% target)

### Architecture
- **RoBERTa Encoder**: Base transformer for text understanding
- **H-GAT**: Hierarchical Graph Attention for semantic relation encoding
- **Leafy Chain Graph**: Structure for representing text with semantic nodes
- **UMLS Integration**: Biomedical knowledge base for entity linking
- **Helper LLM**: External LLM for entity discovery and relation matching

### Dependencies
- Julia 1.10+
- Flux.jl for machine learning
- Transformers.jl for RoBERTa support
- Graphs.jl and MetaGraphs.jl for graph processing
- CSV.jl and JSON3.jl for data handling
- HTTP.jl for web API integration
- BenchmarkTools.jl for performance testing

### Testing
- 50+ test files covering all functionality
- Unit tests for individual components
- Integration tests for end-to-end workflows
- Performance tests for benchmarking
- Scientific validation tests against paper benchmarks

### Documentation
- Comprehensive API documentation
- User guides and tutorials
- Scientific background and algorithm details
- Performance optimization guides
- Troubleshooting and FAQ sections

## [0.0.1] - 2024-12-19

### Added
- Project initialization
- Basic package structure
- Core dependencies setup
- Initial documentation framework
