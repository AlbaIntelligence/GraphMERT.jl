# GraphMERT Comprehensive Technical Specification

## Master Index and Navigation

**Document Version**: 1.0
**Last Updated**: 2025-01-20
**Based On**: GraphMERT Paper (arXiv:2510.09580)

---

## Purpose of This Specification

This comprehensive specification expands the original GraphMERT paper with:

- **Complete algorithmic details** with pseudocode and step-by-step breakdowns
- **Precise data structure definitions** in Julia
- **Mathematical formulations** with worked examples
- **Implementation guidance** for all components
- **Code mappings** to existing implementation (with line numbers)
- **Gap analysis** identifying what needs to be implemented

**Target Audience**: This specification is designed to be detailed enough that AI assistants and human developers can implement a complete, faithful GraphMERT replication from it.

---

## Document Organization

This specification is organized **logically and intellectually** to match how GraphMERT works conceptually, from foundational concepts through to evaluation. For **implementation priorities and practical steps**, see the companion document `00-IMPLEMENTATION-ROADMAP.md`.

---

## Part I: Architecture Foundation

### [01. Architecture Overview](01-architecture-overview.md)

**Lines**: ~400 | **Status**: ✅ Complete

- System architecture and data flow
- Syntactic vs. semantic spaces
- Key architectural decisions
- Component interactions
- Training vs. extraction phases
- High-level workflow diagrams

**Key Concepts**:

- Neurosymbolic AI integration
- Encoder-only transformer approach
- Graph encoding for text
- Vocabulary transfer (syntactic → semantic)

---

### [02. Leafy Chain Graph Structure](02-leafy-chain-graphs.md)

**Lines**: ~450 | **Status**: 🔴 Critical - Mostly Missing

- Graph topology and structure
- Root nodes (syntactic space)
- Leaf nodes (semantic space)
- Edge connectivity rules
- Sequential encoding scheme
- Graph construction algorithms
- Token-level vs. term-level representation
- Padding and fixed-size constraints

**Existing Code**: `GraphMERT/src/graphs/leafy_chain.jl` (30 lines, placeholder)
**Paper Reference**: Section 4.1, Figure 2

**What This Enables**:

- Unified representation for training
- Joint syntactic-semantic learning
- Regular graph structure for efficient encoding

---

### [03. RoBERTa Encoder Architecture](03-roberta-encoder.md)

**Lines**: ~350 | **Status**: ✅ Well Implemented

- RoBERTa base architecture
- Embedding layers (word, position, token type)
- Self-attention mechanisms
- Feed-forward networks
- Layer normalization and residual connections
- Configuration parameters (80M model)
- Tokenization (BioMedBERT)

**Existing Code**: `GraphMERT/src/architectures/roberta.jl` (444 lines, complete)
**Paper Reference**: Section 4.2, uses standard RoBERTa architecture

**Implementation Notes**:

- 12 hidden layers
- 8 attention heads
- Hidden size: 512
- Vocabulary: 30,522 (BioMedBERT tokenizer)

---

### [04. H-GAT Component](04-hgat-component.md)

**Lines**: ~400 | **Status**: ✅ Well Implemented

- Hierarchical Graph Attention Network
- Relation embedding learning
- Token-level attention fusion
- Multi-head attention for graph nodes
- Integration with transformer layers
- Relation-aware node embeddings

**Mathematical Formulation**:

```
e_ij^(r) = LeakyReLU(a_r^T [W_r t_i || W_r h_j])
α_ij^(r) = softmax_j(e_ij^(r))
t_i' = Σ_j α_ij^(r) W_r h_j
```

**Existing Code**: `GraphMERT/src/architectures/hgat.jl` (437 lines, complete)
**Paper Reference**: Section 2.5.2, Equation 1-3

---

### [05. Attention and Graph Encodings](05-attention-mechanisms.md)

**Lines**: ~350 | **Status**: 🟡 Partially Implemented

- Attention decay mask (spatial distance encoding)
- Floyd-Warshall shortest paths
- Exponential decay function
- GELU activation for masking
- Multi-head attention modifications
- Shared mask across layers

**Mathematical Formulation**:

```
sp(i,j) = shortest_path(i, j)
f(sp(i,j)) = λ^GELU(√(sp(i,j)) - p)
Ã = A ⊙ f
```

**Existing Code**: Partially in `hgat.jl`, needs extraction and documentation
**Paper Reference**: Section 4.2.1, Figure 3

---

## Part II: Training Methodology

### [06. Masked Language Modeling (MLM)](06-training-mlm.md)

**Lines**: ~450 | **Status**: ✅ Well Implemented

- Span masking strategy
- Boundary loss (SpanBERT)
- Masking probability (15%)
- Token replacement strategies (80/10/10)
- Loss calculation
- Gradient flow

**Mathematical Formulation**:

```
L_MLM(θ) = -Σ_{t∈M_x} [log p_θ(x_t | G_{∖M_g∪M_x}) + L_SBO(x_t | ...)]
```

**Existing Code**: `GraphMERT/src/training/mlm.jl` (436 lines, complete)
**Paper Reference**: Section 4.2.2, Equation 4

**Implementation Details**:

- Span length: up to 7 tokens
- Geometric distribution for span sampling
- Context-aware prediction

---

### [07. Masked Node Modeling (MNM)](07-training-mnm.md)

**Lines**: ~400 | **Status**: 🔴 Critical - Needs Implementation

- Semantic space masking
- Leaf node prediction
- Relation embedding training
- Gradient backpropagation through H-GAT
- Joint MLM+MNM training
- Loss balancing (μ = 1.0)

**Mathematical Formulation**:

```
L_MNM(θ) = -Σ_{ℓ∈M_g} log p_θ(g_ℓ | G_{∖M_g∪M_x})
L(θ) = L_MLM(θ) + μ·L_MNM(θ)
```

**Existing Code**: `GraphMERT/src/training/mnm.jl` (30 lines, stub only)
**Paper Reference**: Section 4.2.2, Equation 5-6

**Key Differences from MLM**:

- Masks entire leaf spans (not partial)
- Updates relation embeddings
- Predicts semantic tokens

---

### [08. Seed KG Injection](08-seed-kg-injection.md)

**Lines**: ~500 | **Status**: 🔴 Critical - Needs Full Implementation

- Entity linking (UMLS/SapBERT)
- Embedding-based candidate retrieval
- String similarity filtering (Jaccard)
- Contextual triple selection
- Injection algorithm (diversity + relevance)
- Score bucketing strategy

**Algorithm Components**:

1. **Stage 1**: Embedding similarity (top-k candidates)
2. **Stage 2**: String matching (char-3grams)
3. **Stage 3**: Contextual ranking
4. **Stage 4**: Injection algorithm (Algorithm 1 from paper)

**Existing Code**: `GraphMERT/src/training/seed_injection.jl` (19 lines, placeholder)
**Paper Reference**: Section 4.3, Appendix B, Algorithm 1

---

## Part III: Knowledge Extraction

### [09. Triple Extraction Pipeline](09-triple-extraction.md)

**Lines**: ~450 | **Status**: 🔴 Needs Full Implementation

- Entity discovery (helper LLM)
- Relation matching
- Top-k token prediction
- Token combination (helper LLM)
- Similarity filtering (β threshold)
- Deduplication
- Provenance tracking

**Pipeline Stages**:

1. **Head Discovery**: Extract entities from text
2. **Relation Assignment**: Match entities to relations
3. **Tail Prediction**: GraphMERT predicts tokens
4. **Tail Formation**: LLM combines tokens
5. **Filtering**: Similarity check + validation
6. **Output**: Final KG with provenance

**Existing Code**: Scattered across multiple files, needs consolidation
**Paper Reference**: Section 4.4, Figures 4, 7-8

---

## Part IV: Evaluation and Validation

### [10. Evaluation Metrics](10-evaluation-metrics.md)

**Lines**: ~400 | **Status**: 🟡 Partially Implemented

- **FActScore\***: Factuality evaluation with validity
- **ValidityScore**: Ontological alignment
- **GraphRAG**: Graph-level question answering
- Benchmark evaluation (ICD-Bench, MedMCQA, etc.)
- Triple-level vs. graph-level evaluation
- Human evaluation protocols

**Mathematical Formulation**:

```
f(τ) = 𝟙[τ is supported by C(τ)]
FActScore*(G) = E_{τ∈G}[f(τ)]
ValidityScore = count(yes) / total_triples
```

**Existing Code**: `GraphMERT/src/evaluation/` (multiple files)
**Paper Reference**: Section 5.2, 5.3

---

## Part V: Implementation Details

### [11. Data Structures and Types](11-data-structures.md)

**Lines**: ~450 | **Status**: 🟡 Needs Expansion

- Complete Julia type definitions
- Configuration structures
- Input/output data formats
- Graph representations
- Batch structures
- Intermediate representations
- Serialization formats

**Covers**:

- `LeafyChainGraph`
- `GraphMERTConfig`
- `MLMConfig` / `MNMConfig`
- `BiomedicalEntity` / `BiomedicalRelation`
- `KnowledgeGraph`
- `SeedKGConfig`
- Batch types for training

**Existing Code**: `GraphMERT/src/types.jl` (272 lines)
**Paper Reference**: Throughout paper, various sections

---

### [12. Implementation Mapping](12-implementation-mapping.md)

**Lines**: ~400 | **Status**: 🔴 Needs Creation

- Map specification sections to existing code
- Line number references
- Function-by-function analysis
- What's complete vs. incomplete
- Code quality assessment
- Refactoring recommendations

**Analysis Structure**:

- ✅ **Complete**: Working, tested, documented
- 🟡 **Partial**: Working but needs enhancement
- 🔴 **Missing**: Stub or not implemented
- ⚠️ **Needs Refactoring**: Works but needs improvement

---

### [13. Gap Analysis and Priorities](13-gaps-analysis.md)

**Lines**: ~350 | **Status**: 🔴 Needs Creation

- Critical missing components
- Implementation complexity estimates
- Dependency analysis
- Testing requirements
- Documentation gaps
- Performance optimization opportunities

**Priority Levels**:

- **P0 (Blocking)**: Required for basic functionality
- **P1 (High)**: Core features, major gaps
- **P2 (Medium)**: Important but not blocking
- **P3 (Low)**: Nice-to-have, optimizations

---

## Quick Reference

### Critical Missing Components (P0)

1. **Leafy Chain Graph Construction** (02) - Foundation for everything
2. **MNM Training Objective** (07) - Core training component
3. **Seed KG Injection Algorithm** (08) - Required for training data
4. **Triple Extraction Pipeline** (09) - Required for KG generation

### Well-Implemented Components

1. **RoBERTa Encoder** (03) - Complete architecture
2. **H-GAT Component** (04) - Relation embedding fusion
3. **MLM Training** (06) - Span masking implementation

### Needs Enhancement

1. **Attention Mechanisms** (05) - Extract from existing code
2. **Data Structures** (11) - Expand type definitions
3. **Evaluation Metrics** (10) - Complete implementations

---

## How to Use This Specification

### For Implementers

1. Start with **Architecture Overview** (01) to understand the system
2. Read **Leafy Chain Graphs** (02) for the foundational data structure
3. Follow the logical order through training (06-08) and extraction (09)
4. Consult **Implementation Mapping** (12) to see what exists
5. Use **Gap Analysis** (13) to prioritize work

### For Understanding

- Read documents 01-10 in order for complete conceptual understanding
- Each document is self-contained but references others
- Mermaid diagrams provide visual understanding
- Worked examples show data transformations

### For Implementation Planning

- See **`00-IMPLEMENTATION-ROADMAP.md`** for practical steps
- Prioritizes based on dependencies and current state
- Includes testing and validation strategies
- Estimates complexity and effort

---

## Document Conventions

### Mathematical Notation

- **Bold lowercase** (𝐱, 𝐡, 𝐭): vectors
- **Bold uppercase** (𝐖, 𝐀): matrices
- **Calligraphic** (𝒢, ℒ): sets, graphs, loss functions
- **ℝ**: real numbers
- **θ**: model parameters

### Code Conventions

- Julia type definitions use `PascalCase`
- Functions use `snake_case`
- Constants use `SCREAMING_SNAKE_CASE`
- Private functions prefixed with `_`

### Status Indicators

- ✅ **Complete**: Fully implemented and documented
- 🟡 **Partial**: Working but incomplete
- 🔴 **Missing**: Not implemented or stub only
- ⚠️ **Needs Work**: Implemented but needs refactoring

### Cross-References

- **[Section X.Y]**: Reference to paper section
- **→ Doc NN**: Reference to another spec document
- **`file.jl:123`**: Code reference with line number
- **Fig. N**: Reference to paper figure

---

## Maintenance and Updates

This specification should be updated when:

1. Implementation uncovers ambiguities or errors
2. Design decisions change
3. New optimizations are discovered
4. Code structure changes significantly
5. Paper errata or clarifications emerge

**Version History**:

- **1.0** (2025-01-20): Initial comprehensive specification

---

## Additional Resources

- **Original Paper**: `/original_paper/2510.09580 - GraphMERT [...].pdf`
- **Paper LaTeX**: `/original_paper/[...]/paper.tex`
- **Existing Spec**: `/.specify/features/replicate-graphmert/spec.md`
- **Existing Plan**: `/.specify/features/replicate-graphmert/plan.md`
- **Codebase**: `/GraphMERT/src/`

---

## Contact and Contributions

For questions or contributions to this specification:

- Review existing code before proposing changes
- Maintain mathematical rigor and precision
- Include worked examples for complex algorithms
- Keep documents under 500 lines
- Update this index when adding new documents

---

**Next Steps**: See `00-IMPLEMENTATION-ROADMAP.md` for practical implementation guidance.
