# GraphMERT.jl — Retrospective Specification

> What the spec should have said before the first line of code was written.
> Grounded in: arXiv:2510.09580, `original_paper/GraphMert/src/`, and code review findings.
> See `RETROSPECTIVE_SPEC-FULL.md` for complete algorithmic pseudocode.

---

## 1. System invariants

1. Encoder-only transformer; no decoder.
2. Special tokens: `[MASK] [CLS] [SEP] [PAD] [REL] [HEAD] [TAIL]` — all in the tokenizer.
3. Sequence length = `num_roots + num_roots × num_leaves_per_root`; default 128 + 128×7 = **1024**.
4. `max_position_embeddings` **must be ≥ max_sequence_length** (1024). Setting it to 512 is invalid.
5. `rand()` is never used in any production path (extraction, evaluation, persistence).
6. `MockGraphMERTModel` must never be returned from `train_graphmert`.
7. Every `Relation.head` and `Relation.tail` resolves to an existing `Entity.id`.
8. All benchmarks in README must come from a real trained model, not mock losses.

---

## 2. Type contracts

### GraphMERTConfig
```julia
vocab_size              :: Int = 50265   # RoBERTa
hidden_size             :: Int = 768
num_attention_heads     :: Int = 12
num_hidden_layers       :: Int = 12
intermediate_size       :: Int = 3072
max_position_embeddings :: Int = 1024    # ← MUST equal max_sequence_length
max_sequence_length     :: Int = 1024
num_roots               :: Int = 128
num_leaves_per_root     :: Int = 7
attention_decay_alpha   :: Float64 = 0.1
num_relations           :: Int = 50
hgat_num_heads          :: Int = 8
```

### GraphMERTModel (real, not mock)
```
embeddings :: GraphMERTEmbeddings   # word + position + token_type
                                    # with H-GAT rel-embedding injection at [REL] positions
hgat       :: HGATEncoder           # relation embedding subnetwork
layers     :: Vector{GraphMERTLayer} # each layer MUST receive distance_matrix
mlm_head   :: MLMHead               # Dense→GELU→LN→Dense(vocab_size)
mnm_head   :: MNMHead               # Dense→GELU→LN→Dense(1), positions-based
config     :: GraphMERTConfig
```

### LeafyChainGraph key invariants
- `length(nodes) == config.max_sequence_length` always
- `adjacency_matrix[i,i] == 0` (no self-loops)
- `shortest_paths` computed by Floyd-Warshall, not Dijkstra; unreachable = `max_sequence_length`
- At most one triple injected per root; `leaf_tokens[r, 0]=[HEAD]`, `[1]=[REL]`, `[2..N]=[TAIL]`

### KnowledgeGraph
- `entities :: Vector{Entity}`, `relations :: Vector{Relation}`
- Backward-compat virtual properties `source_text` and `triples` must remain

---

## 3. Algorithmic contracts

### 3.1 Attention decay mask (every transformer layer)
```
decay(i,j) = exp(-α × d(i,j))
applied:  scores += log(decay + ε)      # additive, log-space
d(i,j) from Floyd-Warshall shortest paths (not Dijkstra)
```

### 3.2 MNM masking and loss
- Masks **entire leaf group** (all leaves) when a root is selected (p=0.15)
- Loss: `binary_cross_entropy_with_logits`, NOT `crossentropy`
- Labels: binary vector over candidate UMLS concepts
- Gradients must flow through both transformer and H-GAT

### 3.3 Joint training loss
```
L(θ) = L_MLM + μ × L_MNM     where μ = 1.0
```
Must call `Flux.update!` every step with real gradients.

### 3.4 Triple extraction — Stage 3 (tail prediction)
- GraphMERT forward pass; extract logits at leaf positions
- **Never use `rand()` here**

### 3.5 FActScore* computation
- Iterate `kg.relations` directly
- For each relation, LLM verifies the claim against the source passage
- `FActScore* = supported / total` — no cartesian product

### 3.6 Seed KG injection (4 stages)
1. Entity linking: SapBERT embedding (ANN) + character 3-gram Jaccard, top-10 UMLS candidates
2. Triple retrieval: retrieve UMLS triples for each CUI, filter by allowed relation types
3. Contextual selection: Gemini embeddings, cosine similarity, top-40 per entity
4. Injection: bucket by relation type, fill proportionally for diversity

---

## 4. API contracts

### extract_knowledge_graph
```julia
extract_knowledge_graph(text::String, model::GraphMERTModel;
                        options::ProcessingOptions) -> KnowledgeGraph
```
- Empty string → empty KG (not an exception)
- Domain logic routes through `DomainProvider`

### train_graphmert
```julia
train_graphmert(texts, config; seed_kg, callbacks) -> GraphMERTModel
```
- Real `GraphMERTModel` returned (never mock)
- Real gradients with `Flux.update!` every step
- Checkpoint saves actual weights via JLD2

### save_model / load_model
```julia
save_model(model::GraphMERTModel, path::String) -> Nothing
load_model(path::String) -> GraphMERTModel
```
- Must round-trip: `load_model(save_model(m))(x) ≈ m(x)` for same input

### DomainProvider (all four methods required)
```julia
extract_entities(domain, text, config) -> Vector{Entity}
extract_relations(domain, entities, text, config) -> Vector{Relation}
get_ontology_types(domain) -> Vector{String}
validate_relation(domain, head_type, rel_type, tail_type) -> Bool
```

### validate_kg
```julia
validate_kg(kg::KnowledgeGraph) -> ValidityReport
```
- Checks referential integrity + ontological validity
- Does not modify the KG

---

## 5. External dependency contracts

| Dependency | Mode | CI |
|-----------|------|----|
| SapBERT | Required for training | Mock |
| UMLS REST | Required for training | Mock + local cache |
| LLM (Gemini/GPT) | Required for extraction + evaluation | Mock |
| JLD2 | Required for persistence | — |
| Flux.jl ≥ 0.14 | Required | — |

All external integrations must support offline/mock mode for CI.

---

## 6. Extension readiness contracts

| Extension | Required hook |
|----------|--------------|
| DRAG distillation | `distillation_loss(student_kg, teacher_kg, evidence)` in training |
| LinguGKD | `align_features(llm_features, hgat_features, config)` |
| Multi-domain injection | `OntologySource` abstraction over UMLS |
| KG completion | Separate extraction mode that extends an existing KG |

---

*See `RETROSPECTIVE_SPEC-FULL.md` for complete type definitions and pseudocode.*
*See `PARITY_PLAN.md` for the implementation roadmap.*
*See `TASK_LIST.md` for the actionable work items.*
