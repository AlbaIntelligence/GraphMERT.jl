# GraphMERT.jl — Master Task List

> Exhaustive, prioritized work items to reach paper + extension parity.
> Status key: 🔴 Not started · 🟡 Partial · ✅ Done
> See `TASK_LIST-FULL.md` for complete subtask breakdowns.

---

## STREAM A — Correctness (P0 defects)

### A1 — Fix training pipeline to use real model and gradients

**File**: `src/training/pipeline.jl`

- 🔴 A1.1 Replace `MockGraphMERTModel` return type with `GraphMERTModel` in `train_graphmert`
- 🔴 A1.2 Instantiate real `GraphMERTModel` (not `create_mock_graphmert_model`)
- 🔴 A1.3 Call real `forward_pass` inside the training loop
- 🔴 A1.4 Compute real MLM loss from model logits (not `rand()`)
- 🔴 A1.5 Compute real MNM loss from model logits (not `rand()`)
- 🔴 A1.6 Replace "simulate gradient update" comment with `Flux.update!(optimizer, ps, gs)`
- 🔴 A1.7 Remove `MockGraphMERTModel` struct and all its methods
- 🔴 A1.8 Update return type annotation and docstring
- 🔴 A1.9 Add test: `train_graphmert` returns `GraphMERTModel`, not mock

### A2 — Fix Stage 3 tail prediction

**File**: `src/api/extraction.jl`

- 🔴 A2.1 Remove `tail_probs = rand(Float32, vocab_size)` (line 170)
- 🔴 A2.2 Build proper leafy chain graph for the (head, relation) context
- 🔴 A2.3 Run real GraphMERT forward pass on the chain graph
- 🔴 A2.4 Extract logits at the `[TAIL]` leaf positions (not full sequence)
- 🔴 A2.5 Apply softmax to get probabilities; return top-k (token_id, prob) pairs
- 🔴 A2.6 Add unit test with a fixed model to verify top-k are consistent

### A3 — Fix Stage 4 tail formation

**File**: `src/api/extraction.jl`

- 🔴 A3.1 Remove `"entity_$(token_id)"` string generation (lines 200–205)
- 🔴 A3.2 Accept `llm_client` and pass top-k token IDs to LLM with structured prompt
- 🔴 A3.3 Parse LLM response into entity string list
- 🔴 A3.4 Validate returned entities appear in source text (or relax with domain check)
- 🔴 A3.5 Fall back to vocab lookup if LLM unavailable
- 🔴 A3.6 Add unit test with mock LLM

### A4 — Fix model persistence

**File**: `src/models/persistence.jl`

- 🔴 A4.1 Add JLD2 to `Project.toml` dependencies
- 🔴 A4.2 Implement `save_weights(model, path)` using `JLD2.@save`
- 🔴 A4.3 Implement `load_weights(path)` returning a `GraphMERTModel`
- 🔴 A4.4 Implement `save_model(model, path)` saving both weights and config
- 🔴 A4.5 Implement `load_model(path)` restoring weights and config from disk
- 🔴 A4.6 Remove `@warn "not implemented"` stubs
- 🔴 A4.7 Add round-trip test: `load_model(save_model(m, path))(x) ≈ m(x)` for same input
- 🔴 A4.8 Add checkpoint save/restore test covering optimizer state

### A5 — Fix FActScore cartesian product bug

**File**: `src/evaluation/factscore.jl`

- 🔴 A5.1 Replace three-nested-loop cartesian product in `filter_triples_by_confidence` (lines 197–212)
- 🔴 A5.2 New implementation: iterate `kg.relations`; look up `.head` / `.tail` entity IDs
- 🔴 A5.3 Check `relation.confidence >= threshold`; push matching relation indices only
- 🔴 A5.4 Add unit tests verifying correct triple count on a small hand-crafted KG

---

## STREAM B — Architecture alignment (Python reference)

### B1 — Fix position embedding table size

**File**: `src/architectures/roberta.jl`

- 🔴 B1.1 Change `max_position_embeddings` default from 512 to 1024 in `RoBERTaConfig`
- 🔴 B1.2 Change matching default in `GraphMERTConfig`
- 🔴 B1.3 Add assertion: `config.max_position_embeddings >= config.max_sequence_length`
- 🔴 B1.4 Add test: construct model with 1024-token sequence, no index-out-of-bounds

### B2 — Integrate attention decay mask into transformer layers

**Files**: `src/architectures/roberta.jl`, `src/architectures/attention_decay.jl`

- 🔴 B2.1 Verify `AttentionDecayMask` struct exists (or create it)
- 🔴 B2.2 Confirm `decay(i,j) = exp(-α × d(i,j))` and additive log-space application
- 🔴 B2.3 Add `distance_matrix::Matrix{Float32}` parameter to `RoBERTaLayer.forward`
- 🔴 B2.4 Pass `distance_matrix` from `LeafyChainGraph.shortest_paths` through all layers
- 🔴 B2.5 Confirm no layer skips the mask (all 12 layers)
- 🔴 B2.6 Add test: attention scores differ with and without decay mask on same input

### B3 — Wire H-GAT relation embeddings into embedding layer

**Files**: `src/architectures/roberta.jl`, `src/models/graphmert.jl`

- 🔴 B3.1 Add `inject_relation_embeddings!(embeddings, rel_embeds, rel_positions)` helper
- 🔴 B3.2 In `GraphMERTModel.forward`: run initial word+position+token_type embed, then run H-GAT, then replace `[REL]` positions with H-GAT output
- 🔴 B3.3 Verify this matches `graphmert.py` lines 94–99 (loop over batch × rel positions)
- 🔴 B3.4 Add test: embedding values at `[REL]` positions differ from un-injected baseline

### B4 — Implement real `GraphMERTModel.forward`

**File**: `src/models/graphmert.jl`

- 🔴 B4.1 Signature: `(model, input_ids, attention_mask, distance_matrix; relation_ids, rel_positions, kg_adjacency, mlm_labels, mnm_labels, mnm_positions)`
- 🔴 B4.2 Output dict: `last_hidden_state`, `mlm_logits`, `mlm_loss`, `mnm_logits`, `mnm_loss`
- 🔴 B4.3 Implement MLM head: Dense→GELU→LayerNorm→Dense(vocab_size)
- 🔴 B4.4 Implement MNM head: Dense→GELU→LayerNorm→Dense(1), select by `mnm_positions`
- 🔴 B4.5 MLM loss: `crossentropy` ignoring index −100
- 🔴 B4.6 MNM loss: `Flux.logitbinarycrossentropy`
- 🔴 B4.7 Add test: forward pass on batch of size 2, check output shapes

### B5 — Fix MNM loss function

**File**: `src/training/mnm.jl`

- 🔴 B5.1 Replace `Flux.crossentropy(leaf_logits, label + 1)` calls
- 🔴 B5.2 Use `Flux.logitbinarycrossentropy(logit, target)` instead
- 🔴 B5.3 Ensure labels are `Float32` binary vectors, not `Int`
- 🔴 B5.4 Add unit test comparing loss value against manual computation

### B6 — Fix BiomedicalDomain arity

**File**: `src/domains/biomedical/domain.jl`

- 🔴 B6.1 Add 4-arg `extract_entities(domain, text, config, llm_client=nothing)` overload
- 🔴 B6.2 Verify extraction.jl `discover_head_entities` calls the 4-arg version
- 🔴 B6.3 Add test: `extract_entities` with and without LLM client

---

## STREAM C — External integrations

### C1 — LLM client abstraction

**File**: `src/llm/helper.jl`

- 🔴 C1.1 Define `AbstractLLMClient` abstract type with `call_llm(client, prompt) -> String`
- 🔴 C1.2 Implement `MockLLMClient` returning canned responses (for CI)
- 🔴 C1.3 Implement `OpenAIClient` (reads `OPENAI_API_KEY` from env)
- 🔴 C1.4 Implement `GeminiClient` (reads `GOOGLE_API_KEY` from env)
- 🔴 C1.5 Add prompt templates as `const` strings: head discovery, relation matching, tail formation, FActScore verification
- 🔴 C1.6 Add response parser for entity list (bullet/numbered/plain)
- 🔴 C1.7 Add rate limiting (token bucket) and retry logic
- 🔴 C1.8 Add result caching (file-based, keyed by SHA256 of prompt)

### C2 — UMLS REST API client

**File**: `src/biomedical/umls.jl`

- 🔴 C2.1 Implement TGT/ST authentication (reads `UMLS_API_KEY` from env)
- 🔴 C2.2 Implement `lookup_cui(client, entity_text) -> Vector{(CUI, score)}`
- 🔴 C2.3 Implement `retrieve_triples(client, cui, allowed_relations) -> Vector{SemanticTriple}`
- 🔴 C2.4 Add local SQLite cache for CUI lookups and triples
- 🔴 C2.5 Implement `MockUMLSClient` with fixture data for CI
- 🔴 C2.6 Add rate limiting (UMLS allows ~20 req/s)

### C3 — SapBERT entity linking

**File**: `src/biomedical/entity_linking.jl`

- 🔴 C3.1 Define `EntityLinker` abstract type
- 🔴 C3.2 Implement `SapBERTLinker` using ONNX model via `ONNX.jl` or Python subprocess
- 🔴 C3.3 Build/load ANN index (cosine similarity over UMLS concept embeddings)
- 🔴 C3.4 Implement `link_entity(linker, text) -> Vector{(CUI, score)}`
- 🔴 C3.5 Implement `MockEntityLinker` with fixture mappings for CI
- 🔴 C3.6 Add character 3-gram Jaccard filter as secondary reranker

### C4 — Embedding client for contextual selection

**File**: `src/llm/embeddings.jl`

- 🔴 C4.1 Define `AbstractEmbeddingClient` with `embed(client, text) -> Vector{Float32}`
- 🔴 C4.2 Implement `GeminiEmbeddingClient`
- 🔴 C4.3 Implement `MockEmbeddingClient` (random unit vectors, seeded)
- 🔴 C4.4 Add cosine similarity utility `cosine_sim(a, b)`

---

## STREAM D — Training pipeline completeness

### D1 — Seed KG injection pipeline

**File**: `src/training/seed_injection.jl`

- 🔴 D1.1 Stage 1: entity linking using C2 + C3
- 🔴 D1.2 Stage 2: UMLS triple retrieval per CUI
- 🔴 D1.3 Stage 3: contextual selection with C4 (top-40 per entity)
- 🔴 D1.4 Stage 4: bucket by relation type, fill proportionally
- 🔴 D1.5 Inject selected triples into leafy chain graphs
- 🔴 D1.6 Target ~28k triples for diabetes-scale dataset
- 🔴 D1.7 End-to-end test with mock clients

### D2 — Real MNM training step

**File**: `src/training/mnm.jl`

- 🔴 D2.1 Call real `forward_pass_mnm` with masked graph as input
- 🔴 D2.2 Compute joint loss: `L = L_MLM + L_MNM`
- 🔴 D2.3 Compute gradients via `Zygote.gradient`
- 🔴 D2.4 Apply `Flux.update!` on all parameters
- 🔴 D2.5 Verify gradient flows through H-GAT (check `∂L/∂W_hgat ≠ 0`)
- 🔴 D2.6 Add test: loss decreases over 3 steps on fixed toy input

### D3 — Checkpoint system

**File**: `src/models/persistence.jl`

- 🔴 D3.1 Save model weights + `GraphMERTConfig` as JLD2
- 🔴 D3.2 Save optimizer state alongside weights (for resume)
- 🔴 D3.3 Load and restore optimizer state
- 🔴 D3.4 Versioned checkpoint directory (epoch_N, best, latest symlinks)

### D4 — Training metrics logging

**File**: `src/training/pipeline.jl`

- 🔴 D4.1 Log per-step: `step`, `mlm_loss`, `mnm_loss`, `combined_loss`, `elapsed_ms`
- 🔴 D4.2 Log per-epoch: averages, learning rate, checkpoint path
- 🔴 D4.3 CSV output to `logs/train_TIMESTAMP.csv`
- 🔴 D4.4 Optional TensorBoard writer via `TensorBoardLogger.jl`

### D5 — Validation loop

**File**: `src/training/pipeline.jl`

- 🔴 D5.1 Run extraction on held-out texts after each epoch
- 🔴 D5.2 Compute FActScore\* against source texts
- 🔴 D5.3 Log validation metrics alongside train metrics
- 🔴 D5.4 Save best-validation checkpoint

---

## STREAM E — Evaluation completeness

### E1 — FActScore\* with LLM verification

**File**: `src/evaluation/factscore.jl`

- 🔴 E1.1 Implement `calculate_factscore_star(kg, source_texts, llm_client)`
- 🔴 E1.2 Prompt template: claim verification against source passage
- 🔴 E1.3 Parse LLM response: `:supported`, `:not_supported`, `:contradicted`
- 🔴 E1.4 Return `FActScore(score, precision, recall, f1, total, correct, incorrect)`
- 🔴 E1.5 Add mock-LLM test reproducing a known score on a fixed KG

### E2 — ValidityScore

**File**: `src/evaluation/validity.jl`

- 🔴 E2.1 Implement `calculate_validity_score(kg, domain) -> ValidityScore`
- 🔴 E2.2 For each relation, call `domain.validate_relation(head_type, rel_type, tail_type)`
- 🔴 E2.3 Score = fraction of valid relations

### E3 — GraphRAG evaluation

**File**: `src/evaluation/graphrag.jl`

- 🔴 E3.1 Define `GraphRAGConfig` (question set path, retrieval top-k)
- 🔴 E3.2 Implement `evaluate_graphrag(kg, question_set, llm_client)`
- 🔴 E3.3 Retrieval: find triples relevant to each question
- 🔴 E3.4 Answer generation: feed relevant triples as context to LLM
- 🔴 E3.5 Score: exact match / F1 against gold answers

---

## STREAM F — Test correctness

### F1 — Fix known test failures

- 🔴 F1.1 `test/integration/test_extraction_pipeline.jl:103` — fix `match_relations_for_entities` call to 4 args
- 🔴 F1.2 `test/unit/test_api.jl:41` vs `test/integration/.../177` — pick one empty-text contract, update the other
- 🔴 F1.3 Run full test suite; list all failures; file issues for each

### F2 — Ensure CI skips external-API tests

- 🔴 F2.1 Gate all UMLS/LLM/SapBERT tests with `ENV["RUN_EXTERNAL_TESTS"] == "1"`
- 🔴 F2.2 Confirm CI config only runs gated tests by default

### F3 — Integration test: training converges

- 🔴 F3.1 Train 3 steps on 4 toy sentences with mock seed KG
- 🔴 F3.2 Assert `loss[step=3] < loss[step=1]`
- 🔴 F3.3 Assert returned model is `GraphMERTModel`, not mock

### F4 — Persistence round-trip test

- 🔴 F4.1 Save model to temp file
- 🔴 F4.2 Load from temp file
- 🔴 F4.3 Assert `loaded_model(x) ≈ original_model(x)` for fixed toy input

### F5 — Remove/gate README performance claims

- 🔴 F5.1 Remove or annotate "5,200+ tokens/sec" claim as "from mock training"
- 🔴 F5.2 Remove or annotate "FActScore: 70.1%" claim as "not yet reproduced"
- 🔴 F5.3 Add note: benchmarks will be updated once real training runs

---

## STREAM G — Extension hooks

### G1 — Distillation loss

**File**: `src/training/distillation.jl` (new)

- 🔴 G1.1 Define `DistillationConfig`
- 🔴 G1.2 Implement `distillation_loss(student_kg, teacher_kg, evidence_docs, llm)`
- 🔴 G1.3 Wire into training pipeline as optional `distillation_config` parameter

### G2 — Multi-domain seed injection

**File**: `src/training/seed_injection.jl`

- 🔴 G2.1 Define `OntologySource` abstract type with `retrieve_triples`, `get_allowed_relations`
- 🔴 G2.2 Implement `UMLSOntologySource` (current hardcoded behavior)
- 🔴 G2.3 Implement `WikidataOntologySource` (SPARQL endpoint)
- 🔴 G2.4 Make `SeedInjectionConfig.ontology_source::OntologySource`

### G3 — KG completion mode

**File**: `src/api/extraction.jl`

- 🔴 G3.1 Add `extend_knowledge_graph(existing_kg, new_text, model; options)` API
- 🔴 G3.2 Use existing KG entities as seeds for head discovery
- 🔴 G3.3 Skip already-known triples in Stage 5 filtering
- 🔴 G3.4 Return merged KG with provenance for new vs existing triples

---

## Summary counts

| Stream           | Tasks  | Subtasks |
| ---------------- | ------ | -------- |
| A (correctness)  | 5      | 35       |
| B (architecture) | 6      | 30       |
| C (integrations) | 4      | 25       |
| D (training)     | 5      | 24       |
| E (evaluation)   | 3      | 14       |
| F (tests)        | 5      | 16       |
| G (extensions)   | 3      | 11       |
| **Total**        | **31** | **155**  |

---

_See `PARITY_PLAN.md` for phases and dependency order._
_See `RETROSPECTIVE_SPEC.md` for data and API contracts._
