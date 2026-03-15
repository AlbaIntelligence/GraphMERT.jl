# GraphMERT.jl — Parity Plan

> Plan to bring the Julia implementation to full paper and extension parity.
> See `TASK_LIST.md` for the exhaustive work items.

---

## 1. Current health summary

| Subsystem | Status | Blocker |
|----------|--------|---------|
| RoBERTa architecture | ✅ Structurally correct | `max_position_embeddings=512` must be 1024 |
| H-GAT | ✅ Structurally correct | Not wired into embedding layer (relpos injection) |
| Attention decay mask | ⚠️ Implemented but not integrated | Missing from every transformer layer |
| Leafy chain graph | ✅ Correct | — |
| MLM masking + loss | ✅ Correct | — |
| MNM masking | ✅ Correct | Wrong loss function (crossentropy → BCE with logits) |
| Joint training pipeline | ❌ Mock only | `train_graphmert` returns `MockGraphMERTModel`, uses `rand()` losses |
| Triple extraction (stages 1–2) | ⚠️ Partial | Domain arity mismatch (3-arg vs 4-arg) |
| Triple extraction (stages 3–4) | ❌ Placeholders | `rand()` logits; `"entity_N"` tails |
| Model persistence | ❌ Stubs | All weight I/O warns and returns false |
| FActScore* | ❌ Bug | Cartesian product instead of iterating relations |
| ValidityScore | ⚠️ Partial | — |
| LLM client | ❌ Stub | Needed for extraction stages 1,2,4 and evaluation |
| UMLS client | ❌ Stub | Needed for seed injection |
| SapBERT linking | ❌ Not implemented | Needed for seed injection |

---

## 2. Confirmed defects

| ID | File | Lines | Defect |
|----|------|-------|--------|
| D1 | `training/pipeline.jl` | 226 | Returns `MockGraphMERTModel` |
| D2 | `training/pipeline.jl` | 285–288 | Losses are `rand()`; no `Flux.update!` call |
| D3 | `api/extraction.jl` | 170 | Stage 3 tail probs are `rand(Float32, vocab_size)` |
| D4 | `api/extraction.jl` | 200–205 | Stage 4 tails are `"entity_$(token_id)"` strings |
| D5 | `models/persistence.jl` | 131–188 | All weight I/O is `@warn` + return false/nothing |
| D6 | `evaluation/factscore.jl` | 197–212 | O(N²×M) cartesian product; iterates all entity pairs |
| D7 | `architectures/roberta.jl` | 43 | `max_position_embeddings=512` < sequence length 1024 |
| D8 | `training/mnm.jl` | 229, 259 | `Flux.crossentropy` with wrong args; needs BCE with logits |
| D9 | `test/…/test_extraction_pipeline.jl` | 103 | Calls `match_relations_for_entities` with 2 args; needs 4 |
| D10 | `architectures/roberta.jl` | Transformer layers | Decay mask not passed to each layer |
| D11 | `domains/biomedical/domain.jl` | — | `extract_entities` is 3-arg; extraction path tries 4-arg |
| D12 | `test/unit/test_api.jl` vs integration | 41–44 / 177 | Empty-text contract contradiction |

---

## 3. Work streams

### Stream A — Correctness (P0 bugs that block any real output)
- **A1** Fix training pipeline: real `GraphMERTModel`, real `Flux.update!`, real losses
- **A2** Fix tail prediction (Stage 3): use real model logits at leaf positions
- **A3** Fix tail formation (Stage 4): use LLM client to propose entities
- **A4** Fix model persistence: JLD2 weight round-trip
- **A5** Fix FActScore cartesian product bug

### Stream B — Architecture alignment (Python reference parity)
- **B1** Fix `max_position_embeddings = 1024`
- **B2** Implement real `GraphMERTModel.forward` matching `graphmert.py`:
  - H-GAT → rel embeddings → inject at `[REL]` positions → transformer layers
  - Each layer receives `distance_matrix` for decay mask
- **B3** Fix MNM loss: `Flux.binarycrossentropy` (or `logitbinarycrossentropy`)
- **B4** Fix `BiomedicalDomain.extract_entities` arity (3-arg vs 4-arg)

### Stream C — External integrations (with mock mode for CI)
- **C1** LLM client: abstract interface + Gemini/OpenAI implementations + mock
- **C2** UMLS REST client: auth, rate-limiting, CUI lookup, local cache
- **C3** SapBERT entity linking: ONNX export or Python subprocess wrapper
- **C4** Gemini embeddings for contextual triple selection

### Stream D — Training pipeline completeness
- **D1** Seed KG injection (4-stage pipeline using C1–C4)
- **D2** Real MNM training step with gradient flow (joint MLM + MNM)
- **D3** Checkpoint system: save/load weights + optimizer state
- **D4** Training metrics logging (step loss, epoch stats, validation)
- **D5** Validation loop with real FActScore* after each epoch

### Stream E — Evaluation completeness
- **E1** Fix and test FActScore* (uses C1 for LLM verification)
- **E2** Implement ValidityScore with ontology validation
- **E3** Implement GraphRAG evaluation pipeline

### Stream F — Test correctness
- **F1** Fix arity mismatch in `test_extraction_pipeline.jl` line 103
- **F2** Reconcile empty-text contract between unit and integration tests
- **F3** Add integration test: 1-epoch training on mock data, loss decreases
- **F4** Add persistence round-trip test
- **F5** Gate/remove README performance claims until from real training

### Stream G — Extension hooks
- **G1** Distillation loss (`distillation_loss(student, teacher, evidence)`)
- **G2** Multi-domain seed injection via `OntologySource` abstraction
- **G3** KG completion mode (extend existing KG)

---

## 4. Execution order

### Phase 1 — Unblock real training (P0, ~2 weeks)
B1 → B2 → B3 → A1 → D2

### Phase 2 — Unblock real extraction (~2 weeks)
C1 (mock) → A2 → A3 → B4 → F1 → F2

### Phase 3 — Persistence and evaluation (~1 week)
A4 → A5 → E1 → F4

### Phase 4 — Full pipeline (~3 weeks)
C2 → C3 → C4 → D1 → D3 → D4 → D5 → E2 → E3

### Phase 5 — Test hardening (~1 week)
F3 → F5 → all remaining unit tests

### Phase 6 — Extensions (ongoing)
G1 → G2 → G3

---

## 5. Extension descriptions

### DRAG-style KG distillation
New file `src/training/distillation.jl`:
- LLM teacher produces soft KG from same text
- `distillation_loss(student_kg, teacher_kg, evidence_passages)` aligns them
- Reduces hallucination in student extraction

### LinguGKD-style feature alignment
New file `src/training/feature_alignment.jl`:
- Contrastive + layer-adaptive alignment between LLM and H-GAT feature spaces
- Hooks into H-GAT forward pass

### Multi-domain seed injection
Extend `SeedInjectionConfig` to accept `OntologySource` (replaces hardcoded UMLS).
Implement `WikidataSource`, `CustomCSVSource` as initial alternatives.

### Low-resource KG completion
New extraction mode that takes an existing KG as context and predicts missing triples,
using negative sampling from the existing graph.

---

*See `RETROSPECTIVE_SPEC.md` for data and API contracts.*
*See `TASK_LIST.md` for the exhaustive work items.*
