# AGENTS.md

Guidelines for AI coding agents and automated tools working on **GraphMERT.jl**.

This file explains **how to think about the project**, **where to read first**, and **how to safely make changes** that match the existing architecture and roadmap.

---

## 1. Project overview

GraphMERT.jl is a **Julia implementation** of the GraphMERT algorithm:

- Builds **knowledge graphs** from unstructured text.
- Uses a **RoBERTa encoder + H-GAT**.
- Represents documents as **leafy chain graphs**.
- Has **dual training objectives** (MLM + MNM).
- Uses a **pluggable domain system** (biomedical, Wikipedia, and future domains).
- Integrates with **helper LLMs** and external KGs (UMLS, Wikidata) via domain modules.

The implementation is tightly coupled to a **specification set** in `original_paper/expanded_rewrite/`. Those documents are the **ground truth** for algorithms and data structures.

---

## 1.1 Quick snapshot for agents

- **Project status**: See `reports/PROJECT_STATUS.md` for overview; `reports/PARITY_PLAN.md` for current defect list and work streams.
- **Confirmed P0 bugs** (nothing works end-to-end until these are fixed):
  - `train_graphmert` returns `MockGraphMERTModel` with `rand()` losses — no real training.
  - `predict_tail_tokens` (Stage 3) uses `rand(Float32, vocab_size)` — no real inference.
  - `form_tail_from_tokens` (Stage 4) returns `"entity_N"` strings — no real extraction.
  - All weight I/O in `persistence.jl` is stubbed — `save_model`/`load_model` do nothing.
  - `filter_triples_by_confidence` is an O(N²×M) cartesian product — FActScore is wrong.
  - `max_position_embeddings=512` but sequence length is 1024 — embedding table too small.
- **Spec**: `reports/RETROSPECTIVE_SPEC.md` defines all type, algorithm, and API contracts.
- **Roadmap**: `reports/PARITY_PLAN.md` lists 12 confirmed defects and 7 work streams.
- **Tasks**: `reports/TASK_LIST.md` has 155 atomic subtasks ordered by dependency.
- Domains: biomedical and Wikipedia modules exist; `BiomedicalDomain.extract_entities` has a 3-arg signature but the extraction path tries 4 args — always falls back to heuristics.
- Tests: known failures in `test_api.jl` (arity mismatch line 103) and contradicting empty-text contracts between unit and integration tests. See `reports/CODE_REVIEW.md`.
- Reliability pipeline: `validate_kg`, `get_provenance`, `evaluate_factscore`, `clean_kg` are partially implemented; `evaluate_factscore` has the cartesian-product bug above.
- **Do not trust README performance numbers** — they are from mock training with random losses.

---

## 2. Read this first (agent onboarding)

When you start a new task, skim these in order and only dive deeper if needed:

1. **Contracts, defects, and tasks (read first)**
   - `reports/RETROSPECTIVE_SPEC.md` — canonical type, algorithm, and API contracts
   - `reports/PARITY_PLAN.md` — confirmed defect table (D1–D12) and 7 work streams
   - `reports/TASK_LIST.md` — 155 atomic subtasks; find your stream and start there
2. **Roadmap and current status**
   - `original_paper/expanded_rewrite/00-IMPLEMENTATION-ROADMAP-SUMMARY.md`
   - `original_paper/expanded_rewrite/STATUS-SUMMARY.md`
   - `reports/CODE_REVIEW.md` and `reports/GENERALIZATION_PLAN.md`
   - **Reference sources and encoder:** `Contextual_information.md` (root), `original_paper/graphMERT.ipynb`, `original_paper/GraphMert/` (Python clone); see `reports/REFERENCE_SOURCES_AND_ENCODER.md`.
3. **Core architecture and types**
   - `original_paper/expanded_rewrite/01-architecture-overview.md`
   - `original_paper/expanded_rewrite/11-data-structures.md`
   - `GraphMERT/src/GraphMERT.jl` and `GraphMERT/src/types.jl`
4. **Critical algorithms**
   - Leafy chain graphs: `original_paper/expanded_rewrite/02-leafy-chain-graphs.md`, `GraphMERT/src/graphs/leafy_chain.jl`
   - MNM training: `original_paper/expanded_rewrite/07-training-mnm.md`, `GraphMERT/src/training/mnm.jl`
   - Seed KG injection: `original_paper/expanded_rewrite/08-seed-kg-injection.md`, `GraphMERT/src/training/seed_injection.jl`, `GraphMERT/src/seed_injection.jl`
   - Triple extraction: `original_paper/expanded_rewrite/09-triple-extraction.md`, `GraphMERT/src/api/extraction.jl`
4. **Domain system and migration**
   - `reports/GENERALIZATION_PLAN.md`, `reports/GENERALIZATION_SUMMARY.md`, `reports/IMPLEMENTATION_FILES.md`
   - Domain docs: `reports/BIOMEDICAL_DOMAIN_IMPLEMENTATION.md`, `reports/WIKIPEDIA_DOMAIN_IMPLEMENTATION.md`, `reports/DOMAIN_DEVELOPER_GUIDE.md`, `reports/DOMAIN_USAGE_GUIDE.md`, `reports/MIGRATION_GUIDE.md`
   - Code: `GraphMERT/src/domains/interface.jl`, `GraphMERT/src/domains/registry.jl`, and the `GraphMERT/src/domains/biomedical*/` and `GraphMERT/src/domains/wikipedia*/` trees.
5. **Visualization, examples, and user behavior**
   - High-level: `README.md`, `GraphMERT/docs/src/index.md`
   - API docs: `GraphMERT/docs/src/api/index.md`, `GraphMERT/docs/src/api/domain.md`
   - Visualization: `reports/GRAPH_VISUALIZATION_RESEARCH.md`, `GraphMERT/src/visualization/*.jl`
   - Examples overview: `reports/EXAMPLES_README.md`, `examples/*.jl` (start from `examples/00_domain_switching_demo.jl` and `01_*` examples per domain).

Agents should **align all changes** with these specs and docs unless the task explicitly says to refactor or redesign.

---

## 3. Canonical APIs and flows

### 3.1 Knowledge graph extraction

Preferred workflow:

1. **Load or construct a model**:

```julia
using GraphMERT

model = load_model("path/to/pretrained/graphmert_model.onnx")  # persistence-backed
```

2. **Enable and register a domain**:

```julia
include("GraphMERT/src/domains/biomedical.jl")
bio = load_biomedical_domain()
register_domain!("biomedical", bio)
```

3. **Extract a graph**:

```julia
options = ProcessingOptions(domain = "biomedical")
kg = extract_knowledge_graph("Some biomedical text", model; options = options)
```

When modifying or adding APIs, **preserve this shape**:

- Domain is always specified via `ProcessingOptions.domain`.
- Extraction flows through `GraphMERT.extract_knowledge_graph(text, model; options)` in `api/extraction.jl`.
- Domain-specific logic routes through the `DomainProvider` interface.

### 3.2 Domains

- The domain abstraction lives in:
  - `GraphMERT/src/domains/interface.jl`
  - `GraphMERT/src/domains/registry.jl`
- Biomedical and Wikipedia are **reference implementations**:
  - `GraphMERT/src/domains/biomedical.jl` and `GraphMERT/src/domains/biomedical/*.jl`
  - `GraphMERT/src/domains/wikipedia.jl` and `GraphMERT/src/domains/wikipedia/*.jl`

Agents that add new domains or extend existing ones must:

- Implement **all required `DomainProvider` methods** (see `GraphMERT/docs/src/api/domain.md`).
- Keep **domain-specific** code out of core modules (`GraphMERT.jl`, `types.jl`, `api/*`, `training/*`, etc.).
- Update or add **examples and tests** for the new domain.

---

## 4. Testing strategy for agents

Whenever you change behavior, prioritize tests:

1. **Unit tests** (fast feedback)
   - Located in `GraphMERT/test/unit/*.jl`.
   - Cover types, utilities, domain logic, serialization, etc.

2. **Integration tests**
   - Located in `GraphMERT/test/integration/*.jl`.
   - Cover:
     - Leafy chain graph pipeline
     - Training pipeline
     - Extraction pipeline
     - Domain integration
     - Evaluation (FActScore, Validity, GraphRAG)
     - Full pipeline orchestration

3. **Performance tests**
   - Located in `GraphMERT/test/performance/*.jl` and top-level `test_*performance*.jl`.

4. **Progressive tests**
   - `GraphMERT/src/testing/progressive.jl` and `GraphMERT/test/test_progressive_testing.jl` define a progressive testing strategy; use these to avoid regressions as features grow.

**Agent rules:**

- **Never** remove or significantly weaken existing tests without strong justification in commit messages or comments.
- When adding new functionality, **add or extend tests first** (TDD where feasible), guided by the spec documents.
- Clearly separate tests that require **external APIs** (UMLS/Wikidata/LLMs) and mark them so they can be skipped in CI by default (for example, via environment flags).

---

## 5. High-impact areas for agents

> For a prioritized, atomic task list see **`reports/TASK_LIST.md`**. The streams below map to that document.

### Stream A — Fix P0 correctness bugs (highest priority)

Work items: `TASK_LIST.md` §A1–A5

1. **A1 — Real training pipeline**: Replace `MockGraphMERTModel` + `rand()` losses + missing `Flux.update!` in `GraphMERT/src/training/pipeline.jl`.
2. **A2/A3 — Real tail prediction/formation**: Replace `rand(Float32, vocab_size)` and `"entity_N"` strings in `GraphMERT/src/api/extraction.jl` stages 3–4.
3. **A4 — Model persistence**: Implement JLD2 weight save/load in `GraphMERT/src/models/persistence.jl`.
4. **A5 — FActScore bug**: Fix O(N²×M) cartesian product in `GraphMERT/src/evaluation/factscore.jl:197`.

### Stream B — Architecture alignment (Python reference parity)

Work items: `TASK_LIST.md` §B1–B6

5. **B1** — Fix `max_position_embeddings = 512` → 1024 in `RoBERTaConfig`.
6. **B2** — Integrate attention decay mask (`exp(-α×d)`) into every transformer layer.
7. **B3/B4** — Wire H-GAT relation embeddings into embedding layer; implement real `GraphMERTModel.forward`.
8. **B5** — Fix MNM loss: `crossentropy` → `logitbinarycrossentropy`.
9. **B6** — Fix `BiomedicalDomain.extract_entities` arity (3-arg → 4-arg with default).

### Stream C–D — External integrations and full training

Work items: `TASK_LIST.md` §C1–C4, §D1–D5

10. LLM client, UMLS client, SapBERT entity linking, seed KG injection pipeline.
11. Real gradient-flowing MNM training step; checkpoint system; validation loop.

### Stream E–G — Evaluation, tests, extensions

Work items: `TASK_LIST.md` §E–G

12. FActScore* with LLM verification, ValidityScore, GraphRAG.
13. Fix known test failures; add integration + persistence round-trip tests.
14. Distillation hooks, multi-domain seed injection, KG completion mode.

---

## 6. Style and safety guidelines

### 6.1 Code style

- Follow existing **Julia style** in this repo:
  - Use concrete types where possible for performance-critical paths.
  - Prefer clear, descriptive names over cryptic abbreviations.
  - Keep functions small and single-purpose, especially around domain logic and pipeline stages.
- For public APIs:
  - Add docstrings with arguments, returns, and a small example.
  - Avoid breaking changes; if you must, update docs and mention migration paths.

### 6.2 Safety and external dependencies

- Do **not** hard-wire real API keys or secrets. UMLS/Wikidata/LLM integrations must:
  - Read config from environment or config files.
  - Support a **mock mode** used in tests.
- When modifying external integrations:
  - Preserve the ability to run **offline** (mocking, cached data, or conditional skips).
  - Avoid making network calls during unit tests by default.

---

## 7. How agents should plan work

When assigned a task:

1. **Locate the spec**:
   - Find the relevant document under `original_paper/expanded_rewrite/` and read the corresponding sections.
2. **Trace the implementation**:
   - Find the matching module in `GraphMERT/src/**.jl`.
   - Identify existing tests in `GraphMERT/test/**.jl`.
3. **Plan changes**:
   - List the minimal set of functions and files you need to touch.
   - Decide what tests need to be added or extended.
4. **Implement with tests**:
   - For nontrivial changes, add tests first or alongside code.
   - Keep the domain-agnostic core clean; push domain-specific logic into domain modules.
5. **Update docs and examples when APIs change**:
   - If you adjust public behavior, ensure the README, docs, and examples stay consistent.

---

## 8. When in doubt

If a spec and the current code disagree:

- **Spec wins** for algorithmic details, unless there is an explicit note in reports or roadmap documents explaining a deliberate deviation.
- Leave a short note (for example, in commit messages) explaining the discrepancy and how you resolved it.
- Prefer **small, reversible changes** and comprehensive tests to guard behavior.

---

## 9. Token budget and editing guidelines

- **Overall goal**: keep files **small, focused, and non-duplicative** so agents and humans can load context cheaply.
- **Line limit**: Target **400–500 lines** per file for specs and operational docs. If a file grows beyond that, split it (e.g. short stub + `*-FULL.md` or per-section files) and keep the main file as an index or summary that links to the parts.
- **When to create a new file**: When adding content would push an existing doc over the line limit, or when the content is a distinct concern (e.g. new domain or module). Prefer one more small file over one large file.
- **Specs and plans** (`original_paper/expanded_rewrite/*.md`, `reports/*.md`):
  - Treat as **canonical**; do not duplicate large sections elsewhere. **Link** instead of copying.
  - Use short-named files (e.g. `00-IMPLEMENTATION-ROADMAP.md`) for the concise version and `*-FULL.md` for the long form.
- **Operational docs** (`AGENTS.md`, `README.md`, domain guides):
  - Keep each under the line limit; use short lists and pointers to deeper specs.
- **Narrative / archive material**:
  - Add a short **front-matter summary** and mark as background; reference from core docs, do not duplicate.
- **Code and tests**:
  - Avoid long comment blocks that restate specs; link to the doc or test. Centralize shared helpers in tests.

Agents implementing new features should **reduce or keep steady** line count in touched files unless adding genuinely new functionality.

### 9.1 Delegation (sub-agents)

For automated workflows, these roles can be mapped to separate agents:

- **doc-refactor-agent**: Docs and reports only; optimize structure, reduce duplication, add summaries.
- **api-doc-agent**: `GraphMERT/docs/src/api/*.md` and public API comments; keep aligned with code and tests.
- **code-structure-agent**: Split or reorganize large `.jl` files without changing behavior.

---

By following this `AGENTS.md`, AI agents and human contributors can make changes that are **aligned with the research spec**, **safe for users**, and **consistent with the evolving architecture** of GraphMERT.jl.
