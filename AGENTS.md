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

- Domains: biomedical and Wikipedia modules exist, but domain-level `extract_entities` / `extract_relations` are still partially stubbed; extraction falls back to heuristics when domains are not registered or fail.
- Tests: core extraction unit tests (`test_extraction.jl`) pass; `test_api.jl` and the full `runtests.jl` suite still have known failures (domain wiring, model persistence). See `reports/CODE_REVIEW.md` for the latest list.
- Model: RoBERTa + H-GAT + leafy chain are implemented; MNM forward pass, seed injection, and some evaluation and domain helpers are still incomplete.
- Specs: treat `original_paper/expanded_rewrite/` and `reports/` as canonical; this file only routes you to them and encodes safe agent behavior.

---

## 2. Read this first (agent onboarding)

When you start a new task, skim these in order and only dive deeper if needed:

1. **Roadmap and current status (specs)**
   - `original_paper/expanded_rewrite/00-IMPLEMENTATION-ROADMAP-SUMMARY.md`
   - `original_paper/expanded_rewrite/STATUS-SUMMARY.md`
   - `reports/CODE_REVIEW.md` and `reports/GENERALIZATION_PLAN.md`
2. **Core architecture and types**
   - `original_paper/expanded_rewrite/01-architecture-overview.md`
   - `original_paper/expanded_rewrite/11-data-structures.md`
   - `GraphMERT/src/GraphMERT.jl` and `GraphMERT/src/types.jl`
3. **Critical algorithms**
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

Agents looking for high-impact tasks should focus here:

1. **Training pipeline completeness**
   - Implement proper MNM batching and integration in `GraphMERT/src/training/pipeline.jl`.
   - Replace any `mnm_batch = nothing` or `TODO` stubs with spec-compliant logic.

2. **Tail prediction and formation**
   - `predict_tail_tokens` and `form_tail_from_tokens` currently rely on placeholder behavior.
   - Use `original_paper/expanded_rewrite/09-triple-extraction.md` to:
     - Pull real scores from `GraphMERTModel`.
     - Implement meaningful tail candidates.
     - Optionally integrate the helper LLM for tail formation in a controlled, testable way.

3. **Model persistence**
   - Wire `load_model` and persistence utilities to real on-disk formats, backed by `models/persistence.jl` and `scripts/import_model_weights.jl`.
   - Add tests that:
     - Save a small model.
     - Reload and verify equivalence for key operations.

4. **Domain tests**
   - Add `test/domains/test_biomedical_domain.jl`, `test/domains/test_wikipedia_domain.jl`, and `test/domains/test_domain_registry.jl`.
   - Verify extraction, validation, confidence scoring, and evaluation metrics through the domain interface.

5. **Placeholder cleanup and benchmarks**
   - Replace placeholder implementations in:
     - `GraphMERT/src/benchmarking/benchmarks.jl`
     - `GraphMERT/src/optimization/*.jl`
     - Any remaining placeholder functions in domain graph utilities.
   - Ensure benchmarks reflect actual performance claims in the README.

6. **Visualization**
   - Stabilize and document the visualization layer in `GraphMERT/src/visualization/*.jl`.
   - Add a dedicated docs page and ensure `examples/visualization_example.jl` and `examples/wikipedia/02_wikipedia_visualization.jl` stay in sync.

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
