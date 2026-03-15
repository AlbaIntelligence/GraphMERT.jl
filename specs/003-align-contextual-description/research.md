# Research: Reliability Pipeline (003-align-contextual-description)

**Status**: Complete  
**Last Updated**: 2026-03-15

---

## 1. Provenance record shape

**Decision**: Attach a structured provenance record to each triple (relation): at minimum **document identifier** and **segment/sentence identifier** (or character span). Optionally keep a legacy string field for backward compatibility.

**Rationale**: Contextual_information.md requires sentence-level extraction so every triple is traceable to a source span. Existing `Relation`/`KnowledgeRelation` have a `provenance::String`; we add or extend to a structured type so that (document_id, segment_id) are first-class and queryable without parsing strings.

**Alternatives considered**:
- **Provenance as string only**: Rejected; harder to query and validate; no guaranteed structure.
- **Separate provenance store (keyed by triple id)**: Rejected for MVP; increases indirection; in-memory attachment is sufficient and matches “each triple carries provenance.”

**Implementation note**: Extend or add type in `types.jl`; extraction pipeline in `api/extraction.jl` must populate it at triple creation time. Domain providers may supply document/segment boundaries.

---

## 2. ValidityScore definition and ontology validation

**Decision**: **ValidityScore** = proportion (or count) of triples that satisfy the seed ontology (allowed relation types, entity types, domain/range constraints). Run validation as a function that takes (KG, ontology) and returns a score plus per-triple or aggregate report. Use existing `evaluation/validity.jl` as the implementation home; extend to accept domain ontology from DomainProvider when available.

**Rationale**: Paper and Contextual_information.md cite ValidityScore (~68.7% for GraphMERT vs ~43% LLM); ontology-consistent construction is a differentiator. Domain providers already have or can expose relation/entity schemas.

**Alternatives considered**:
- **Validity as binary (valid/invalid) only**: Rejected; spec and narrative require a score for comparison and monitoring.
- **Ontology from external file only**: Rejected; domains (biomedical, Wikipedia) should supply ontology via DomainProvider for consistency with existing architecture.

---

## 3. FActScore definition and factuality evaluation

**Decision**: **FActScore** = proportion of triples deemed factually correct when compared to reference or ground-truth data (e.g., curated triples or gold KG). Implement as a function that takes (KG, reference_data) and returns a score; reference data is optional—when absent, factuality evaluation is not run (no score). Use/extend `evaluation/factscore.jl`; support domain-specific reference loaders where applicable.

**Rationale**: Contextual_information.md defines factuality as “triples supported by the corpus” and cites FActScore (~69.8%); implementation uses reference comparison when available. Optional reference keeps the system usable in settings where no gold data exists.

**Alternatives considered**:
- **FActScore without reference (e.g., confidence-only)**: Rejected for the metric name; FActScore in the paper is defined relative to ground truth. Confidence-based filtering is part of KG cleaning, not FActScore.
- **Mandatory reference data**: Rejected; spec and assumptions state reference may be optional (e.g., evaluation harness or specific domains).

---

## 4. KG cleaning policy

**Decision**: **KG cleaning** = configurable step that removes or rectifies triples that are (a) unsupported (no or weak provenance/support), (b) low-confidence (below threshold), or (c) contradicted (e.g., by another triple or by reference). Policy is configurable (thresholds, whether to drop vs. flag, optional rectification rules). Output is a new KG (cleaned) that can be used as augmented seed.

**Rationale**: Contextual_information.md describes a KG cleaning stage that boosts FActScore and supports iterative improvement. Configurable policy allows different domains and use cases (strict vs. lenient).

**Alternatives considered**:
- **Fixed rules only**: Rejected; spec says “configurable policy.”
- **Cleaning as a separate package**: Rejected; cleaning is part of the reliability pipeline and should live in GraphMERT for cohesion and domain integration.

---

## 5. Encoder in extraction path

**Decision**: **Wire model persistence** so that `load_model(path)` returns a model object that includes the RoBERTa encoder (and H-GAT) and is the same object type used in training. Extraction path in `api/extraction.jl` already receives “model”; ensure that when a full checkpoint is loaded (via `models/persistence.jl`), the extraction pipeline calls the encoder (no stub or ONNX-only path that bypasses it). Optionally support loading pretrained RoBERTa weights into the Julia RoBERTa module.

**Rationale**: REFERENCE_SOURCES_AND_ENCODER and spec require extraction to use the compact encoder when a full model is loaded; currently persistence is a stub so extraction may not run RoBERTa.

**Alternatives considered**:
- **Keep ONNX-only inference**: Rejected for “full implementation”; spec requires encoder-in-path.
- **Replace RoBERTa with another encoder**: Rejected; constitution and spec keep RoBERTa-class architecture.

---

## 6. Iterative seed re-use

**Decision**: **Augmented seed** = allow a cleaned or curated KG to be supplied as additional seed input for a subsequent extraction or training run. Document at least one path (e.g., “export cleaned KG → load as seed file/config → run extraction or training”). Implementation may extend existing seed injection (e.g., `training/seed_injection.jl`, domain-specific seed loaders) to accept a KG in memory or from file.

**Rationale**: Contextual_information.md describes “improved KG fed back as augmented seed for further refinement”; spec FR-006 and SC-006 require at least one documented path.

**Alternatives considered**:
- **Training-only seed re-use**: Accepted as one path; extraction-time “context KG” could be a future extension; MVP is training or a single documented re-use path.
- **No seed re-use**: Rejected; spec requires it.

---

## 7. Documentation mapping

**Decision**: **Documentation** must (1) describe the reliability narrative (provenance, FActScore, ValidityScore, KG cleaning, ontology, human-in-the-loop) in user- and agent-facing docs, and (2) map Contextual_information.md to project capabilities and gaps (e.g., in `reports/REFERENCE_SOURCES_AND_ENCODER.md` or equivalent). Update PROJECT_STATUS and AGENTS.md as needed so that “reliability pipeline” and “encoder in extraction” are clearly stated.

**Rationale**: Spec FR-007 and SC-007; ensures “reflect the interesting description” is satisfied and guides contributors.

**Alternatives considered**: None; requirement is explicit in spec.
