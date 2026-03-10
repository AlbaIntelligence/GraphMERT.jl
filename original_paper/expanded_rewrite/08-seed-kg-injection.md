# 08: Seed KG Injection (Short)

Short specification of the **seed knowledge graph injection** pipeline. For full details, see `08-seed-kg-injection-FULL.md`.

---

## 1. Purpose

Seed KG injection enriches training data with **trusted triples** from an external KG (e.g. UMLS, Wikidata) by:

- Linking detected entities to KG nodes.
- Selecting high-quality triples.
- Injecting them into leafy chain graphs as leaves before training.

---

## 2. High-level stages

1. **Entity linking**
   - Map text entities to KG identifiers (CUI/QID/etc.).
   - Use domain-specific heuristics and/or neural linking.

2. **Candidate triple collection**
   - For each linked entity, fetch related triples from the KG.

3. **Scoring and selection**
   - Score triples based on relevance, confidence, diversity.
   - Bucket by score and relation type; select a limited number per entity.

4. **Injection into graphs**
   - Place selected triples into leaf slots of the leafy chain graph.
   - Respect capacity constraints and keep padding consistent.

Implementation references:

- `training/seed_injection.jl`
- `src/seed_injection.jl`
- Domain-specific helpers for UMLS/Wikidata.

---

## 3. Configuration

Controlled by `SeedInjectionConfig` in `src/types.jl`, including:

- Thresholds for entity linking and triple scores.
- Max candidates/triples per entity and per sequence.
- Injection ratio and bucket sizes.

---

## 4. Practical guidance

- Keep selection logic simple and deterministic at first; refine only when core training is stable.
- Use this file as a **conceptual overview**; detailed algorithms, edge cases, and examples should live in `08-seed-kg-injection-FULL.md`.

