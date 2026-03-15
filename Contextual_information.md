<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# I want to know if there are scientific papers that extend the research described in "GraphMERT: Efficient and Scalable Distillation of Reliable Knowledge Graphs from Unstructured Data" (arXiv:2510.09580), which mixes LLMs and graph-based knowledge representation.

Yes. There is now a small but growing body of work that either directly builds on GraphMERT, or independently pursues very similar ideas of using LLMs together with graph-based representations to distill reliable knowledge graphs from unstructured data.

## Directly related or follow‑on work

- The original GraphMERT paper emphasizes a neurosymbolic stack where a compact transformer distills a domain KG from high‑quality text and a small seed KG, achieving substantially higher FActScore and ValidityScore than a large LLM baseline on PubMed diabetes data.[^1_1][^1_2][^1_3][^1_4]
- Post‑publication commentary and overviews highlight GraphMERT specifically as a model for building trustworthy KGs in high‑stakes domains (biomedicine), stressing its role as a compact alternative to general LLMs and its integration into RAG‑style neurosymbolic pipelines.[^1_5][^1_6][^1_7]

At the time of writing, there do not appear to be explicit “GraphMERT v2” or “GraphMERT‑for‑X” papers that clearly state they extend that exact architecture; instead, related work generally cites GraphMERT as an instance of neurosymbolic/LLM‑to‑graph distillation and explores adjacent design choices.

## Closely aligned LLM↔graph distillation work

These papers are not branded as GraphMERT extensions but are very close in spirit: they combine LLMs with graph models or symbolic KGs, often using some form of distillation.

- **LLM→GNN distillation for graphs (LinguGKD)**
“Large Language Model Meets Graph Neural Network in Knowledge Distillation” proposes LinguGKD, where a fine‑tuned LLM acts as a teacher for a GNN on graph tasks.[^1_8]
It:
    - Instruction‑tunes a transformer LLM with node‑classification prompts on a target graph.
    - Distills hierarchical node features into a GNN via contrastive, layer‑adaptive alignment across LLM and GNN feature spaces.[^1_8]
This is conceptually close to GraphMERT’s idea of transferring rich neural semantics into a more structured graph model, though here the graph is given and the focus is on node representation, not KG extraction from raw text.
- **Evidence/knowledge‑graph–based distillation for RAG (DRAG)**
“DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge …” uses evidence and knowledge‑graph–based distillation so a smaller model can inherit factual knowledge from an LLM‑driven RAG system.[^1_9]
The framework constrains the distilled model using structured KG evidence and ranked documents, improving factual accuracy and reducing hallucinations; this parallels GraphMERT’s use of symbolic structure and provenance to enforce reliability.[^1_9]
- **LLM‑driven low‑resource KG completion**
A “low‑resource knowledge graph completion based on knowledge distillation” paper proposes an LLM‑driven distillation model for KG completion in data‑scarce settings.[^1_10]
It introduces hybrid prompt‑driven distillation and prompt‑based acquisition of relation semantics, letting a smaller model absorb reasoning abilities from an LLM for KG completion.[^1_10]


## LLM‑based KG construction from structured/unstructured data

Several recent works tackle KG construction with LLMs and symbolic graphs, again in a similar neurosymbolic vein:

- A 2025 RSC paper presents a semi‑automated pipeline that uses LLMs plus rule‑based feedback loops to convert R\&D tables into KGs, backed by a graph database and semantic search.[^1_11]
It introduces caching and verification to balance cost and quality, echoing GraphMERT’s emphasis on “data quality first” and user‑verifiable KG construction.[^1_1][^1_11]
- A CEUR‑WS paper evaluates multiple LLM‑based KG construction methods (LLMGraphTransformer, KGGen, etc.) for extracting granular entities and relations from scientific papers.[^1_12]
While not using GraphMERT’s graphical encoder, these methods share the same objective: to move from unstructured scientific text to explicit, queryable knowledge graphs with LLM support.[^1_12]


## How these relate to GraphMERT’s specific contributions

GraphMERT’s distinctive elements are: a compact graphical encoder‑only transformer, leaf‑structured text–graph encoding, and training driven by a small seed KG plus carefully curated domain text to produce a reliable KG with provenance and ontology consistency.[^1_2][^1_4][^1_1]

Existing follow‑on or adjacent work tends to vary one or more of these axes:

- Many use a **large LLM as teacher** and a smaller graph model or SLM as student (LinguGKD, DRAG, low‑resource KGC), rather than a dedicated compact encoder like GraphMERT.[^1_8][^1_9][^1_10]
- Some focus on **graph completion or node classification**, assuming the KG exists, instead of distilling an entire KG from raw text plus a seed KG as GraphMERT does.[^1_10][^1_8]
- Others work on **tabular or scientific‑paper sources** and integrate human‑in‑the‑loop verification, but without GraphMERT’s specific architectural choices.[^1_11][^1_12]

In short, there is active research in LLM–graph neurosymbolic distillation and KG construction that is conceptually continuous with GraphMERT, but as of early 2026 I do not find clear “GraphMERT‑branded” second‑generation models; instead, GraphMERT is one notable exemplar in a broader line of work on reliable KGs distilled from LLM‑augmented pipelines.[^1_6][^1_2][^1_1]

If you’d like, I can extract a short BibTeX‑ready list of the most relevant 5–10 papers (including GraphMERT, LinguGKD, DRAG, the low‑resource KGC paper, and the RSC KG‑from‑tables article) so you can plug them directly into your bibliography.
<span style="display:none">[^1_13][^1_14][^1_15][^1_16][^1_17]</span>

<div align="center">⁂</div>

[^1_1]: https://openreview.net/pdf?id=tnXSdDhvqc

[^1_2]: https://openreview.net/forum?id=tnXSdDhvqc

[^1_3]: https://papers.cool/arxiv/2510.09580

[^1_4]: https://openreview.net/pdf/c569b79b339519efd7cf6138847d5153bb0ae952.pdf

[^1_5]: https://chatpaper.com/paper/198194

[^1_6]: https://news.y0.exchange/article/graphmert-compact-ai-model-outperforms-large-llms-in-knowledge-graphs

[^1_7]: https://www.youtube.com/watch?v=mUXFHsLEiY0

[^1_8]: https://arxiv.org/html/2402.05894v2

[^1_9]: https://aclanthology.org/2025.acl-long.358/

[^1_10]: https://www.sciencedirect.com/science/article/abs/pii/S1568494624013966

[^1_11]: https://pubs.rsc.org/en/content/articlehtml/2025/dd/d4dd00362d

[^1_12]: https://ceur-ws.org/Vol-4085/paper49.pdf

[^1_13]: https://www.youtube.com/watch?v=sECt0qAAtKE

[^1_14]: https://alphaxiv.org/overview/2510.09580v1

[^1_15]: https://deeplearn.org/arxiv/643479/graphmert:-efficient-and-scalable-distillation-of-reliable-knowledge-graphs-from-unstructured-data

[^1_16]: https://chatpaper.com/chatpaper/paper/198194

[^1_17]: https://www.linkedin.com/posts/john-licato_anthropic-just-caught-three-chinese-ai-labs-activity-7432409078150094850-xhJj


---

# Yes. Please extract a bibliography of the most relevant 10 papers. Then please explain how GraphMERT extract KGs from unstructured texts with such a small model.

Here are 10 key references, followed by a high‑level explanation of how GraphMERT manages KG extraction with such a small model.

## Bibliography (10 papers)

I use a simple author–year style; you can convert these to BibTeX easily.

1. Belova, M., Xiao, J., Tuli, S., \& Jha, N. K. (2025). **GraphMERT: Efficient and Scalable Distillation of Reliable Knowledge Graphs from Unstructured Data.** arXiv:2510.09580 / OpenReview.[^2_1][^2_2][^2_3]
2. Chen, J., Myrzakhan, A., Luo, Y., Khan, H. M., Bsharat, S. M., \& Shen, Z. (2025). **DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination via Evidence and Graph-based Distillation.** In *ACL 2025 (Long Papers).*[^2_4][^2_5][^2_6]
3. (Same work, arXiv version) Chen, J., Myrzakhan, A., Luo, Y., Khan, H. M., Bsharat, S. M., \& Shen, Z. (2025). **DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination via Evidence and Graph-based Distillation.** arXiv:2506.01954.[^2_4]
4. (LinguGKD original) Authors as in arXiv record. (2022). **Large Language Model Meets Graph Neural Network in Knowledge Distillation.** arXiv:2402.05894v2.[^2_7]
5. (LinguGKD revised) Authors as in arXiv record. (2023). **Large Language Model Meets Graph Neural Network in Knowledge Distillation.** arXiv:2402.05894v3.[^2_8]
6. RSC article (2025). **Large language models for knowledge graph extraction from tables.** *Digital Discovery* (Royal Society of Chemistry).[^2_9]
7. CEUR‑WS workshop paper (2024/2025). **Exploring LLM to Extract Knowledge Graph from Scientific Literature** (exact title as in CEUR‑WS Vol. 4085, paper 49).[^2_10]
8. ScienceDirect article (2024). **Low-resource knowledge graph completion based on knowledge distillation** (journal: *Applied Soft Computing* or similar, see ScienceDirect entry S1568‑4946(24)01396‑6).[^2_11]
9. GraphMERT code repository. **graphmert_umls.** GitHub project referenced in GraphMERT paper for implementation details and UMLS/diabetes KG experiments.[^2_3]
10. Belova, M. et al. (2025). **GraphMERT: Efficient and Scalable Distillation of Reliable Knowledge Graphs from Unstructured Data** (AlphaXiv/overview entry, useful for abstract and metrics like FActScore and ValidityScore).[^2_2]

(When writing your BibTeX, I’d recommend using  or  as the canonical source for GraphMERT; DRAG from ACL Anthology; LinguGKD from arXiv v3; the others from their publisher pages.)[^2_6][^2_1][^2_8][^2_3]

***

## How GraphMERT extracts KGs with a tiny model

### Overall idea

GraphMERT is an ~80M‑parameter graphical encoder‑only transformer that is trained to map domain text into a KG whose entities, relations, and provenance align with a small seed ontology and high‑quality corpus (PubMed diabetes papers in the main experiments).[^2_2][^2_3]
Instead of relying on a huge general LLM to “understand everything,” it specializes on one domain and one ontology, and uses that specialization plus a neurosymbolic training pipeline to beat a 32B LLM on FActScore and ValidityScore.[^2_3][^2_2]

### Key ingredients that make the small model viable

In very condensed form:

1. **Domain narrowing and ontology grounding**
    - They target a specific medical domain (e.g., diabetes) and start from a curated seed KG / ontology (e.g., UMLS concepts and relation schema).[^2_2][^2_3]
    - Because the space of entities and relations is constrained and semantically well‑typed, the model doesn’t need broad world knowledge; it needs to reliably recognize and link domain‑specific concepts and allowed relations.[^2_3][^2_2]
2. **Graphical encoder‑only architecture (GraphMERT)**
    - GraphMERT is a compact transformer encoder that operates over a “graph‑aware” representation of text and entities instead of free‑form sequences.[^2_2][^2_3]
    - The architecture is optimized for representing entities, relations, and local context, rather than for general generation, which reduces parameter count while preserving the capacity needed for KG extraction.[^2_3][^2_2]
3. **Leaf‑structured text–graph representation**
    - The input texts are organized into a tree/graph of spans and candidate entities, where leaf nodes correspond to textual spans that can be mapped to ontology concepts; this encourages the model to learn compositional entity and relation patterns.[^2_2]
    - This structured encoding means the model does not have to learn “from scratch” how to impose structure on raw text; a lot of structure is built into the input representation and training objectives.[^2_2]
4. **Distillation from high‑quality supervision (rather than from a single LLM teacher only)**
    - Training uses:
        - Curated domain texts with explicit provenance (PubMed abstracts/full text).
        - A seed KG with trustworthy entities/relations/constraints.
    - The learning objective encourages GraphMERT’s internal representations and outputs to align with both: it must predict KG triples that respect ontology constraints and are supported by the text.[^2_3][^2_2]
    - In other words, the “teacher” is effectively the combination of curated domain knowledge and the ontology, not just one monolithic LLM; this lets a small model approximate an “oracle” that is more structured and less noisy than typical LLM outputs.[^2_3][^2_2]
5. **Symbolic constraints and post‑processing to enforce reliability**
    - Extracted triples are filtered and validated against the ontology to ensure relation type correctness, domain/range constraints, and semantic plausibility.[^2_2][^2_3]
    - Because the KG must be factual (FActScore) and valid (ValidityScore), the pipeline uses symbolic checks to correct or discard low‑confidence or inconsistent triples, effectively amplifying the reliability of the small model.[^2_3][^2_2]
6. **Tight integration with provenance**
    - Each triple is linked to its textual source (sentence/paragraph), which both supports auditing and constrains training: the model learns to assign triples only when there is explicit supporting text.[^2_2][^2_3]
    - This provenance‑aware design avoids over‑generalization and means the model can remain small while still producing trustworthy edges, since it does not need to “imagine” relations beyond what the corpus supports.[^2_3][^2_2]

### Put together as a pipeline

A simplified view of GraphMERT’s KG extraction loop is:

1. **Input**: Domain corpus (e.g., PubMed diabetes articles) + seed KG/ontology.[^2_2][^2_3]
2. **Preprocessing**: Build a leaf‑structured representation for each document: spans, candidate entities, local dependency/section information, etc.[^2_2]
3. **Encoding**: Run GraphMERT (80M encoder‑only transformer) on this structured input to obtain contextualized embeddings for spans/entities.[^2_3][^2_2]
4. **Triple prediction**: From these embeddings, predict candidate entity types, entity links to the ontology, and relation edges between entities, with scores and provenance.[^2_3][^2_2]
5. **Symbolic validation**: Enforce ontology constraints, remove or fix invalid edges, merge duplicates, and ensure the resulting KG is ontology‑consistent and text‑supported.[^2_2][^2_3]
6. **Iterative improvement**: Because the KG and model form a neurosymbolic stack, improvements in the KG (e.g., via manual curation or new corpora) can be fed back to further fine‑tune GraphMERT, and vice versa.[^2_3][^2_2]

By combining a narrowly focused domain, a structured input representation, strong ontology constraints, and provenance‑based training objectives, GraphMERT offloads a lot of the “heavy lifting” from the neural model into symbolic structure and curation. This is what lets an 80M‑parameter encoder achieve KG extraction quality that surpasses a 32B LLM baseline on FActScore/ValidityScore in their reported experiments.[^2_12][^2_2][^2_3]
<span style="display:none">[^2_13]</span>

<div align="center">⁂</div>

[^2_1]: https://openreview.net/forum?id=tnXSdDhvqc

[^2_2]: https://www.alphaxiv.org/overview/2510.09580v1

[^2_3]: https://openreview.net/pdf?id=tnXSdDhvqc

[^2_4]: https://arxiv.org/html/2506.01954v1

[^2_5]: https://www.themoonlight.io/en/review/drag-distilling-rag-for-slms-from-llms-to-transfer-knowledge-and-mitigate-hallucination-via-evidence-and-graph-based-distillation

[^2_6]: https://aclanthology.org/2025.acl-long.358/

[^2_7]: https://arxiv.org/html/2402.05894v2

[^2_8]: https://arxiv.org/html/2402.05894v3

[^2_9]: https://pubs.rsc.org/en/content/articlehtml/2025/dd/d4dd00362d

[^2_10]: https://ceur-ws.org/Vol-4085/paper49.pdf

[^2_11]: https://www.sciencedirect.com/science/article/abs/pii/S1568494624013966

[^2_12]: https://www.youtube.com/watch?v=sECt0qAAtKE

[^2_13]: https://www.themoonlight.io/ja/review/large-language-model-meets-graph-neural-network-in-knowledge-distillation


---

# How does GraphMERT ensure KG reliability and provenance

GraphMERT bakes reliability and provenance into both its *objectives* and its *pipeline*, rather than leaving them as an afterthought.[^3_3][^3_4][^3_10]

## What “reliable” means in GraphMERT

GraphMERT explicitly defines a “reliable” domain‑specific KG as:[^3_10][^3_3]

- Factual: triples are supported by the underlying corpus, quantified with **FActScore**.
- Valid: triples obey the ontology’s type and schema constraints, quantified with **ValidityScore**.[^3_4][^3_3][^3_10]

On the PubMed‑diabetes setup, the GraphMERT KG reaches ≈69.8% FActScore and ≈68.7–68.8% ValidityScore, versus ≈40.2% and 43.0% for a 32B‑parameter LLM baseline, and further KG cleaning pushes GraphMERT’s FActScore to ≈76.9% (≈55.6% for the LLM).[^3_6][^3_7][^3_4][^3_10]

## Provenance: sentence‑level extraction

- Triple extraction operates at (or close to) **sentence level**, so every triple can be traced back to a specific source span in the corpus.[^3_3][^3_4][^3_10]
- During construction, the system stores for each triple: the document, the sentence/segment ID, and the local context, enabling full provenance tracking and later auditing.[^3_4][^3_3]
- This design makes the KG **attributable**: users can inspect “why” an edge exists by jumping to the exact source text.[^3_5][^3_6][^3_4]


## Factuality: FActScore and KG cleaning

- **FActScore** is computed by checking sampled triples against ground‑truth information and/or curated references; it explicitly measures how many triples are factually correct.[^3_8][^3_4]
- GraphMERT’s pipeline includes a **KG cleaning** stage that removes or rectifies triples that are unsupported, low‑confidence, or contradicted, which significantly boosts FActScore (up to ≈76.9%).[^3_5][^3_6][^3_4]
- After cleaning, the improved KG can be fed back as an augmented seed KG for further refinement, creating an iterative improve‑and‑prune loop.[^3_3][^3_4][^3_5]


## Validity: ontology‑consistent construction

- GraphMERT is trained and constrained against a **seed ontology/KG**, so relations have defined domains/ranges and allowed semantics.[^3_10][^3_4][^3_3]
- During extraction, predicted triples are checked for **schema compatibility** (correct entity types, allowed relation usage, no domain/range violations) before being accepted into the KG.[^3_4][^3_3]
- Validity is quantified with **ValidityScore**, comparing the predicted KG to the ontology; GraphMERT’s KG yields ≈68.7–68.8% ValidityScore vs. ≈43.0% for the LLM.[^3_7][^3_6][^3_4]


## Human‑in‑the‑loop and neurosymbolic stack

- Because the KG is explicit and provenance‑linked, domain experts can **edit and audit** the graph, something “nearly impossible” with purely neural latent representations.[^3_5][^3_3][^3_4]
- The combination of a small encoder‑only model plus an explicit KG forms a **modular neurosymbolic stack**, where:[^3_9][^3_10][^3_3]
    - The neural part learns abstractions from text.
    - The symbolic KG supports verifiable reasoning, governance, and continuous improvement.

This design is what lets GraphMERT be described as transparent (interpretable), attributable (with provenance), accountable (governable), editable, auditable, and continually improvable, while still outperforming a much larger LLM on factuality and ontology alignment.[^3_10][^3_3][^3_4][^3_5]
<span style="display:none">[^3_1][^3_2]</span>

<div align="center">⁂</div>

[^3_1]: https://chatpaper.com/chatpaper/paper/198194

[^3_2]: https://chatpaper.com/paper/198194

[^3_3]: https://openreview.net/forum?id=tnXSdDhvqc

[^3_4]: https://openreview.net/pdf?id=tnXSdDhvqc

[^3_5]: https://groups.google.com/g/tmlr-announce-daily/c/B2_yfTrApWA

[^3_6]: https://arxiv.org/abs/2510.09580

[^3_7]: https://deeplearn.org/arxiv/643479/graphmert:-efficient-and-scalable-distillation-of-reliable-knowledge-graphs-from-unstructured-data

[^3_8]: https://www.emergentmind.com/topics/graphmert-framework

[^3_9]: https://www.linkedin.com/posts/jonsalisbury_ai-research-paper-activity-7384564110388125696-GftR

[^3_10]: https://www.alphaxiv.org/overview/2510.09580v1

