# RAG Research Sources

Last curated: 2026-04-14

This file collects primary or official sources for the RAG techniques that are
most relevant to this repository. The goal is not to be exhaustive. The goal is
to keep a clean, current shortlist of canonical papers and official lab docs.

Selection rules:

- Prefer official lab docs, ACL Anthology, OpenReview, ICLR/EMNLP/NAACL pages.
- Use arXiv only when no stable proceedings version exists.
- Avoid tertiary explainers, blogs, and derivative summaries.
- When a newer public record exists but is weaker than the canonical source
  (for example a withdrawn submission), both are listed and labeled clearly.

## Quick index

| Topic | Best current primary source found | Institution(s) | Date | Status | Why it matters here |
|---|---|---|---|---|---|
| Contextual Retrieval / "contextual chunking" | Anthropic engineering post + Anthropic cookbook | Anthropic | 2024-09-19 / 2024-09-13 | Official docs, not a paper | Closest match to the repo's contextual chunking approach |
| RAPTOR | ICLR 2024 paper | Stanford University | 2024-01-16 | Peer-reviewed | Hierarchical retrieval via recursive summaries |
| HyDE | ACL 2023 paper | Carnegie Mellon University + University of Waterloo | 2023-07 | Peer-reviewed | Hypothetical document generation before dense retrieval |
| CRAG | CoRR/arXiv 2024 preprint | Authors listed; stable affiliation not clearly exposed on the primary page | 2024-01-29 | Preprint | Retrieval quality estimation and corrective re-retrieval |
| ColBERTv2 | NAACL 2022 paper | Stanford University + Georgia Tech | 2022-07 | Peer-reviewed | Late interaction retrieval / pre-filter reranking |
| Self-RAG | ICLR 2024 paper | University of Washington + Allen Institute for AI + IBM Research AI | 2024-02-02 | Peer-reviewed | Adaptive retrieval plus self-critique during generation |
| RAG-Fusion | arXiv 2024 note | Zackary Rackauckas; Infineon context stated in abstract | 2024-01-31 / 2024-02-21 rev. | Informal arXiv publication | Multi-query retrieval plus Reciprocal Rank Fusion |
| Active RAG | EMNLP 2023 paper | CMU-led author list | 2023-10-07 | Peer-reviewed | Retrieval triggered repeatedly during generation |
| Reciprocal Rank Fusion | SIGIR 2009 short paper | University of Waterloo + Google | 2009-07-19 | Foundational paper | Core fusion method used in many modern hybrid RAG stacks |

## 1. Contextual Retrieval / "Contextual Chunking"

Important terminology note:

- In production RAG discussions, "contextual chunking" is often used loosely.
- Anthropic's official term is "Contextual Retrieval".
- The technique is more specific than chunk sizing alone: it combines
  chunk-specific context with embeddings and BM25.

Primary official sources:

1. Anthropic, "Introducing Contextual Retrieval", published 2024-09-19.
   https://www.anthropic.com/engineering/contextual-retrieval
2. Anthropic Claude Cookbook, "Enhancing RAG with contextual retrieval",
   published 2024-09-13.
   https://platform.claude.com/cookbook/capabilities-contextual-embeddings-guide

Why this source matters:

- Anthropic frames the method as two coupled ideas:
  - Contextual Embeddings
  - Contextual BM25
- The engineering post reports large retrieval gains and is still the clearest
  official source for the method.
- The cookbook is the best implementation-oriented companion document.

Key takeaways:

- Add short chunk-specific context before embedding.
- Reuse the same context for BM25 / lexical indexing, not only dense search.
- Anthropic explicitly recommends stacking this with reranking.

Repo relevance:

- This is the closest primary source to the current repo's "context prefix" and
  parent/child retrieval design.
- If this repo is simplified, this is the first family of sources to preserve.

## 2. RAPTOR

Canonical source:

1. Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie,
   Christopher D. Manning, "RAPTOR: Recursive Abstractive Processing for
   Tree-Organized Retrieval", ICLR 2024.
   OpenReview page: https://openreview.net/forum?id=GN921JHCRw
   PDF: https://openreview.net/pdf?id=GN921JHCRw

Institution:

- Stanford University

Exact public dates:

- OpenReview published: 2024-01-16
- Last modified on the OpenReview record: 2024-04-14

Why it matters:

- RAPTOR is the canonical hierarchical-summary retrieval paper.
- The core idea is recursive clustering, summarization, and retrieval across
  multiple abstraction levels.

Important implementation note:

- Many "RAPTOR-like" systems only add section or document summaries.
- That is not equivalent to the full paper's recursive tree construction.

Repo relevance:

- If the repo keeps RAPTOR, this is the paper to measure fidelity against.
- If the repo only keeps summary chunks, it should be documented as
  "RAPTOR-inspired", not as full RAPTOR.

## 3. HyDE

Canonical source:

1. Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan,
   "Precise Zero-Shot Dense Retrieval without Relevance Labels", ACL 2023.
   ACL Anthology: https://aclanthology.org/2023.acl-long.99/
   PDF: https://aclanthology.org/2023.acl-long.99.pdf

Institutions:

- Language Technologies Institute, Carnegie Mellon University
- David R. Cheriton School of Computer Science, University of Waterloo

Exact public date:

- ACL 2023 proceedings month: 2023-07

Why it matters:

- HyDE is still the canonical primary source for hypothetical document
  embeddings.
- The method generates a hypothetical answer-like document first and retrieves
  against that dense representation instead of relying only on the raw query.

Repo relevance:

- This directly matches the repo's optional HyDE query branch.
- As of 2026-04-14, I did not find a newer canonical paper that supersedes HyDE
  itself in the same role.

## 4. CRAG

Best current primary sources found:

1. Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling,
   "Corrective Retrieval Augmented Generation", CoRR / arXiv 2024.
   OpenReview CoRR record: https://openreview.net/forum?id=FL7roDGPpP
   arXiv DOI record: https://doi.org/10.48550/arXiv.2401.15884
2. Later public venue record:
   OpenReview ICLR 2025 withdrawn submission:
   https://openreview.net/forum?id=JnWJbrnaUE

Exact public dates:

- arXiv submission date found on the public record: 2024-01-29
- CoRR/OpenReview record last modified: 2024-09-30
- Later ICLR 2025 submission record published: 2024-09-26
- Later ICLR 2025 submission record modified: 2024-11-26

Status note:

- The 2024 source is still a preprint / CoRR-style record.
- I did not find an accepted archival conference version newer than that.
- The later ICLR 2025 record is explicitly marked as withdrawn.

Why it matters:

- CRAG introduces a retrieval evaluator, corrective branching, and refinement of
  retrieved evidence before final generation.

Repo relevance:

- This maps closely to retrieval-confidence checks and retry / reformulation
  logic.
- If the repo keeps "CRAG", use the preprint as the canonical source and treat
  the withdrawn 2025 submission only as a later public record, not as stronger
  evidence.

## 5. ColBERT / ColBERTv2

Best canonical source for modern usage:

1. Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts,
   Matei Zaharia, "ColBERTv2: Effective and Efficient Retrieval via Lightweight
   Late Interaction", NAACL 2022.
   ACL Anthology: https://aclanthology.org/2022.naacl-main.272/
   PDF: https://aclanthology.org/2022.naacl-main.272.pdf

Institutions:

- Stanford University
- Georgia Institute of Technology

Exact public date:

- NAACL proceedings month: 2022-07

Why it matters:

- ColBERTv2 is the practical late-interaction source most teams mean today when
  they say "ColBERT" in production retrieval stacks.
- It improves quality while sharply reducing the storage cost of late
  interaction.

Repo relevance:

- The repo uses `colbert-ir/colbertv2.0`.
- This is the correct paper to cite for that model family.

## 6. Self-RAG

Canonical source:

1. Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi,
   "Self-RAG: Learning to Retrieve, Generate, and Critique through
   Self-Reflection", ICLR 2024.
   OpenReview page: https://openreview.net/forum?id=hSyW5go0v8
   PDF: https://openreview.net/pdf?id=hSyW5go0v8

Institutions:

- University of Washington
- Allen Institute for AI
- IBM Research AI

Exact public dates:

- OpenReview published: 2024-02-02
- Last modified on the OpenReview record: 2024-03-11

Why it matters:

- Self-RAG is one of the clearest primary sources for retrieval on demand plus
  self-critique signals inside the generation process itself.

Repo relevance:

- This is the main paper to consult if the repo ever evolves from pipeline-level
  retrieval heuristics toward model-level adaptive retrieval and critique.

## 7. RAG-Fusion

Best public source found:

1. Zackary Rackauckas, "RAG-Fusion: a New Take on Retrieval-Augmented
   Generation", arXiv:2402.03367.
   arXiv abstract page: https://arxiv.org/abs/2402.03367
   DOI: https://doi.org/10.48550/arXiv.2402.03367

Exact public dates:

- Submitted on arXiv: 2024-01-31
- Revised on arXiv: 2024-02-21
- Journal reference listed on arXiv:
  International Journal on Natural Language Computing, Vol. 13, No. 1,
  February 2024

Status note:

- This is not a flagship lab paper in the same class as RAPTOR, HyDE,
  ColBERTv2, or Self-RAG.
- DBLP classifies the arXiv record as "Informal or Other Publication".

Why it matters:

- It is still useful as a practical reference for multi-query generation plus
  Reciprocal Rank Fusion.

Repo relevance:

- If the repo adds multi-query retrieval plus fusion, this is the direct named
  reference.
- It should be cited with a maturity warning.

## 8. Active RAG

Canonical source:

1. Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu,
   Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, Graham Neubig,
   "Active Retrieval Augmented Generation", EMNLP 2023.
   OpenReview page: https://openreview.net/forum?id=WLZX3et7VT&noteId=MC4TUfGjJr

Exact public dates:

- OpenReview published: 2023-10-07
- Last modified on the OpenReview record: 2023-12-01

Why it matters:

- Active RAG is a useful adjacent source when comparing static one-shot
  retrieval against repeated retrieval during long-form generation.

Repo relevance:

- Not directly implemented in the current repo, but conceptually adjacent to
  HyDE, CRAG, and future adaptive retrieval work.

## 9. Reciprocal Rank Fusion

Foundational source:

1. Gordon V. Cormack, Charles L. A. Clarke, Stefan Buettcher,
   "Reciprocal rank fusion outperforms condorcet and individual rank learning
   methods", SIGIR 2009.
   DOI: https://dl.acm.org/doi/10.1145/1571941.1572114
   DBLP: https://dblp.org/rec/conf/sigir/CormackCB09.html

Institutions:

- University of Waterloo
- Google

Exact public date:

- SIGIR 2009 conference date / DOI-linked record: 2009-07-19

Why it matters:

- RRF is the core fusion method behind many modern dense+sparse or multi-query
  RAG systems.
- RAG-Fusion depends on this foundation.

Repo relevance:

- This repo already uses RRF for hybrid retrieval.

## Practical reading order for this repo

If the goal is to keep the reading list minimal and directly useful for this
codebase, read in this order:

1. Anthropic Contextual Retrieval docs
2. HyDE
3. ColBERTv2
4. RAPTOR
5. CRAG
6. Self-RAG
7. RAG-Fusion
8. Active RAG
9. RRF

## Notes on recency

As of 2026-04-14, I did not find newer canonical papers that clearly supersede
the original core references for RAPTOR, HyDE, ColBERTv2, or Self-RAG.

The main exceptions are:

- Contextual Retrieval, where the best sources are official Anthropic docs
  rather than a formal conference paper.
- CRAG, where the strongest public source is still the 2024 preprint lineage.
- RAG-Fusion, which is useful in practice but should be treated as a lower
  maturity source than the peer-reviewed papers above.
