
**Goal:** Implement a system to use GNNs + an LLM over knowledge graphs to integreate literature with knowledge graphs made from large public datasets to turn a free-text biomedical question into a grounded answer with summarized biomedical information and PMIDs / experiment IDs. 

<img width="4624" height="2838" alt="image" src="https://github.com/user-attachments/assets/9f274f12-ac4f-48e0-81c9-174de8731681" />


## 🏗️ Architecture (high-level)

## 🏗️ Architecture (high-level)

```mermaid
flowchart TD
    UQ["User Q"] --> E[Embedder]
    E --> OS[OpenSearch]
    OS -->|seeds| N[Neptune (openCypher)]
    N -->|"expanded 1–2 hops"| P[PCST Pruner (GNN-aware)]
    P -->|"compact subgraph + snippets"| L[LLM Answerer]
    L -->|"grounded answer + citations"| A[Answer]
```



🧭 End-to-end flow

Embed: Encode the user question with a sentence embedding model (e.g., bge-large, text-embedding-3-large) and optionally a domain adapter.

Seed retrieval (OpenSearch): Query two vector indices:

ckg_vectors (curated/clinical KG nodes + doc chunks)

pkg_vectors (public datasets KG nodes + literature chunks)
Merge top-k seeds with score normalization.

KG expansion (Neptune openCypher): From seeds, expand 1–2 hops with label filters and degree caps to avoid hubs; fetch node/edge attributes and supporting citations.

PCST-like pruning: Run a Prize-Collecting Steiner Tree approximation on the expanded subgraph using:

prizes: seed proximity, evidence count, recency, doc quality

costs: edge weights / degree penalties
Output a compact, evidence-rich subgraph.

Grounded answer (LLM): Serialize the pruned subgraph + top evidence snippets into a grounded prompt. The LLM must cite PMIDs/experiment IDs and include a short evidence table.




