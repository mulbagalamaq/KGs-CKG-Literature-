
Goal: Implement a system to use GNNs + an LLM over knowledge graphs to integreate literature with knowledge graphs made from large public datasets to turn a free-text biomedical question into a grounded answer with summarized biomedical information and PMIDs / experiment IDs. 

<img width="4624" height="2838" alt="image" src="https://github.com/user-attachments/assets/9f274f12-ac4f-48e0-81c9-174de8731681" />

# WORK IN PROGRESS
Question → embed → OpenSearch top‑k seeds (across both CKG and PKG vectors).
Seeds → Neptune openCypher expansion (1–2 hops, label filters, degree caps).
Expanded graph → PCST-like pruning → compact, evidence-rich subgraph.
Subgraph + snippets → LLM → grounded answer with PMIDs/experiment IDs.

[User Q]
   │
   ▼
[Embedder] ──► [OpenSearch] ──► seeds
                               │
                               ▼
                        [Neptune (openCypher)]
                               │ expanded 1–2 hops
                               ▼
                       [PCST Pruner (GNN-aware)]
                               │ compact subgraph + snippets
                               ▼
                           [LLM Answerer]
                               │
                               ▼
                        grounded answer + citations


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

📦 Repository layout
├── api/                    # FastAPI service (REST)
│   ├── main.py
│   ├── routers/
│   └── schemas.py
├── pipelines/
│   ├── embed_and_seed.py   # question → embeddings → OpenSearch seed search
│   ├── expand_graph.py     # Neptune openCypher expansion
│   ├── prune_pcst.py       # PCST-like Steiner pruning
│   ├── ground_and_answer.py# snippets + subgraph → LLM
│   └── evaluate.py
├── gnn/
│   ├── models.py           # GNN encoder/edge scorer
│   └── train.py
├── prompts/
│   └── grounded_answer.md  # system & few-shot templates
├── ops/
│   ├── docker-compose.yml
│   ├── Dockerfile.api
│   ├── Dockerfile.worker
│   └── terraform/          # optional IaC for OpenSearch/Neptune
├── scripts/
│   ├── index_opensearch.py # create vector indices & ingest docs
│   ├── neptune_load.py     # bulk load nodes/edges to Neptune
│   └── demo_query.sh
├── config/
│   ├── default.yaml
│   └── labels.schema.json
├── tests/
│   └── ...
├── README.md
└── LICENSE

