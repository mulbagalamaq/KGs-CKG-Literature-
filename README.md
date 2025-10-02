
**Goal:** Implement a system to use GNNs + an LLM over knowledge graphs to integreate literature with knowledge graphs made from large public datasets to turn a free-text biomedical question into a grounded answer with biomedical information and PMIDs / experiment IDs. 

## ✨ TL;DR (Summary)

- **Question → Embedding → Retrieval**: Encode the user’s question and fetch top-k seeds from Clinical + Public KGs.
- **Graph Expansion**: Expand seeds in Neptune with openCypher (1–2 hops, label filters, degree caps).
- **Pruning**: Apply a PCST-like algorithm to create a compact, evidence-rich subgraph.
- **Answering**: Feed the subgraph + snippets into an LLM to generate a grounded answer with biomedical information and PMIDs/experiment IDs.

<img width="4624" height="2838" alt="image" src="https://github.com/user-attachments/assets/c978497e-25b2-44d3-980a-0ca070720078" />

## Methods
1. Ingest raw exports → we rely on CSVs/TSVs under data/local/primeKG/exports and data/local/pubmedkg/exports.
2. Convert to graph-ready CSVs → scripts produce data/graph/primekg/ and data/graph/pkg/ directories with nodes.csv + edges.csv.
3. Load into a graph database → either Amazon Neptune (bulk loader JSON). Both keep the original labels/properties.
4. Build embeddings + vector index → Amazon OpenSearch stores embeddings for KNN lookup.
5. Question answering → we embed the question, pull top vectors(K neighbours) from OpenSearch, expand the graph (Neptune) 1–2 hops (2 layers of GNN) with namespace filters, prune (using PCST algorithm), and feed the trimmed subgraph first to an GNN and then to an LLM for a cited answer.
