## Overview
We study question answering over a heterogeneous biomedical knowledge graph by coupling graph-native retrieval with neural reasoning. Given a question $Q_i$, we seek an answer set $A_i \subseteq V$ from a large graph $G=(V,E)$. Our pipeline formalizes subgraph retrieval as a Prize-Collecting Steiner Tree (PCST) optimization, then conditions a GNN+LLM reader on the retrieved subgraph. This design reduces hallucinations by grounding in the graph while scaling beyond a single-context window via structured retrieval, following recent GraphRAG advances inspired by G-Retriever (NVIDIA technical blog) and the original formulation in the G-Retriever paper.

**Goal:** Implement a system to use GNNs + LLM to integrate knowledge graphs made from large public datasets with literature to turn a free-text biomedical question into a grounded answer with biomedical information and PMIDs / experiment IDs. 

## ✨ TL;DR 
- **Question → Embedding → Retrieval**: Encode the user’s question and fetch top-k seeds from Clinical + Public KGs.
- **Graph Expansion**: Expand seeds in Neptune with openCypher (1–2 hops, label filters, degree caps).
- **Pruning**: Apply a PCST-like algorithm to create a compact, evidence-rich subgraph.
- **Answering**: Feed the subgraph + snippets into an LLM to generate a grounded answer with biomedical information and PMIDs/experiment IDs.

<img width="4997" height="2808" alt="image" src="https://github.com/user-attachments/assets/c583b661-5391-464c-ba31-b47afb9ece95" />

## Methods
### Knowledge graph and indexing
We construct a heterogeneous biomedical KG with node labels (for example, Drug, Disease, GeneOrProtein) and rich textual attributes. Data are validated against schema constraints at ingest (unique identifiers by label; referential integrity). Text fields are embedded using a sentence embedding model and materialized as a node property. We build:
- a vector index (cosine similarity) over node embeddings for semantic seeding, and
- graph-native indices (labels/properties) for efficient neighborhood expansion.
For production, the graph is persisted in Amazon Neptune with artifacts stored in Amazon S3; Neo4j can be used for local workflows. Configuration is managed in `configs/default.yaml`, and graph loaders reside in `src/ingest/`.
### Retrieval and base subgraph construction
For each question $Q_i$, we compute its embedding $\mathbf{q}_i$ and retrieve top-$k$ seed nodes $S_i \subset V$ via vector similarity:
$$
S_i = \operatorname*{arg\,topk}_{v \in V}\ \cos(\mathbf{q}_i,\mathbf{x}_v).
$$
We expand the 1–h neighborhood (optionally typed/filtered) around $S_i$ to induce a base subgraph $G_i=(V_i,E_i)$. This balances recall and tractability in dense biomedical regions.
### Prize assignment and PCST pruning
We assign node and edge prizes based on semantic alignment to the question and structural roles:
$$
p(v) = \alpha\, \cos(\mathbf{q}_i,\mathbf{x}_v) + \beta\, \mathbb{I}[v \in S_i],\quad
p(e) = \gamma\, w(e),
$$
where $w(e)$ captures relation salience (for example, curated edge types), and $\alpha,\beta,\gamma \ge 0$. Let $c(e)\!>\!0$ be traversal costs (e.g., uniform or type-specific). We solve a PCST variant on $G_i$ to produce a compact, connected pruned subgraph $G_i^*=(V_i^*,E_i^*)$:
$$
G_i^* = \operatorname*{arg\,max}_{T \subseteq G_i\ \text{connected}}
\Bigg[\sum_{v \in V(T)} p(v) + \sum_{e \in E(T)} p(e) - \lambda \sum_{e \in E(T)} c(e)\Bigg],
$$
with sparsity controlled by $\lambda \ge 0$. This concentrates question-relevant evidence while limiting neighborhood explosion, consistent with GraphRAG/G-Retriever.
### Neural reader: GNN + LLM integration
We encode $G_i^*$ with a GNN (e.g., GATv1 in PyTorch Geometric), producing node representations $\mathbf{h}_v$ that capture multi-hop structure and textual priors. We serialize $G_i^*$ into a compact textual description (ordered nodes with names/descriptions plus typed edges) and condition an instruction-tuned LLM on:
- question text,
- a soft prompt derived from the GNN outputs (pooled or node-wise), and
- the serialized subgraph context.
Following G-Retriever, we freeze the LLM and adapt via soft prompting over GNN outputs to preserve pretrained language capabilities while aligning attention to graph evidence.
### Training objectives
Supervision is provided as triplets $\{(Q_i,A_i,G_i)\}$ where $A_i \subseteq V$ denotes answer nodes. We optimize a joint objective:
$$
\mathcal{L} = \underbrace{\mathcal{L}_{\text{node}}}_{\text{answer identification}} + \eta \, \underbrace{\mathcal{L}_{\text{gen}}}_{\text{grounded generation}},
$$
where
- $\mathcal{L}_{\text{node}}$ is a multi-label node classification or listwise ranking loss over $V_i^*$ using scores $s_v = \mathrm{MLP}(\mathbf{h}_v)$, e.g.,
$$
\mathcal{L}_{\text{node}} = - \sum_{v \in V_i^*} \big[y_v \log \sigma(s_v) + (1-y_v)\log (1-\sigma(s_v))\big],
$$
with $y_v = \mathbb{I}[v \in A_i]$;
- $\mathcal{L}_{\text{gen}}$ is the token-level negative log-likelihood for answer text conditioned on $(Q_i, \text{serialization}(G_i^*), \text{soft-prompt}(\{\mathbf{h}_v\}))$.
At inference, we return top-$k$ node predictions and the generated answer; optionally, we ensemble by appending unique high-prize nodes from $G_i^*$ to improve recall@K, as observed in pipeline variants.
### System implementation
Retrieval and pruning are implemented in `src/retrieval/g_retriever.py` and `src/retrieval/expand.py`; the neural reader and reranking are implemented in `src/gnn/pyg_rag.py` and orchestrated by `src/rag/pipeline.py`. Graph storage uses Amazon Neptune with S3-backed artifacts in production, with strict schema validation and uniqueness constraints at ingest. Configuration is managed via `configs/default.yaml`, and loaders for Neptune/Neo4j are in `src/ingest/`.
### Reproducibility and evaluation
We fix seeds for deterministic runs, log all hyperparameters, and report Hits@K, Recall@K, and MRR on biomedical benchmarks with multi-hop questions. We ablate $k$, expansion radius, prize schedules, and $\lambda$ to quantify PCST sensitivity, aligning with prior observations on hyperparameter coupling in GraphRAG.
### References
- NVIDIA Technical Blog: Boosting Q&A Accuracy with GraphRAG Using PyG and Graph Databases — see <https://developer.nvidia.com/blog/boosting-qa-accuracy-with-graphrag-using-pyg-and-graph-databases/>
- G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering (arXiv:2402.07630) — see <https://arxiv.org/pdf/2402.07630>
