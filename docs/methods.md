### Methods

#### Overview
We study question answering over a heterogeneous biomedical knowledge graph by coupling graph-native retrieval with neural reasoning. Given a question $Q_i$, we seek an answer set $A_i \subseteq V$ from a large graph $G=(V,E)$. Our pipeline formalizes subgraph retrieval as a Prize-Collecting Steiner Tree (PCST) optimization, then conditions a GNN+LLM reader on the retrieved subgraph. This design reduces hallucinations by grounding in the graph while scaling beyond a single-context window via structured retrieval, following recent GraphRAG advances inspired by G-Retriever (NVIDIA technical blog) and the original formulation in the G-Retriever paper.

#### Knowledge graph and indexing
We construct a heterogeneous biomedical KG with node labels (for example, Drug, Disease, GeneOrProtein) and rich textual attributes. Data are validated against schema constraints at ingest (unique identifiers by label; referential integrity). Text fields are embedded using a sentence embedding model and materialized as a node property. We build:
- a vector index (cosine similarity) over node embeddings for semantic seeding, and
- graph-native indices (labels/properties) for efficient neighborhood expansion.

For production, the graph is persisted in Amazon Neptune with artifacts stored in Amazon S3; configuration is managed in `configs/default.yaml`, and graph loaders reside in `src/ingest/`.

#### Retrieval and base subgraph construction
For each question $Q_i$, we compute its embedding $\mathbf{q}_i$ and retrieve top-$k$ seed nodes $S_i \subset V$ via vector similarity:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=S_i%20=%20\operatorname{arg\,topk}_{v%20\in%20V}%20\cos(\mathbf{q}_i,\mathbf{x}_v)" alt="S_i expression" />
</p>
We expand the 1–h neighborhood (optionally typed/filtered) around $S_i$ in Amazon Neptune via an IAM-signed openCypher query that enforces namespace label prefixes and degree caps, inducing a base subgraph $G_i=(V_i,E_i)$ that balances recall and tractability in dense biomedical regions.

#### Prize assignment and PCST pruning
We assign node and edge prizes based on semantic alignment to the question and structural roles:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\begin{aligned}%20p(v)%20&=%20\alpha%20\cos(\mathbf{q}_i,\mathbf{x}_v)%20+%20\beta%20\mathbb{I}[v%20\in%20S_i]%20\\%20p(e)%20&=%20\gamma%20w(e)%20\end{aligned}" alt="Prize assignment" />
</p>
where $w(e)$ captures relation salience (for example, curated edge types), and $\alpha,\beta,\gamma \ge 0$. Let $c(e)\!>\!0$ be traversal costs (e.g., uniform or type-specific). We solve a PCST variant on $G_i$ to produce a compact, connected pruned subgraph $G_i^*=(V_i^*,E_i^*)$:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=G_i^*%20=%20\underset{T%20\subseteq%20G_i\,\text{connected}}{\operatorname{arg\,max}}%20\left[\sum_{v%20\in%20V(T)}%20p(v)%20+%20\sum_{e%20\in%20E(T)}%20p(e)%20-%20\lambda%20\sum_{e%20\in%20E(T)}%20c(e)\right]" alt="PCST objective" />
</p>
with sparsity controlled by $\lambda \ge 0$. This concentrates question-relevant evidence while limiting neighborhood explosion, consistent with GraphRAG/G-Retriever.

#### Neural reader: GNN + LLM integration
We encode $G_i^*$ with a GNN (e.g., GATv1 in PyTorch Geometric), producing node representations $\mathbf{h}_v$ that capture multi-hop structure and textual priors. We serialize $G_i^*$ into a compact textual description (ordered nodes with names/descriptions plus typed edges) and condition an instruction-tuned LLM on:
- question text,
- a soft prompt derived from the GNN outputs (pooled or node-wise), and
- the serialized subgraph context.

Following G-Retriever, we freeze the LLM and adapt via soft prompting over GNN outputs to preserve pretrained language capabilities while aligning attention to graph evidence.

#### Training objectives
Supervision is provided as triplets $\{(Q_i,A_i,G_i)\}$ where $A_i \subseteq V$ denotes answer nodes. We optimize a joint objective:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}%20=%20\underbrace{\mathcal{L}_{\text{node}}}_{\text{answer%20identification}}%20+%20\eta%20\underbrace{\mathcal{L}_{\text{gen}}}_{\text{grounded%20generation}}" alt="Joint loss" />
</p>
where
- $\mathcal{L}_{\text{node}}$ is a multi-label node classification or listwise ranking loss over $V_i^*$ using scores $s_v = \mathrm{MLP}(\mathbf{h}_v)$, e.g.,

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{\text{node}}%20=%20-%20\sum_{v%20\in%20V_i^*}%20\left[y_v%20\log%20\sigma(s_v)%20+%20(1-y_v)\log%20(1-\sigma(s_v))\right]" alt="Node loss" />
</p>
with $y_v = \mathbb{I}[v \in A_i]$;
- $\mathcal{L}_{\text{gen}}$ is the token-level negative log-likelihood for answer text conditioned on $(Q_i, \text{serialization}(G_i^*), \text{soft-prompt}(\{\mathbf{h}_v\}))$.

At inference, we return top-$k$ node predictions and the generated answer; optionally, we ensemble by appending unique high-prize nodes from $G_i^*$ to improve recall@K, as observed in pipeline variants.

#### System implementation
Retrieval and pruning are implemented in `src/retrieval/g_retriever.py` and `src/retrieval/expand_neptune.py`; the neural reader and reranking are implemented in `src/gnn/pyg_rag.py` and orchestrated by `src/rag/pipeline.py`. Graph storage uses Amazon Neptune with S3-backed artifacts in production, with strict schema validation and uniqueness constraints at ingest. Configuration is managed via `configs/default.yaml`, and loaders for Neptune reside in `src/ingest/`.

#### Reproducibility and evaluation
We fix seeds for deterministic runs, log all hyperparameters, and report Hits@K, Recall@K, and MRR on biomedical benchmarks with multi-hop questions. We ablate $k$, expansion radius, prize schedules, and $\lambda$ to quantify PCST sensitivity, aligning with prior observations on hyperparameter coupling in GraphRAG.

#### References
- NVIDIA Technical Blog: Boosting Q&A Accuracy with GraphRAG Using PyG and Graph Databases — see `https://developer.nvidia.com/blog/boosting-qa-accuracy-with-graphrag-using-pyg-and-graph-databases/`
- G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering (arXiv:2402.07630) — see `https://arxiv.org/pdf/2402.07630`



