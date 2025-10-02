
**Goal:** Implement a system to use GNNs + LLM over knowledge graphs to integrate literature with knowledge graphs made from large public datasets to turn a free-text biomedical question into a grounded answer with biomedical information and PMIDs / experiment IDs. 

## ✨ TL;DR (Summary)

- **Question → Embedding → Retrieval**: Encode the user’s question and fetch top-k seeds from Clinical + Public KGs.
- **Graph Expansion**: Expand seeds in Neptune with openCypher (1–2 hops, label filters, degree caps).
- **Pruning**: Apply a PCST-like algorithm to create a compact, evidence-rich subgraph.
- **Answering**: Feed the subgraph + snippets into an LLM to generate a grounded answer with biomedical information and PMIDs/experiment IDs.

<img width="4624" height="2838" alt="image" src="https://github.com/user-attachments/assets/c978497e-25b2-44d3-980a-0ca070720078" />

## Methods


1. **Data Preparation**  
   - Collect PrimeKG and PubMedKG exports.  
   - Normalize file naming (PrimeKG edges/nodes, PubMedKG publications/mentions/citations).  
   - Convert both datasets into Neptune-style CSVs with namespace labels (`PRIME_*`, `PKG_*`).

2. **Graph Loading**  
   - Load the converted CSVs into Neo4j (or Neptune) while preserving all properties.  
   - Create uniqueness constraints on `~id` for each namespace.

3. **Embedding Pipeline**  
   - Generate document embeddings for PrimeKG nodes and PubMedKG publications.  
   - Generate node embeddings that blend structural and textual context.  
   - Store embeddings in OpenSearch for vector similarity search.

4. **Retrieval and Expansion**  
   - Embed the user question and run KNN over OpenSearch to get seed entities.  
   - Expand the graph around those seeds (1–2 hops, namespace + degree limits).  
   - Prune the subgraph to maintain a manageable size.

5. **PyG Fact Fusion**  
   - Build a PyTorch Geometric graph from the pruned subgraph.  
   - Fuse graph embeddings with textual embeddings to score facts.  
   - Keep the top-ranked facts for prompting.

6. **LLM Answering**  
   - Assemble a prompt with selected facts and citations.  
   - Call the LLM to produce a concise answer anchored in the retrieved evidence.
