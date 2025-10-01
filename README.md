# KGs_plus_Literature
Implement a system to use GNN and/or LLM to use knowledge graphs to integrate literature with knowledge graphs made from large public datasets (CIVIC, TCGA, CPTAC)

<img width="914" height="488" alt="image" src="https://github.com/user-attachments/assets/4f1bb8e9-783c-4c61-906b-e0fb60d7d065" />

# WORK IN PROGRESS
Question → embed → OpenSearch top‑k seeds (across both CKG and PKG vectors).
Seeds → Neptune openCypher expansion (1–2 hops, label filters, degree caps).
Expanded graph → PCST-like pruning → compact, evidence-rich subgraph.
Subgraph + snippets → LLM → grounded answer with PMIDs/experiment IDs.
