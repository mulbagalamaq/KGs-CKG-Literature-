
Implement a system to use GNN and LLM to use knowledge graphs to integrate literature with knowledge graphs made from large public datasets 

<img width="4624" height="2838" alt="image" src="https://github.com/user-attachments/assets/48a6981f-56c9-4bce-b735-c9a426fbf9e7" />


# WORK IN PROGRESS
Question → embed → OpenSearch top‑k seeds (across both CKG and PKG vectors).
Seeds → Neptune openCypher expansion (1–2 hops, label filters, degree caps).
Expanded graph → PCST-like pruning → compact, evidence-rich subgraph.
Subgraph + snippets → LLM → grounded answer with PMIDs/experiment IDs.
