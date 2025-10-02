
Implement a system to use GNN and LLM to use knowledge graphs to integrate literature with knowledge graphs made from large public datasets 

<img width="4624" height="2838" alt="image" src="https://github.com/user-attachments/assets/9f274f12-ac4f-48e0-81c9-174de8731681" />



# WORK IN PROGRESS
Question → embed → OpenSearch top‑k seeds (across both CKG and PKG vectors).
Seeds → Neptune openCypher expansion (1–2 hops, label filters, degree caps).
Expanded graph → PCST-like pruning → compact, evidence-rich subgraph.
Subgraph + snippets → LLM → grounded answer with PMIDs/experiment IDs.
