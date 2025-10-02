# Dual-KG GraphRAG (CKG + PubMedKG)

This project demonstrates a dual-knowledge-graph workflow that ingests the Clinical Knowledge Graph (CKG) and PubMedKG side-by-side into a single Amazon Neptune cluster. No harmonization or synthetic data generation is performed—each KG keeps its own schema using namespaced labels (`CKG_*`, `PKG_*`). Downstream GraphRAG retrieval performs query-time joins across the two graphs using exact identifier matches and vector similarity.

## Repository layout

- `configs/` – YAML configuration files for ingest, loading, and runtime services.
- `src/data/` – dataset ingestion utilities that expect pre-exported CSV/TSV files.
- `src/ingest/` – converters that map raw exports to Neptune bulk-load openCypher CSVs.
- `src/retrieval/` – GraphRAG expansion, pruning, and vector store helpers.
- `src/embeddings/` – document and node embedding builders sourcing graph outputs.
- `docs/dual_kg.md` – detailed explanation of the dual-KG architecture and operations.

## Prerequisites

1. Export CKG data (e.g., via Neo4j APOC) into CSV files matching the expected schema.
2. Download PubMedKG CSV/TSV extracts.
3. Place files under `data/local/ckg/exports/` and `data/local/pubmedkg/exports/` with filenames configured in `configs/ingest_ckg.yaml` and `configs/ingest_pkg.yaml`.
4. Provide AWS credentials with permissions for Amazon S3 and Neptune bulk loading.

## Quickstart

```bash
make setup
make data                 # optional: validates input extracts are present
make ckg_to_neptune       # writes data/graph/ckg/nodes.csv and edges.csv
make pkg_to_neptune       # writes data/graph/pkg/nodes.csv and edges.csv
make load_ckg             # emits Neptune loader JSON for s3://.../graph/ckg/
make load_pkg             # emits Neptune loader JSON for s3://.../graph/pkg/
make neo4j_load           # loads both CSV sets into Neo4j (local bolt://)
```

Upload the generated CSV files to the configured S3 bucket prefixes, then submit the loader JSON payloads to Neptune’s bulk loader API separately for each KG namespace.

For Neo4j-based development, ensure a Neo4j instance is running (e.g., Docker `neo4j:5`) and update credentials in `configs/neo4j.yaml` before running `make neo4j_load`.

## Embeddings and Retrieval

1. Populate OpenSearch vectors and Neptune-expanded subgraphs:
   ```bash
   make embed     # builds document & node embeddings from graph outputs
   make qa        # runs a sample GraphRAG question answering pipeline
   ```
2. The retrieval pipeline executes vector KNN against OpenSearch, expands matching nodes up to two hops in Neptune while respecting `CKG_*` and `PKG_*` namespaces, prunes the subgraph, and assembles an LLM answer with citations.

For a Neo4j-only run, use:
```bash
make embed
make qa_neo4j
```

## API & UI

```bash
make api         # FastAPI endpoint for QA
make ui          # Streamlit interface (defaults to port 8501)
```

See `docs/dual_kg.md` for deeper architectural context, query strategies, and operational guidance.
