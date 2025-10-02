"""Utilities to compute text/document embeddings from dual CKG + PubMedKG graphs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd
from sentence_transformers import SentenceTransformer

from src.utils.config import load_config
from src.utils.io import ensure_dir, write_dataframe
from src.utils.seed import seed_everything


LOGGER = logging.getLogger(__name__)


def build_document_embeddings(config_path: str) -> Path:
    cfg = load_config(config_path)
    seed_everything()

    model_name = cfg.get("embedding_model.document_model", "sentence-transformers/all-MiniLM-L12-v2")
    model = SentenceTransformer(model_name)

    docs = _collect_documents(cfg)
    if not docs:
        raise ValueError("No publications or findings available for text embeddings.")

    texts = [doc["text"] for doc in docs]
    embeddings = model.encode(texts, show_progress_bar=False)

    df_emb = pd.DataFrame(embeddings)
    df_emb.insert(0, "id", [doc["id"] for doc in docs])
    df_emb.insert(1, "type", [doc["type"] for doc in docs])
    df_emb.insert(2, "text", texts)

    output_dir = ensure_dir(cfg.get("paths.embeddings_dir", "data/embeddings"))
    output_path = write_dataframe(df_emb, Path(output_dir) / "document_embeddings.csv", index=False)
    LOGGER.info("Wrote document embeddings to %s", output_path)
    return output_path


def _collect_documents(cfg) -> List[dict]:
    docs: List[dict] = []
    pkg_nodes = Path(cfg.get("pkg_ingest.output_dir", "data/graph/pkg")) / "nodes.csv"
    if not pkg_nodes.exists():
        raise FileNotFoundError(f"PubMedKG nodes CSV missing: {pkg_nodes}")

    publications = pd.read_csv(pkg_nodes)
    publications = publications[publications["~label"] == "PKG_Publication"]
    for _, row in publications.iterrows():
        pmid = row["~id"]
        text_parts = ["PKG_Publication"]
        for column, value in row.items():
            if column in {"~id", "~label"}:
                continue
            if pd.isna(value):
                continue
            text_parts.append(f"{column}: {value}")
        text = " | ".join(text_parts)
        docs.append({"id": f"PKG::{pmid}", "text": text, "type": "PKG_Publication"})

    ckg_nodes = Path(cfg.get("ckg_ingest.output_dir", "data/graph/ckg")) / "nodes.csv"
    if not ckg_nodes.exists():
        raise FileNotFoundError(f"CKG nodes CSV missing: {ckg_nodes}")

    findings = pd.read_csv(ckg_nodes)
    findings = findings[findings["~label"] == "CKG_Finding"]
    for _, row in findings.iterrows():
        finding_id = row["~id"]
        text_parts = ["CKG_Finding"]
        for column, value in row.items():
            if column in {"~id", "~label"}:
                continue
            if pd.isna(value):
                continue
            text_parts.append(f"{column}: {value}")
        text = " | ".join(text_parts)
        docs.append({"id": f"CKG::{finding_id}", "text": text, "type": "CKG_Finding"})

    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build document embeddings")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    build_document_embeddings(args.config)


if __name__ == "__main__":  # pragma: no cover
    main()

