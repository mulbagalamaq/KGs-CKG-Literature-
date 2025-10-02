"""Convert CKG CSV exports into Neptune openCypher bulk load format."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.utils.config import load_config
from src.utils.io import ensure_dir, write_dataframe


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map CKG CSV exports to Neptune openCypher CSVs")
    parser.add_argument("--config", default="configs/ingest_ckg.yaml")
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--nodes-glob")
    parser.add_argument("--relationships-glob")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    input_dir = Path(args.input_dir or cfg.get("ckg_ingest.input_dir"))
    output_dir = ensure_dir(args.output_dir or cfg.get("ckg_ingest.output_dir", "data/graph/ckg"))

    nodes_glob = args.nodes_glob or cfg.get("ckg_ingest.nodes_glob", "ckg_nodes_*.csv")
    relationships_glob = args.relationships_glob or cfg.get("ckg_ingest.relationships_glob", "ckg_relationships_*.csv")

    LOGGER.info("Loading CKG input from %s", input_dir)

    node_files = sorted(input_dir.glob(nodes_glob))
    rel_files = sorted(input_dir.glob(relationships_glob))

    if not node_files:
        raise FileNotFoundError(f"No node CSVs matching pattern '{nodes_glob}' in {input_dir}")
    if not rel_files:
        raise FileNotFoundError(f"No relationship CSVs matching pattern '{relationships_glob}' in {input_dir}")

    node_df = _concat(node_files)
    rel_df = _concat(rel_files)

    if node_df.empty:
        raise ValueError("CKG node CSVs produced no rows")
    if rel_df.empty:
        raise ValueError("CKG relationship CSVs produced no rows")

    LOGGER.info("Loaded %s nodes and %s relationships", len(node_df), len(rel_df))

    mapped_nodes = _map_nodes(node_df)
    mapped_edges = _map_edges(rel_df)

    node_path = output_dir / "nodes.csv"
    edge_path = output_dir / "edges.csv"

    write_dataframe(mapped_nodes, node_path, index=False)
    write_dataframe(mapped_edges, edge_path, index=False)

    LOGGER.info("Wrote Neptune-formatted nodes to %s", node_path)
    LOGGER.info("Wrote Neptune-formatted edges to %s", edge_path)


def _concat(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True, sort=False)


def _map_nodes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "id" not in df.columns or "label" not in df.columns:
        raise KeyError("Node CSV must contain 'id' and 'label' columns")

    df.rename(columns={"id": "~id", "label": "~label"}, inplace=True)
    df["~label"] = df["~label"].apply(lambda label: f"CKG_{label}")
    return df


def _map_edges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = {"start_id", "end_id", "type"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Relationship CSV missing columns: {', '.join(sorted(missing))}")

    df.rename(columns={"start_id": "~from", "end_id": "~to", "type": "~label"}, inplace=True)
    df["~label"] = df["~label"].apply(lambda rel: f"CKG_{rel}")
    return df


if __name__ == "__main__":  # pragma: no cover
    main()
