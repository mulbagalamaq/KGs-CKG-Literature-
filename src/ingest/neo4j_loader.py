"""Load dual CKG + PubMedKG CSV exports into Neo4j."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, Iterable

from neo4j import GraphDatabase

from src.utils.config import load_config


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load CKG + PubMedKG CSVs into Neo4j")
    parser.add_argument("--config", default="configs/neo4j.yaml", help="Path to configuration file")
    parser.add_argument("--ckg-dir", default="data/graph/ckg", help="Directory containing CKG nodes/edges CSVs")
    parser.add_argument("--pkg-dir", default="data/graph/pkg", help="Directory containing PubMedKG nodes/edges CSVs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    load_neo4j_from_dirs(cfg, Path(args.ckg_dir), Path(args.pkg_dir))


def load_neo4j_from_dirs(cfg, ckg_dir: Path, pkg_dir: Path) -> None:
    """Programmatic entry: load CKG and PKG CSV directories into Neo4j.

    Args:
        cfg: Loaded configuration object from src.utils.config.load_config
        ckg_dir: Directory containing CKG nodes.csv and edges.csv
        pkg_dir: Directory containing PKG nodes.csv and edges.csv
    """
    neo4j_cfg = _read_neo4j_config(cfg.raw.get("neo4j", {}))
    ckg_dir = Path(ckg_dir).resolve()
    pkg_dir = Path(pkg_dir).resolve()

    if not ckg_dir.exists():
        raise FileNotFoundError(f"CKG graph directory not found: {ckg_dir}")
    if not pkg_dir.exists():
        raise FileNotFoundError(f"PKG graph directory not found: {pkg_dir}")

    LOGGER.info("Loading Neo4j data from %s (CKG) and %s (PKG)", ckg_dir, pkg_dir)

    driver = GraphDatabase.driver(neo4j_cfg["uri"], auth=(neo4j_cfg["username"], neo4j_cfg["password"]))
    try:
        with driver.session(database=neo4j_cfg.get("database", "neo4j")) as session:
            _create_constraints(session)
            _load_dataset(session, ckg_dir / "nodes.csv", ckg_dir / "edges.csv", namespace="CKG")
            _load_dataset(session, pkg_dir / "nodes.csv", pkg_dir / "edges.csv", namespace="PKG")
    finally:
        driver.close()


def _read_neo4j_config(raw: Dict[str, str]) -> Dict[str, str]:
    missing = [key for key in ("uri", "username", "password") if not raw.get(key)]
    if missing:
        raise ValueError(f"Neo4j configuration missing keys: {', '.join(missing)}")
    return raw


def _create_constraints(session) -> None:
    constraints = [
        "CREATE CONSTRAINT ckg_id IF NOT EXISTS FOR (n:CKG) REQUIRE n.`~id` IS UNIQUE",
        "CREATE CONSTRAINT pkg_id IF NOT EXISTS FOR (n:PKG) REQUIRE n.`~id` IS UNIQUE",
    ]
    for statement in constraints:
        session.run(statement)
    LOGGER.info("Ensured Neo4j uniqueness constraints for CKG and PKG namespaces")


def _load_dataset(session, nodes_path: Path, edges_path: Path, *, namespace: str) -> None:
    if not nodes_path.exists():
        LOGGER.warning("Nodes CSV not found for namespace %s: %s", namespace, nodes_path)
        return
    if not edges_path.exists():
        LOGGER.warning("Edges CSV not found for namespace %s: %s", namespace, edges_path)
        return

    node_rows = list(_read_csv(nodes_path))
    edge_rows = list(_read_csv(edges_path))

    LOGGER.info("Loading %s nodes and %s edges for namespace %s", len(node_rows), len(edge_rows), namespace)

    for row in node_rows:
        _merge_node(session, row)

    for row in edge_rows:
        _create_relationship(session, row)

    LOGGER.info("Finished loading namespace %s", namespace)


def _merge_node(session, row: Dict[str, str]) -> None:
    node_id = row.get("~id")
    if not node_id:
        LOGGER.warning("Skipping node without ~id: %s", row)
        return

    dynamic_label = row.get("~label")
    namespace_label = _namespace_from_label(dynamic_label)
    labels = [label for label in {namespace_label, dynamic_label} if label]

    properties = {k: _coerce_value(v) for k, v in row.items() if v is not None}

    query = (
        "MERGE (n {`~id`: $id}) "
        "SET n += $props "
        "WITH n "
        "CALL apoc.create.addLabels(n, $labels) YIELD node "
        "RETURN node"
    )
    session.run(query, id=node_id, props=properties, labels=labels)


def _create_relationship(session, row: Dict[str, str]) -> None:
    start_id = row.get("~from")
    end_id = row.get("~to")
    rel_type = row.get("~label")
    if not start_id or not end_id or not rel_type:
        LOGGER.warning("Skipping relationship with missing identifiers: %s", row)
        return

    properties = {k: _coerce_value(v) for k, v in row.items() if k not in {"~from", "~to"} and v is not None}

    query = (
        "MATCH (a {`~id`: $start_id}), (b {`~id`: $end_id}) "
        "CALL apoc.create.relationship(a, $type, $props, b) YIELD rel "
        "RETURN rel"
    )
    session.run(query, start_id=start_id, end_id=end_id, type=rel_type, props=properties)


def _read_csv(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cleaned = {key: (value if value != "" else None) for key, value in row.items()}
            yield cleaned


def _namespace_from_label(label: str | None) -> str | None:
    if not label:
        return None
    if label.startswith("CKG_"):
        return "CKG"
    if label.startswith("PKG_"):
        return "PKG"
    return label.split("_")[0]


def _coerce_value(value: str | None) -> object:
    if value is None:
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value



if __name__ == "__main__":  # pragma: no cover
    main()
