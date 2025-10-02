"""Graph expansion utilities using Neo4j Cypher queries."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from neo4j import GraphDatabase

from src.utils.config import load_config


LOGGER = logging.getLogger(__name__)


def expand_subgraph(config_path: str, seed_nodes: List[str], hops: int = 2, max_degree: int = 10) -> Dict:
    if not seed_nodes:
        return []

    cfg = load_config(config_path)
    neo4j_cfg = cfg.get("neo4j")
    if not neo4j_cfg:
        raise ValueError("Neo4j configuration is required for graph expansion")

    namespaces = cfg.get("retrieval.namespaces.include", []) or ["CKG_", "PKG_"]
    uri = neo4j_cfg.get("uri")
    username = neo4j_cfg.get("username")
    password = neo4j_cfg.get("password")
    database = neo4j_cfg.get("database", "neo4j")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        with driver.session(database=database) as session:
            result = session.run(
                _build_query(namespaces),
                ids=seed_nodes,
                hops=hops,
                maxDegree=max_degree,
                namespaces=namespaces,
            )

            records = []
            for record in result:
                nodes = [_to_node_dict(node) for node in record["nodes"]]
                rels = [_to_rel_dict(rel) for rel in record["rels"]]
                records.append({"nodes": nodes, "rels": rels})
            return records
    finally:
        driver.close()


def _build_query(namespaces: List[str]) -> str:
    label_check = " OR ".join(
        f"ANY(label IN labels(n) WHERE label STARTS WITH '{ns}')" for ns in namespaces
    )
    rel_check = " OR ".join(f"type(r) STARTS WITH '{ns}'" for ns in namespaces)
    cypher = [
        "MATCH (n)",
        "WHERE n.`~id` IN $ids AND (" + label_check + ")",
        "MATCH p=(n)-[r*..$hops]-(m)",
        "WHERE ALL(r IN relationships(p) WHERE (" + rel_check + "))",
        "  AND ALL(r IN relationships(p) WHERE size((startNode(r))--()) <= $maxDegree)",
        "RETURN nodes(p) AS nodes, relationships(p) AS rels",
        "LIMIT 2000",
    ]
    return "\n".join(cypher)


def _to_node_dict(node) -> Dict:
    node_dict: Dict = {}
    for key in node.keys():
        node_dict[key] = node[key]

    node_dict["~id"] = node.get("~id")
    node_dict["~label"] = _first_namespace_label(node)
    return node_dict


def _to_rel_dict(rel) -> Dict:
    rel_dict: Dict = {}
    for key in rel.keys():
        rel_dict[key] = rel[key]

    rel_dict["~label"] = rel.type
    rel_dict["~from"] = rel.start_node.get("~id")
    rel_dict["~to"] = rel.end_node.get("~id")
    return rel_dict


def _first_namespace_label(node) -> Optional[str]:
    for label in node.labels:
        if label.startswith("CKG_") or label.startswith("PKG_"):
            return label
    return None

