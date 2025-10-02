from __future__ import annotations

from typing import Dict, List

from src.gnn.features import build_feature_matrix
from src.gnn.rerank import rerank_candidates


def build_features(nodes: List[Dict]) -> List[List[float]]:
    """Return a simple feature matrix for nodes."""
    return build_feature_matrix(nodes)


def rerank(nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
    """Return nodes reranked using the optional GNN stub."""
    return rerank_candidates(nodes, edges)


