"""LLM interface for producing answers from GraphRAG context with optional PyG fusion."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from src.gnn.pyg_rag import (
    build_pyg_from_subgraph,
    encode_texts,
    fuse_vectors,
    gnn_graph_embedding,
    rank_facts,
    set_structural_vectors,
    structural_fact_vectors,
    textify_subgraph,
)
from src.qa.prompt import build_prompt
from src.retrieval.g_retriever import initialize_vector_store, retrieve_graph_context
from src.utils.config import load_config
from src.utils.seed import seed_everything


LOGGER = logging.getLogger(__name__)


def answer_question(config_path: str, question: str) -> Dict:
    """Return answer dict with optional PyG-enhanced ranking."""
    seed_everything()
    cfg = load_config(config_path)
    model_name = cfg.get("embedding_model.document_model", "sentence-transformers/all-MiniLM-L12-v2")
    model = SentenceTransformer(model_name)
    question_vector = model.encode([question], show_progress_bar=False)[0].tolist()

    retrieval = retrieve_graph_context(config_path, question_vector)
    nodes = retrieval["nodes"]
    edges = retrieval["edges"]

    prompt_nodes = nodes
    prompt_edges = edges
    fact_texts: List[str] = []

    if cfg.get("pyg_rag", {}).get("enabled", True):
        try:
            fact_texts = textify_subgraph(nodes, edges)
            if fact_texts:
                data, node_index = build_pyg_from_subgraph(nodes, edges)
                gnn_vec = np.asarray(gnn_graph_embedding(data, cfg), dtype=np.float32)

                combined_text = [" ".join(fact_texts)]
                llm_matrix = encode_texts(combined_text, cfg)
                llm_vec = llm_matrix[0] if llm_matrix.shape[0] else np.zeros((model.get_sentence_embedding_dimension(),), dtype=np.float32)

                node_embeddings = None
                if hasattr(data, "node_embeddings"):
                    node_embeddings = data.node_embeddings.detach().cpu().numpy()
                struct_vectors = structural_fact_vectors(nodes, edges, node_embeddings, node_index)
                set_structural_vectors(struct_vectors)

                fused = fuse_vectors(gnn_vec, llm_vec)

                ranked_indices = rank_facts(fused, fact_texts, cfg)
                top_facts = cfg.get("pyg_rag", {}).get("top_facts", 40)
                top_indices = ranked_indices[:top_facts]
                prompt_nodes, prompt_edges = _filter_facts_by_indices(top_indices, nodes, edges)
                LOGGER.info("PyG fusion selected %s facts for prompt", len(top_indices))
            else:
                LOGGER.info("PyG fusion skipped: empty fact list")
        except Exception as exc:  # pragma: no cover - fallback safety
            LOGGER.warning("PyG fusion failed (%s); falling back to baseline", exc)
        finally:
            set_structural_vectors(None)

    prompt = build_prompt(question, prompt_nodes, prompt_edges, retrieval["evidence"])

    api_base = cfg.get("llm.api_base")
    api_key = cfg.get("llm.api_key")
    temperature = cfg.get("llm.temperature", 0.2)
    max_tokens = cfg.get("llm.max_new_tokens", 512)
    model_name = cfg.get("llm.model_name")

    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "changeme":
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": "You are a biomedical assistant. Provide concise answers with experiment IDs and PMIDs as citations.",
            },
            {"role": "user", "content": prompt},
        ],
    }

    response = requests.post(f"{api_base}/chat/completions", headers=headers, data=json.dumps(payload), timeout=60)
    response.raise_for_status()
    data = response.json()
    answer_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    return {
        "question": question,
        "answer": answer_text,
        "prompt": prompt,
        "nodes": prompt_nodes,
        "edges": prompt_edges,
        "evidence": retrieval["evidence"],
    }


def _filter_facts_by_indices(indices: List[int], nodes: List[Dict], edges: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Return only the nodes/edges corresponding to selected fact indices."""
    from src.gnn.pyg_rag import get_fact_lookup

    lookup = get_fact_lookup()
    selected_nodes: List[Dict] = []
    selected_edges: List[Dict] = []

    for idx in indices:
        if idx >= len(lookup):
            continue
        fact_type, fact_idx = lookup[idx]
        if fact_type == "node" and fact_idx < len(nodes):
            selected_nodes.append(nodes[fact_idx])
        elif fact_type == "edge" and fact_idx < len(edges):
            selected_edges.append(edges[fact_idx])

    if not selected_nodes and not selected_edges:
        return nodes, edges

    selected_node_ids = {node.get("~id") for node in selected_nodes}
    selected_edges = [edge for edge in selected_edges if edge.get("~from") in selected_node_ids and edge.get("~to") in selected_node_ids]
    return selected_nodes, selected_edges


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GraphRAG QA pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--question-file", default="configs/demo_questions.yaml")
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.initialize:
        initialize_vector_store(args.config)

    if args.question_file.endswith(".json"):
        questions = json.loads(Path(args.question_file).read_text(encoding="utf-8"))
    else:
        import yaml  # type: ignore

        questions = yaml.safe_load(Path(args.question_file).read_text(encoding="utf-8"))

    for item in questions:
        result = answer_question(args.config, item["question"])
        LOGGER.info("Answer: %s", result["answer"])


if __name__ == "__main__":  # pragma: no cover
    main()

