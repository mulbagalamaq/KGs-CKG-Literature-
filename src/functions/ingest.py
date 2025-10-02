from __future__ import annotations

from pathlib import Path

from src.utils.config import load_config
from src.ingest.neo4j_loader import load_neo4j_from_dirs


def load_neo4j(config_path: str, ckg_dir: str | Path, pkg_dir: str | Path) -> None:
    """Load CKG and PKG CSV directories into Neo4j using config.

    Args:
        config_path: Path to YAML config with `neo4j` connection settings.
        ckg_dir: Directory containing CKG nodes.csv and edges.csv.
        pkg_dir: Directory containing PKG nodes.csv and edges.csv.
    """
    cfg = load_config(config_path)
    load_neo4j_from_dirs(cfg, Path(ckg_dir), Path(pkg_dir))


