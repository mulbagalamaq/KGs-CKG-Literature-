"""CKG ingest utilities for raw exports (no synthetic data)."""

from __future__ import annotations

import argparse
import logging
from typing import Dict

import pandas as pd

from .base import BaseIngestor
from src.utils.config import load_config
from src.utils.seed import seed_everything


LOGGER = logging.getLogger(__name__)

README = """\
Usage: provide Clinical Knowledge Graph CSV exports under the configured
`paths.local_data_dir/ckg/processed` directory. The ingestor will fail fast if
any required file is missing or empty. No synthetic defaults are generated.
"""


class CkgIngestor(BaseIngestor):
    """Load CKG experiments, measurements, findings, pathways, and samples."""

    REQUIRED_FILES = {
        "ckg_experiments": "ckg_experiments.csv",
        "ckg_samples": "ckg_samples.csv",
        "ckg_measurements": "ckg_measurements.csv",
        "ckg_findings": "ckg_findings.csv",
        "ckg_pathways": "ckg_pathways.csv",
        "ckg_interactions": "ckg_interactions.csv",
    }

    def ingest(self) -> Dict[str, pd.DataFrame]:
        seed_everything()

        data = {}
        for label, filename in self.REQUIRED_FILES.items():
            path = self.staging_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"CKG ingest missing required file: {path}")
            df = pd.read_csv(path)
            if df.empty:
                raise ValueError(f"CKG ingest found empty file: {path}")
            LOGGER.info("Loaded %s rows from %s", len(df), path)
            data[label] = df

        if self.sample_size:
            for key in ["ckg_experiments", "ckg_measurements", "ckg_findings", "ckg_interactions"]:
                data[key] = data[key].head(self.sample_size)

        return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CKG exports")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ingestor = CkgIngestor(name="ckg", config=cfg, sample_size=args.limit or 0)
    ingestor.run()


if __name__ == "__main__":  # pragma: no cover
    print(README)
    main()


