PYTHON ?= python3
VENV ?= .venv
ACTIVATE = source $(VENV)/bin/activate

.PHONY: help setup data ckg_to_neptune pkg_to_neptune load_ckg load_pkg neo4j_load embed qa qa_neo4j api ui teardown lint test

help:
	@echo "Available targets: setup, data, ckg_to_neptune, pkg_to_neptune, load_ckg, load_pkg, neo4j_load, embed, qa, qa_neo4j, api, ui, teardown, lint, test"

setup:
	@test -d $(VENV) || python3 -m venv $(VENV)
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -r requirements.txt

data:
	$(ACTIVATE) && $(PYTHON) -m src.data.ckg --config configs/default.yaml
	$(ACTIVATE) && $(PYTHON) -m src.data.pubmedkg --config configs/default.yaml

ckg_to_neptune:
	$(ACTIVATE) && $(PYTHON) -m src.ingest.ckg_to_neptune --config configs/ingest_ckg.yaml

pkg_to_neptune:
	$(ACTIVATE) && $(PYTHON) -m src.ingest.pkg_to_neptune --config configs/ingest_pkg.yaml

load_ckg:
	$(ACTIVATE) && $(PYTHON) -m src.ingest.neptune_loader --config configs/default.yaml --prefix graph/ckg/

load_pkg:
	$(ACTIVATE) && $(PYTHON) -m src.ingest.neptune_loader --config configs/default.yaml --prefix graph/pkg/

neo4j_load:
	$(ACTIVATE) && $(PYTHON) -m src.ingest.neo4j_loader --config configs/neo4j.yaml --ckg-dir data/graph/ckg --pkg-dir data/graph/pkg

embed:
	$(ACTIVATE) && $(PYTHON) -m src.embeddings.text_embed --config configs/default.yaml
	$(ACTIVATE) && $(PYTHON) -m src.embeddings.node_embed --config configs/default.yaml
	$(ACTIVATE) && $(PYTHON) -m src.retrieval.vector_store --config configs/default.yaml

qa:
	$(ACTIVATE) && $(PYTHON) -m src.qa.answer --config configs/default.yaml --question-file configs/demo_questions.yaml

qa_neo4j:
	$(ACTIVATE) && $(PYTHON) -m src.qa.answer --config configs/neo4j.yaml --question-file configs/demo_questions.yaml

api:
	$(ACTIVATE) && uvicorn src.api.app:app --reload

ui:
	$(ACTIVATE) && streamlit run src/ui/app.py --server.port=8501

teardown:
	$(ACTIVATE) && $(PYTHON) -m src.utils.aws --teardown --config configs/default.yaml

lint:
	$(ACTIVATE) && ruff check src tests

test:
	$(ACTIVATE) && pytest -q

