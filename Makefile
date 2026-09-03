.PHONY: install test test-ml test-backend test-frontend test-e2e \
        backend frontend migrate lint-frontend build-frontend check-datasets

VENV := .venv/bin

install:
	python3.11 -m venv .venv
	$(VENV)/pip install --upgrade pip
	$(VENV)/pip install -e ".[full]"
	cd frontend && npm install

test: test-ml test-backend
	@echo "Run 'make test-frontend' / 'make test-e2e' separately (need Node)."

test-ml:
	$(VENV)/pytest ml/tests

test-backend:
	$(VENV)/pytest backend/tests

test-frontend:
	cd frontend && npm test

test-e2e:
	cd frontend && npm run test:e2e

backend:
	$(VENV)/uvicorn backend.app.main:app --reload

frontend:
	cd frontend && npm run dev

migrate:
	$(VENV)/alembic -c backend/alembic.ini upgrade head

lint-frontend:
	cd frontend && npm run lint

build-frontend:
	cd frontend && npm run build

check-datasets:
	$(VENV)/python -m ml.scripts.download_datasets
