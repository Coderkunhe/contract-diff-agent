.PHONY: install run test clean web web-install

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install:
	@test -d $(VENV) || python3 -m venv $(VENV)
	@echo "==> Installing dependencies..."
	$(PIP) install --upgrade pip -q
	$(PIP) install -e . -q
	$(PIP) install fastapi "uvicorn[standard]" python-multipart jinja2 -q
	@echo "==> Done. Run 'source .venv/bin/activate' or use 'make run'"

run:
	@test -f .env || { echo "ERROR: .env not found. Copy .env.example or create one with ANTHROPIC_API_KEY"; exit 1; }
	$(PYTHON) -m src.main docs/天猫服务协议2015\(2\).pdf docs/天猫服务协议2026\(2\).pdf -o data/diff_result.json

run-validate:
	@test -f .env || { echo "ERROR: .env not found."; exit 1; }
	$(PYTHON) -m src.main docs/天猫服务协议2015\(2\).pdf docs/天猫服务协议2026\(2\).pdf -o data/diff_result.json --validate

web:
	@test -f .env || { echo "ERROR: .env not found."; exit 1; }
	$(PYTHON) -m web.app

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	rm -rf data/*.json data/jobs/ data/uploads/
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ web/__pycache__
	rm -rf .pytest_cache

clean-all: clean
	rm -rf $(VENV)
