PYTHON ?= python3

.PHONY: test coverage lint typecheck validate smoke check build clean

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

coverage:
	$(PYTHON) -m coverage erase
	PYTHONPATH=src $(PYTHON) -m coverage run --branch -m unittest discover -s tests
	$(PYTHON) -m coverage report -m

lint:
	$(PYTHON) -m ruff check src tests scripts

typecheck:
	$(PYTHON) -m pyright

validate:
	PYTHONPATH=src $(PYTHON) scripts/validate_release.py

smoke:
	PYTHONPATH=src $(PYTHON) scripts/smoke_test.py

check:
	$(PYTHON) -m compileall -q src tests scripts
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) coverage
	$(MAKE) smoke
	$(MAKE) validate

build:
	$(PYTHON) scripts/build_release.py

clean:
	rm -rf build dist *.egg-info .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
