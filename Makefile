VENV = .venv

PYTHON = $(VENV)/bin/python3

SRC = src

PARAMETERS = --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/funct.json


install:
	uv sync


$(VENV):
	@if [ ! -d $(VENV) ]; then \
		make install; \
	fi


run: $(VENV)
	uv run $(PYTHON) -m $(SRC) $(PARAMETERS)


debug: $(VENV)
	uv run $(PYTHON) -m pdb -m $(SRC) $(PARAMETERS)


lint: $(VENV)
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs


lint-strict: $(VENV)
	uv run flake8 src
	uv run mypy src --strict


clean:

	rm -rf $(VENV)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .mypy_cache
	rm -rf src/.mypy_cache

.PHONY: install run debug clean lint lint-strict