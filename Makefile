DIRS		  = ttfemesh tests
LIBS		  = ttfemesh

default: all

all: format_check static test test_coverage security imports

format:
	black --line-length 100 $(DIRS)
	isort --multi-line 3 --trailing-comma --force-grid-wrap 0 --use-parentheses --line-width 100 $(DIRS)

format_check:
	black --line-length 100 --check $(DIRS)
	isort --multi-line 3 --trailing-comma --force-grid-wrap 0 --use-parentheses --line-width 100 $(DIRS) --check-only

static:
	flake8 --max-line-length=100 $(DIRS)
	mypy $(DIRS) --ignore-missing-imports --no-strict-optional

test:
	pytest tests/

test_coverage:
	coverage run --source=ttfemesh --module pytest -v tests/ && coverage report -m
	coverage xml

security:
	bandit --configfile bandit.yml --recursive $(DIRS)

imports:
	vulture --ignore-names=side_effect $(DIRS)
	pip-missing-reqs $(DIRS) --ignore-module=pytest
	pip-extra-reqs $(DIRS)

.PHONY: help docs-% format format_check static test test_coverage secrets_check security imports
