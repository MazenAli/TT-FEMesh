SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = build
DIRS		  = ttfemesh tests

default: all

all: format_check static test_coverage secrets_check security imports

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

docs-%:
	@$(SPHINXBUILD) -M $* "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

format:
	black --line-length 100 $(DIRS)
	isort --multi-line 3 --trailing-comma --force-grid-wrap 0 --use-parentheses --line-width 100 $(DIRS)

format_check:
	black --line-length 100 --check $(DIRS)
	isort --multi-line 3 --trailing-comma --force-grid-wrap 0 --use-parentheses --line-width 100 $(DIRS) --check-only

static:
	flake8 $(DIRS)
	mypy $(DIRS) --ignore-missing-imports --no-strict-optional

test:
	pytest tests/

test_coverage:
	coverage run --source=ttfemesh --module pytest -v tests/ && coverage report -m
	coverage xml

security:
	bandit --configfile bandit.yml --recursive $(DIRS)

imports:
	vulture $(DIRS)
	pip-missing-reqs $(DIRS)

.PHONY: help docs-% format format_check static test test_coverage secrets_check security imports
