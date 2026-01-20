DIR = .
SOURCES=${DIR}/cop
TESTS=${DIR}/tests

update:
	uv lock -P s1-ard-core

install: 
	uv sync

clean:
	rm -rf .venv/ */__pycache__/ */*/__pycache__/ */.ipynb_checkpoints */*/.ipynb_checkpoints

jupyter:
	uv run --with "ipykernel<7" --with jupyter jupyter server --allow-root --port 8889 

format:
	set -e
	uv run --extra formatter python -m black ${SOURCES}
	uv run --extra formatter --frozen python -m isort ${SOURCES} ${TESTS}
	uv run --extra linter --frozen python -m flake8 ${SOURCES} ${TESTS}
	uv run --extra security --frozen python -m bandit -c ./bandit.yml -r ${SOURCES}
	uv run --extra complexity --frozen python -m xenon --max-average B --max-modules C --max-absolute D ${SOURCES}
	uv run --extra typing --frozen python -m mypy ${SOURCES} --allow-redefinition
