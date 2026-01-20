REGISTRY :=
IMAGE_NAME := s1-worker
GIT_PAT :=
IMAGE_TAG := latest
SOURCES := s1_l12_rp
TESTS := tests


docker-build:
	docker build --pull --build-arg GIT_PAT=$(GIT_PAT) -t $(REGISTRY)$(IMAGE_NAME):$(IMAGE_TAG) .

docker-push:
	docker push $(REGISTRY)$(IMAGE_NAME):$(IMAGE_TAG)

format:
	uv run --extra formatter --frozen -m black ${SOURCES} ${TESTS}
	uv run --extra formatter --frozen -m isort ${SOURCES} ${TESTS}
	uv run --extra linter --frozen -m flake8 ${SOURCES} ${TESTS}

check-typing:
	uv run --extra typing --frozen -m mypy ${SOURCES}
