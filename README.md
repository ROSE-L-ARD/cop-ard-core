<!--
  Copyright 2023 European Union

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

# COP ARD CORE

Core scientific algorithm library of the COP ARD processing chain.

`cop-ard-core` implements the low-level SAR processing kernels as
NumPy/SciPy/Numba code exposed through Xarray and Dask: SLC I/O, orbit and
geometry computation, image matching, Enhanced Spectral Diversity (ESD),
split-spectrum processing, DEM handling, ionospheric estimation, detrending,
phase unwrapping, warping/resampling, and the CMLP/GMLP building blocks.

It is the bottom layer of the stack:
`cop-ard-processor` → `cop-ard-app` → `cop-ard-core`.

## Project Structure

This project is organized as follows:

* `cop_ard_core/`: source code of the project's main Python package
  (SLC/orbit/matching/ESD/split-spectrum/DEM/ionosphere/detrend/unwrap/warp
  kernels, CMLP/GMLP building blocks, logging, utilities; typed, `py.typed`).
* `tests/`: unit and integration tests (`test_*.py`) with shared fixtures
  (`conftest.py`).
* `comment_coverage.awk`: comment-coverage gate script used by CI.
* `makefile`: `format`, `lint`, `security`, `complexity`, `typecheck`, `check`, `test`, `check-secrets`, and `protect-secrets` targets.
* `Dockerfile`: container image build.
* `pyproject.toml` / `uv.lock`: package metadata and pinned dependency
  lockfile.

## Installation

The project requires Python 3.13 (`>=3.13, <3.14`) and is managed with
[`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
```

This creates the local `.venv` from `pyproject.toml` and the pinned
`uv.lock`; `uv` provisions Python 3.13 automatically.

## Development

Quality gates are driven by the `makefile`: `make all` runs `format`
(`ruff format` and import sorting over the package and `tests/`) followed
by `check` (`ruff check`, `bandit`, `xenon`, and `mypy`); `make test` runs
the test suite.

The GitHub Actions pipeline (`.github/workflows/ci.yml`) runs on every push
to `main` and on every pull request against `main`, enforcing:

* Linting (`ruff check --no-fix`) and formatting check (`ruff format --check`)
* Typing (`mypy`)
* Unit tests (`pytest`) with coverage gate (`pytest --cov`, fail below 70%,
  measured with `NUMBA_DISABLE_JIT=1` so Numba kernels are instrumented)
* Complexity (`xenon --max-average b --max-modules c --max-absolute d`)
* Vulnerability scanning (`bandit` with `bandit.yml`)
* Docstring coverage (`docstr-coverage`) and comment coverage
  (`radon raw` + `comment_coverage.awk`, threshold 20%)

## Run tests

To execute the tests locally using `pytest`, run:

```bash
uv run --extra tests pytest tests/
```

## Updating dependencies

`cop-ard-core` has no in-stack git dependencies; it pins only third-party
libraries. To refresh a pinned dependency in `uv.lock`, run:

```bash
uv lock -P <package>
```

To upgrade all dependencies:

```bash
uv lock --upgrade
```

To apply the updated lockfile to the local virtual environment, run:

```bash
uv sync --frozen
```
