# Onion: Other Networks In One Network

![Vantage6 Algorithm CI](https://github.com/Health-AI-Consortium/Onion/actions/workflows/github_actions.yaml/badge.svg)

Onion is an experiment in **interoperability between federated learning (FL) platforms**.

Today, an FL study lives inside a single platform: Vantage6, nvFlare, Flower, or another.
Participants on different platforms cannot join the same study, even when they all want to
collaborate on the same model. Onion's answer is a **hierarchical architecture**: two
networks running their own (different) platforms become interoperable by having their
**aggregation servers act as regular nodes in a network one layer above**. Raw data never
leaves the bottom layer; only model updates travel up and down the hierarchy.

## How it works

In Onion, a network's aggregation server can be plugged into the network one level up
as if it were an ordinary participating node:

```
             ┌─────────────────────────────────────────────────────┐
             │                  Top-level network                  │
             │                                                     │
             │  ┌───────────────┐               ┌───────────────┐  │
             │  │  Aggregator A │               │  Aggregator B │  │
             │  └───────┬───────┘               └───────┬───────┘  │
             └──────────┼───────────────────────────────┼──────────┘
                        │                               │
             ┌──────────┴──────────┐         ┌──────────┴──────────┐
             │      Network A      │         │      Network B      │
             │   (its platform)    │         │   (its platform)    │
             │                     │         │                     │
             │  ┌─────┐  ┌─────┐   │         │  ┌─────┐  ┌─────┐   │
             │  │ org │  │ org │   │         │  │ org │  │ org │   │
             │  └─────┘  └─────┘   │         │  └─────┘  └─────┘   │
             └─────────────────────┘         └─────────────────────┘
```

Training runs as a loop of tasks across the layers:

1. The top-level network runs a **`central`** task that dispatches a **`local`** task to
   the participating nodes — which, at this layer, are the aggregators of the lower networks.
2. Each node checks the `END_NODE` environment variable to decide what "local" means:
   - `END_NODE=TRUE` → the node has data of its own, so it **trains locally**
     (`train_locally`) and returns its updated weights.
   - `END_NODE=FALSE` → the node is a **bridge** (`sub_central`): it connects to the
     server of the network *below* it and launches a `central` task there. The lower
     network then trains across its own organizations and returns aggregated weights
     back up.
3. The top-level `central` task collects the results from all of its nodes, aggregates
   them (weighted averaging), and starts the next round.

So the same algorithm code runs at every layer; only the `END_NODE` switch and the
configuration tell a node whether it is a data-holding leaf or a bridge to another network.
The hierarchy can in principle be extended to more than two levels.

## Current status

Onion is in **early development**. What exists today is a two-level proof of concept built
on [Vantage6](https://vantage6.ai/):

- A 2-level setup (top-level network + one or more sub-networks) where every layer is a
  Vantage6 collaboration.
- A small PyTorch CNN (`base_NN`) trained on a partitioned MNIST dataset, one partition per
  bottom-level organization.
- The `central` / `local` / `sub_central` task flow described above, including the
  `sub_central` bridge that connects to a lower Vantage6 server via environment variables.
- A mock-based test (`test_v6.py`) that exercises the full two-level flow using Vantage6's
  `MockAlgorithmClient` — this runs in CI.

Work in progress: the weight-aggregation step is still a stub, and the real multi-server
deployment flow is being finalized. The `Dockerfile` builds the standard Vantage6
algorithm image, and CI pushes it to GitHub Container Registry on release tags. The API
and file layout will keep changing as more layers and platforms are added.

## Roadmap

The design goal is a platform-agnostic layering rule: *each platform's aggregator server
implements the same "node in the upper network" interface.* Adding a platform then means
writing one adapter that lets that platform's aggregator participate as a node (and expose
its own sub-network underneath).

- [ ] **Vantage6** — proof of concept (this repository)
- [ ] **nvFlare** adapter
- [ ] **Flower** adapter
- [ ] Additional platforms as needed
- [ ] More than two hierarchy levels, driven by an explicit architecture definition

## Project layout

| Path | Description |
| --- | --- |
| `src_v6/central.py` | The `central` task: dispatched by a node, orchestrates the `local` tasks of the network below it and aggregates the results. |
| `src_v6/local.py` | The `local` task entry point. Branches on `END_NODE`: `train_locally` (leaf with data) or `sub_central` (bridge to the lower network). |
| `src_v6/mock_local.py` | Mock-client variant of `local`, used by the test suite. |
| `src_v6/models.py` / `src_v6/model.py` | The toy model (`base_NN`, a 3-layer CNN for MNIST) and a small model factory. |
| `src_v6/average.py` | Weight aggregation across client results (WIP stub). |
| `Dockerfile` | Standard Vantage6 algorithm image (based on `algorithm-base:4.15`); the entrypoint wraps the `src_v6` package via `wrap_algorithm()`. |
| `test_v6.py` | Two-level end-to-end mock test run in CI. |
| `test/test_data/` | MNIST downloader and partitioning utilities for the proof of concept. |
| `.github/workflows/github_actions.yaml` | CI: lint, mock test, and Docker image build/push on release tags. |

## Getting started

> This is a work in progress — the documented flow below is the **mock test**, which
> currently works end to end. Deploying to real multi-server Vantage6 networks is being
> finalized.

### Prerequisites

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) for environment and dependency management

### Install

```bash
git clone git@github.com:Health-AI-Consortium/Onion.git
cd Onion
uv sync
```

### Download the test data

The proof of concept trains on MNIST, which is not committed to the repository
(`test/test_data/MNIST/` is gitignored). Download it once before running the test:

```bash
uv add --dev kagglehub
python test/test_data/download_MNIST.py
```

### Configure

Copy the example environment file and adjust it for your setup:

```bash
cp .env.example .env
```

| Variable | Description |
| --- | --- |
| `END_NODE` | `TRUE` if this node has local data (bottom of the hierarchy), `FALSE` if it bridges to another network/server. Set per node. |
| `SERVER_URL` | URL of the (sub-)server this node talks to. |
| `SERVER_PORT` | Port of the (sub-)server. |
| `SERVER_API` | API path, e.g. `/api`. |
| `USERNAME` | Username for authenticating against the server. |
| `PASSWORD` | Password for authenticating against the server. |
| `PRIVATE_KEY` | Path to a private key, used when the collaboration has encryption enabled. |

### Run the mock test

The test builds a two-level mock network (a bridge node on top of two mock
organizations) and runs the full `central` → `sub_central` → `train_locally` flow:

```bash
python test_v6.py
```

### Running against real servers

The algorithm is deployed as a standard Vantage6 algorithm: the `Dockerfile` is based on
Vantage6's `algorithm-base` image, and its entrypoint wraps the `src_v6` package with
`wrap_algorithm()`. CI builds the image and pushes it to GitHub Container Registry when
you create a release tag (`release/vX.Y.Z`, or `dev`); you then point a Vantage6
algorithm configuration at that image. Exact multi-node deployment steps will be
documented once the deployment flow is finalized.

## Contributing

Pull requests are welcome. The repo uses `uv` for dependency management and a pre-commit
setup; CI runs flake8 and the mock test on every push to `main`.

Onion is developed by the [Health AI Consortium](https://github.com/Health-AI-Consortium).
