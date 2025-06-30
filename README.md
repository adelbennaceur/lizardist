# LizarDist 🦎📡

A minimal distributed training framework using MPI and PyTorch.

## Overview

LizarDist is an educational implementation of some distributed training approaches. The idea is showcases the internals of parallel training without relying on high-level abstractions.

## Tech Stack

- Python 3
- PyTorch (torch.nn, torch.autograd)
- mpi4py (for communication)
- NumPy

## Directory Structure

```
src/
├── engine/          # Core model and training components
├── dist/            # MPI communication and synchronization
├── datasets/        # Dataset implementations
├── examples/        # Training examples
├── utils/           # Utility functions
```

## Installation

This project was tested on ubuntu 22.04 LTS

install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

install `MPICH`

```bash
sudo apt-get install mpich
```

Create a vitrtual venv for uv

```bash
uv venv && source .venv/bin/activate
```

install dependencies

```bash
uv pip install -r pyproject.toml
```

## Running Examples

This project was tested on 2 GPUS Rented from [vast.ai](https://vast.ai/).

```bash
uv run mpirun -np 2 python examples/train_mnist.py
```

### Core Components

- `dist/communicator.py`: MPI communication wrapper
- `dist/sync.py`: Gradient synchronization and parameter broadcasting wrapper
- `engine/model.py`: Model definitions
- `utils/logger.py`: Rank-aware logging

# 🚀 Next Steps

Currently, LizarDist only supports data parallelism using manual gradient synchronization via MPI.

- [x] Implement Gradient Bucketing: Group gradients into larger buckets before allreduce to reduce communication overhead. Benchmark and compare the number of collective calls (CC) before and after.

- [ ] Add Ring AllReduce: Expirement/Implment ring AllReduce.

- [ ] Model Parallelism: split model layers across multiple processes. Start with basic layer sharding to study compute/memory trade-offs.

- [ ] Pipeline Parallelism: Divide the model into sequential stages across ranks.

- [ ] Tensor Parallelism: Partition individual tensor operations (e.g., matrix multiplications) across processes.

# ⚠️ Disclaimer

LizarDist is an educational project built as a personal exercise to understand and implement distributed training fundamentals. It is not intended for production use.
