import logging

import torch
from mpi4py import MPI

from lizardist.distributed.communicator import Communicator
from lizardist.utils.logger import get_logger


class Synchronizer:
    def __init__(self, comm: Communicator, use_bucketing: bool = True) -> None:
        """Initialize synchronizer with MPI communicator.

        Args:
            comm: MPI communicator
            use_bucketing: Whether to use gradient bucketing
        """
        self.comm = comm
        self.use_bucketing = use_bucketing
        self.logger = get_logger(self.comm.get_rank(), "synchronizer", level=logging.INFO)

        # Set bucketing preference in communicator
        if hasattr(comm, "use_bucketing"):
            comm.use_bucketing = use_bucketing

    def sync_gradients(self, model: torch.nn.Module, use_bucketing: bool | None = None) -> None:
        """Synchronize gradients across all processes.

        Args:
            model: PyTorch model with gradients to sync
            use_bucketing: Whether to use gradient bucketing.
        """
        self.logger.debug(f"Rank {self.comm.get_rank()}: Starting gradient synchronization")

        # Collect all gradients that need synchronization
        gradients = []
        param_map = []

        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.detach().cpu().numpy())
                param_map.append(param)
        if gradients:
            self.logger.debug(f"Rank {self.comm.get_rank()}: Found {len(gradients)} gradients to sync")
            # Use bucketing if enabled
            if use_bucketing is None:
                use_bucketing = self.use_bucketing

            if use_bucketing:
                self.logger.debug(f"Rank {self.comm.get_rank()}: Using bucketed AllReduce")
                # Perform bucketed AllReduce on all gradients
                reduced_gradients = self.comm.bucketed_allreduce(gradients, op=MPI.SUM)
            else:
                self.logger.debug(f"Rank {self.comm.get_rank()}: Using individual AllReduce operations")
                # Perform individual AllReduce operations
                reduced_gradients = [self.comm.allreduce(t, op=MPI.SUM) for t in gradients]

            self.logger.debug(f"Rank {self.comm.get_rank()}: Starting gradient updates")
            # Update gradients
            for param, reduced_grad in zip(param_map, reduced_gradients, strict=False):
                avg_grad = reduced_grad / self.comm.get_world_size()
                param.grad.copy_(torch.from_numpy(avg_grad).to(param.device))
            self.logger.debug(f"Rank {self.comm.get_rank()}: Completed gradient updates")

    def broadcast_parameters(self, model: torch.nn.Module, root: int = 0) -> None:
        """Broadcast model parameters from root process to all processes."""
        self.logger.debug(f"Rank {self.comm.get_rank()}: Starting parameter broadcast from root {root}")

        for param in model.parameters():
            param_np = param.data.detach().cpu().numpy()
            param_np = self.comm.bcast(param_np, root=root)
            param.data.copy_(torch.from_numpy(param_np).to(param.device))

        self.logger.debug(f"Rank {self.comm.get_rank()}: Completed parameter broadcast")

    def get_communication_stats(self) -> dict[str, int | float]:
        """Get communication statistics."""
        return self.comm.get_bucket_stats()
