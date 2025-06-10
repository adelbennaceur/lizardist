import torch
from lizardist.distributed.communicator import Communicator
from mpi4py import MPI


class Synchronizer:
    def __init__(self, comm: Communicator) -> None:
        """Initialize synchronizer with MPI communicator."""
        self.comm = comm

    def sync_gradients(self, model: torch.nn.Module) -> None:
        """Synchronize gradients across all processes using AllReduce."""
        for param in model.parameters():
            if param.grad is not None:
                # torch tensor to numpy array for MPI communication
                grad_np = param.grad.detach().cpu().numpy()
                # AllReduce to average gradients
                avg_grad_np = self.comm.allreduce(grad_np, op=MPI.SUM)
                avg_grad_np /= self.comm.get_world_size()
                # back to torch tensor and update gradient
                param.grad.copy_(torch.from_numpy(avg_grad_np).to(param.device))

    def broadcast_parameters(self, model: torch.nn.Module, root: int = 0) -> None:
        """Broadcast model parameters from root process to all processes."""
        for param in model.parameters():
            param_np = param.data.detach().cpu().numpy()
            param_np = self.comm.bcast(param_np, root=root)
            param.data.copy_(torch.from_numpy(param_np).to(param.device))
