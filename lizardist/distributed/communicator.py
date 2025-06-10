from mpi4py import MPI
import numpy as np


class Communicator:
    def __init__(self) -> None:
        """Initialize MPI communicator."""
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def get_rank(self) -> int:
        """Get the rank of the current process."""
        return self.rank

    def get_world_size(self) -> int:
        """Get the total number of processes."""
        return self.size

    def barrier(self):
        """Synchronize all processes."""
        self.comm.Barrier()

    def allreduce(self, data: np.ndarray, op: MPI.Op = MPI.SUM) -> np.ndarray:
        """Perform AllReduce operation on data."""
        result = np.empty_like(data)
        self.comm.Allreduce(data, result, op=op)
        return result

    def bcast(self, data: np.ndarray, root: int = 0) -> np.ndarray:
        """Broadcast data from root process to all processes."""
        return self.comm.bcast(data, root=root)

    def gather(self, data: np.ndarray, root: int = 0) -> list[np.ndarray]:
        """Gather data from all processes to root process."""
        return self.comm.gather(data, root=root)

    def scatter(self, data: list[np.ndarray], root: int = 0) -> np.ndarray:
        """Scatter data from root process to all processes."""
        return self.comm.scatter(data, root=root)

    def finalize(self) -> None:
        """Finalize MPI communication."""
        MPI.Finalize()
