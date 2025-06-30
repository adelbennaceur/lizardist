import numpy as np
from mpi4py import MPI


class Communicator:
    def __init__(self, bucket_size: int = 1024 * 1024, use_bucketing: bool = True) -> None:
        """Initialize MPI communicator with optional bucketing support.

        Args:
            bucket_size: Maximum size of each bucket in bytes (only used if use_bucketing=True)
            use_bucketing: Whether to use gradient bucketing
        """
        self.comm: MPI.Comm = MPI.COMM_WORLD
        self.rank: int = self.comm.Get_rank()
        self.size: int = self.comm.Get_size()
        self.bucket_size: int = bucket_size if use_bucketing else 0
        self.use_bucketing: bool = use_bucketing
        self._reset_bucket_stats()

    def _reset_bucket_stats(self) -> None:
        """Reset bucket statistics."""
        self.total_allreduce_calls = 0
        self.total_bytes_sent = 0
        self.total_buckets = 0

    def get_rank(self) -> int:
        """Get the rank of the current process."""
        return self.rank

    def get_world_size(self) -> int:
        """Get the total number of processes."""
        return self.size

    def barrier(self) -> None:
        """Synchronize all processes."""
        self.comm.Barrier()

    def allreduce(self, data: np.ndarray, op: MPI.Op = MPI.SUM) -> np.ndarray:
        """Perform AllReduce operation on data."""
        self.total_allreduce_calls += 1
        self.total_bytes_sent += data.nbytes
        result = np.empty_like(data)
        self.comm.Allreduce(data, result, op=op)
        return result

    def bcast(self, data: np.ndarray, root: int = 0) -> np.ndarray:
        """Broadcast data from root process to all processes."""
        return self.comm.bcast(data, root=root)

    def gather(self, data: np.ndarray, root: int = 0) -> None | list[np.ndarray]:
        """Gather data from all processes to root process."""
        return self.comm.gather(data, root=root)  # type: ignore

    def scatter(self, data: list[np.ndarray], root: int = 0) -> np.ndarray:
        """Scatter data from root process to all processes."""
        return self.comm.scatter(data, root=root)

    def finalize(self) -> None:
        """Finalize MPI communication."""
        MPI.Finalize()

    def get_bucket_stats(self) -> dict[str, int | float]:
        """Get bucketing statistics."""
        return {
            "total_allreduce_calls": self.total_allreduce_calls,
            "total_bytes_sent": self.total_bytes_sent,
            "average_bytes_per_call": self.total_bytes_sent / max(1, self.total_allreduce_calls),
        }

    def bucketed_allreduce(self, tensors: list[np.ndarray], op: MPI.Op = MPI.SUM) -> list[np.ndarray]:
        """Perform bucketed AllReduce on a list of tensors.

        Args:
            tensors: List of numpy arrays to reduce
            op: MPI operation to use (default: MPI.SUM)

        Returns:
            List of reduced numpy arrays
        """
        bucket_size = 1024 * 1024  # 1MB bucket size
        current_bucket: list[np.ndarray] = []
        current_size = 0
        results: list[np.ndarray] = []
        shapes: list[tuple[int, ...]] = []

        for tensor in tensors:
            tensor_size = tensor.nbytes
            if current_size + tensor_size > bucket_size and current_bucket:
                # Process current bucket
                bucket_array = np.concatenate(current_bucket)
                reduced_bucket = self.allreduce(bucket_array, op=op)
                split_idx = 0
                for t, shape in zip(current_bucket, shapes, strict=False):
                    size = t.size
                    results.append(reduced_bucket[split_idx : split_idx + size].reshape(shape))
                    split_idx += size
                current_bucket = []
                current_size = 0
                self.total_buckets += 1

            shapes.append(tensor.shape)
            current_bucket.append(tensor.flatten())
            current_size += tensor_size

        # process remaining tensors in the last bucket
        if current_bucket:
            bucket_array = np.concatenate(current_bucket)
            reduced_bucket = self.allreduce(bucket_array, op=op)
            split_idx = 0
            for t, shape in zip(current_bucket, shapes, strict=False):
                size = t.size
                results.append(reduced_bucket[split_idx : split_idx + size].reshape(shape))
                split_idx += size
            current_bucket = []
            current_size = 0
            self.total_buckets += 1

        return results
