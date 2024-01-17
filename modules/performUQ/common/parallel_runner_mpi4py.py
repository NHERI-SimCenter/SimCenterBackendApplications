from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor


class ParallelRunnerMPI4PY:
    def __init__(self, run_type: str = "runningRemote") -> None:
        self.run_type = run_type
        self.comm = MPI.COMM_WORLD
        self.num_processors = self.get_num_processors()
        self.pool = self.get_pool()

    def get_num_processors(self) -> int:
        num_processors = self.comm.Get_size()
        if num_processors is None:
            num_processors = 1
        if num_processors < 1:
            raise ValueError(
                "Number of processes must be at least 1. Got {num_processors}"
            )
        return num_processors

    def get_pool(self) -> MPIPoolExecutor:
        self.pool = MPIPoolExecutor(max_workers=self.num_processors)
        return self.pool

    def close_pool(self) -> None:
        self.pool.shutdown()
