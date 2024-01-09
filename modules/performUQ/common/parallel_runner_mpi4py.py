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
                "Number of processes must be at least 1.                 "
                f"                Got {num_processors}"
            )
        return num_processors

    def get_pool(self) -> MPIPoolExecutor:
        self.pool = MPIPoolExecutor(max_workers=self.num_processors)
        return self.pool

    def close_pool(self) -> None:
        self.pool.shutdown()

        # def run(self, func, iterable, chunksize: int = 1,
        #         unordered: bool = False) -> list:
        #     # try:
        #     #     isinstance(self.pool, MPIPoolExecutor)
        #     # except AttributeError:
        #     #     self.pool = self.get_pool()
        #     return list(self.pool.starmap(fn=func, iterable=iterable,
        #                                 chunksize=chunksize,
        #                                 unordered=unordered))
