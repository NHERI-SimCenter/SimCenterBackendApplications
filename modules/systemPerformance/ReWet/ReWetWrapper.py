
# fmk's greatest python application!
# licencse: BSD 3

import argparse, json
import importlib

def runReWet(asset_type, input_dir, running_parallel):

    print('About to run ReWet: \n\t asset_type', asset_type, '\n\t inputDIR: ', input_dir, '\n\t running_parallel: ', running_parallel)

    #
    # import mpi only if running parallel as we do not require install for local runs
    #

    numP = 1
    procID = 0
    if (running_parallel == True):
        mpi_spec = importlib.util.find_spec("mpi4py")
        found = mpi_spec is not None
        if found:
            import mpi4py
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        numP = self.comm.Get_size()
        procID = self.comm.Get_rank();
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--asset_type')
    parser.add_argument('--input_dir')
    parser.add_argument('--running_parallel', default="False")    
    args = parser.parse_args()    
    
    runReWet(args.asset_type, args.input_dir, args.running_parallel)
