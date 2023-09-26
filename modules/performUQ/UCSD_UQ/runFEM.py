"""
authors: Dr. Frank McKenna*, Aakash Bangalore Satish*, Mukesh Kumar Ramancha, Maitreya Manoj Kurumbhati,
and Prof. J.P. Conte
affiliation: SimCenter*; University of California, San Diego

"""

import os
import subprocess
import shutil
import numpy as np
import glob
from typing import Union
from numpy.typing import NDArray
import sys
import traceback


def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            try:
                if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                    shutil.copy2(s, d)
            except Exception as ex:
                msg = f"Could not copy {s}. The following error occurred: \n{ex}"
                return msg
    return "0"


def append_msg_in_out_file(msg, out_file="ops.out"):
    if glob.glob(out_file):
        with open(out_file, "r") as text_file:
            error_FEM = text_file.read()

        startingCharId = error_FEM.lower().find("error")

        if startingCharId >0:
            startingCharId = max(0,startingCharId-20)
            endingID = max(len(error_FEM),startingCharId+200)
            errmsg = error_FEM[startingCharId:endingID]
            errmsg=errmsg.split(" ", 1)[1]
            errmsg=errmsg[0:errmsg.rfind(" ")]
            msg += "\n"
            msg += "your model says...\n"
            msg += "........\n" + errmsg + "\n........ \n"
            msg += "to read more, see " + os.path.join(os.getcwd(), out_file)
    
    return msg


def runFEM(particleNumber, parameterSampleValues, variables, workdirMain, log_likelihood, calibrationData, numExperiments,
           covarianceMatrixList, edpNamesList, edpLengthsList, scaleFactors, shiftFactors, workflowDriver):
    """ 
    this function runs FE model (model.tcl) for each parameter value (par)
    model.tcl should take parameter input
    model.tcl should output 'output$PN.txt' -> column vector of size 'Ny'
    """

    workdirName = ("workdir." + str(particleNumber + 1))
    analysisPath = os.path.join(workdirMain, workdirName)

    if os.path.isdir(analysisPath):
        shutil.rmtree(analysisPath)
    
    os.mkdir(analysisPath)

    # copy templatefiles
    templateDir = os.path.join(workdirMain, "templatedir")
    copytree(templateDir, analysisPath)

    # change to analysis directory
    os.chdir(analysisPath)

    # write input file and covariance multiplier values list
    covarianceMultiplierList = []
    parameterNames = variables["names"]
    with open("params.in", "w") as f:
        f.write('{}\n'.format(len(parameterSampleValues) - len(edpNamesList)))
        for i in range(len(parameterSampleValues)):
            name = str(parameterNames[i])
            value = str(parameterSampleValues[i])
            if name.split('.')[-1] != 'CovMultiplier':
                f.write('{} {}\n'.format(name, value))
            else:
                covarianceMultiplierList.append(parameterSampleValues[i])

    #subprocess.run(workflowDriver, stderr=subprocess.PIPE, shell=True)

    returnCode = subprocess.call(
       workflowDriver,
       shell=True,
       stdout=subprocess.DEVNULL,
       stderr=subprocess.STDOUT,
    )    # subprocess.check_call(workflow_run_command, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)



    # Read in the model prediction
    if os.path.exists('results.out'):
        with open('results.out', 'r') as f:
            prediction = np.atleast_2d(np.genfromtxt(f)).reshape((1, -1))

        os.chdir("../")
        return log_likelihood(calibrationData, prediction, numExperiments, covarianceMatrixList, edpNamesList,
                            edpLengthsList, covarianceMultiplierList, scaleFactors, shiftFactors)
    else:
        os.chdir("../")
        return -np.inf
    

class ModelEvaluationError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class ModelEval:
    def __init__(self, num_rv: int, full_path_of_tmpSimCenter_dir: str, 
                 list_of_dir_names_to_copy_files_from: list[str], 
                 list_of_rv_names: list[str], driver_filename: str, 
                 length_of_results: int,
                 workdir_prefix: str = "workdir",
                 ignore_nans: bool = True) -> None:
        
        self.num_rv = num_rv
        self.full_path_of_tmpSimCenter_dir = full_path_of_tmpSimCenter_dir
        self.list_of_dir_names_to_copy_files_from = list_of_dir_names_to_copy_files_from
        self.list_of_rv_names = list_of_rv_names
        self.driver_filename = driver_filename
        self.length_of_results = length_of_results
        self.workdir_prefix = workdir_prefix
        self.ignore_nans = ignore_nans

        if self.num_rv != len(self.list_of_rv_names):
            raise ModelEvaluationError(f"Error during model specification: Inconsistency between number of rvs ({num_rv = }) and length of list of rv names ({len(list_of_rv_names) = })")
    
    def _check_size_of_sample(self, sample_values: NDArray) -> None:
        num_samples = len(sample_values)
        if num_samples > 1:
            msg = f"Do one simulation at a time. There were {num_samples} samples provided in the sample value {sample_values}."
            raise ModelEvaluationError(msg)
        
        for i in range(num_samples):
            num_values_in_each_sample = len(sample_values[i])
            if num_values_in_each_sample != self.num_rv:
                msg = f"Expected {self.num_rv} values in each sample, found {num_values_in_each_sample} in {sample_values}."
                raise ModelEvaluationError(msg)        
    
    def _create_workdir(self, simulation_number: int) -> str:
        workdir = os.path.join(self.full_path_of_tmpSimCenter_dir, f"{self.workdir_prefix}.{simulation_number + 1}")
        if os.path.exists(workdir):
            for root, dirs, files in os.walk(workdir):
                for file in files:
                    try:
                        os.unlink(os.path.join(root, file))
                    except:
                        msg = f"Could not remove file {file} from {workdir}."
                        raise ModelEvaluationError(msg)
                for dir in dirs:
                    try:
                        shutil.rmtree(os.path.join(root, dir))
                    except:
                        msg = f"Could not remove directory {dir} from {workdir}."
                        raise ModelEvaluationError(msg)

        for src_dir in self.list_of_dir_names_to_copy_files_from:
            src = os.path.join(self.full_path_of_tmpSimCenter_dir, src_dir)
            msg = copytree(src, workdir)
            if msg != "0":
                raise ModelEvaluationError(msg)
        return workdir
    
    def _create_params_file(self, sample_values: NDArray, workdir: str) -> None:
        list_of_strings_to_write = []
        list_of_strings_to_write.append(f"{self.num_rv}")
        for i, rv in enumerate(self.list_of_rv_names):
            list_of_strings_to_write.append(f"{rv} {sample_values[0][i]}")
        try: 
            with open(os.path.join(workdir, "params.in"), "w") as f:
                f.write("\n".join(list_of_strings_to_write))
        except Exception as ex:
            raise ModelEvaluationError(f"Failed to create params.in file in {workdir}. The following error occurred: \n{ex}")

    def _execute_driver_file(self, workdir) -> None:
        command = f"{os.path.join(workdir, self.driver_filename)} 1> model_eval.log 2>&1"
        os.chdir(workdir)
        completed_process = subprocess.run(command, shell=True)
        try:
            completed_process.check_returncode()
        except subprocess.CalledProcessError as ex:
            returnStringList = ["Failed to run the model."]
            returnStringList.append(f"The command to run the model was {ex.cmd}")
            returnStringList.append(f"The return code was {ex.returncode}")
            returnStringList.append(f"The following error occurred: \n{ex}")
            raise ModelEvaluationError(f"\n\n".join(returnStringList))
    
    def _read_outputs_from_results_file(self, workdir) -> NDArray:
        if glob.glob("results.out"):
            outputs = np.loadtxt("results.out").flatten()
        else:
            msg = f"Error running FEM: 'results.out' missing at {workdir}\n"
            msg = append_msg_in_out_file(msg, out_file="ops.out")
            raise ModelEvaluationError(msg)

        if outputs.shape[0] == 0:
            msg = "Error running FEM: 'results.out' is empty\n"
            msg = append_msg_in_out_file(msg, out_file="ops.out")
            raise ModelEvaluationError(msg)
        
        if outputs.shape[0] != self.length_of_results:
            msg = f"Error running FEM: 'results.out' contains {outputs.shape[0]} values, expected to get {self.length_of_results} values\n"
            msg = append_msg_in_out_file(msg, out_file="ops.out")
            raise ModelEvaluationError(msg)

        if not self.ignore_nans:
            if np.isnan(np.sum(outputs)):
                msg = f"Error running FEM: Response value in {workdir} is NaN"
                raise ModelEvaluationError(msg)

        return outputs

    def evaluate_model_once(self, simulation_number: int, sample_values: NDArray) \
        -> Union[str, NDArray]:
        outputs = ""
        try:
            sample_values = np.atleast_2d(sample_values)
            self._check_size_of_sample(sample_values)
            workdir = self._create_workdir(simulation_number)
            self._create_params_file(sample_values, workdir)
            self._execute_driver_file(workdir)
            outputs = self._read_outputs_from_results_file(workdir)
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            outputs = f"\nSimulation number: {simulation_number}\n" + f"Samples values: {sample_values}\n" 
            outputs += "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        finally:
            pass
        return outputs
