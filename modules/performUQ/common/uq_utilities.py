import glob  # noqa: D100, INP001
import json
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from multiprocessing import get_context
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Union

import numpy as np
import numpy.typing as npt
import quoFEM_RV_models
import scipy.stats
from ERAClasses.ERADist import ERADist
from ERAClasses.ERANataf import ERANataf
from numpy.typing import NDArray

if quoFEM_RV_models not in sys.modules:
    import quoFEM_RV_models


def _copytree(src, dst, symlinks=False, ignore=None) -> str:  # noqa: FBT002
    if not os.path.exists(dst):  # noqa: PTH110
        os.makedirs(dst)  # noqa: PTH103
    for item in os.listdir(src):
        s = os.path.join(src, item)  # noqa: PTH118
        d = os.path.join(dst, item)  # noqa: PTH118
        if os.path.isdir(s):  # noqa: PTH112
            _copytree(s, d, symlinks, ignore)
        else:
            try:
                if (
                    not os.path.exists(d)  # noqa: PTH110
                    or os.stat(s).st_mtime - os.stat(d).st_mtime > 1  # noqa: PTH116
                ):
                    shutil.copy2(s, d)
            except Exception as ex:  # noqa: BLE001
                msg = f'Could not copy {s}. The following error occurred: \n{ex}'
                return msg  # noqa: RET504
    return '0'


def _append_msg_in_out_file(msg, out_file_name: str = 'ops.out') -> str:
    if glob.glob(out_file_name):  # noqa: PTH207
        with open(out_file_name) as text_file:  # noqa: PTH123
            error_FEM = text_file.read()  # noqa: N806

        startingCharId = error_FEM.lower().find('error')  # noqa: N806

        if startingCharId > 0:
            startingCharId = max(0, startingCharId - 20)  # noqa: N806
            endingID = max(len(error_FEM), startingCharId + 200)  # noqa: N806
            errmsg = error_FEM[startingCharId:endingID]
            errmsg = errmsg.split(' ', 1)[1]
            errmsg = errmsg[0 : errmsg.rfind(' ')]
            msg += '\n'
            msg += 'your model says...\n'
            msg += '........\n' + errmsg + '\n........ \n'
            msg += 'to read more, see ' + os.path.join(os.getcwd(), out_file_name)  # noqa: PTH109, PTH118

    return msg


class ModelEvaluationError(Exception):  # noqa: D101
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class SimCenterWorkflowDriver:  # noqa: D101
    def __init__(
        self,
        full_path_of_tmpSimCenter_dir: str,  # noqa: N803
        list_of_dir_names_to_copy_files_from: list[str],  # noqa: FA102
        list_of_rv_names: list[str],  # noqa: FA102
        driver_filename: str,
        length_of_results: int,
        workdir_prefix: str = 'workdir',
        ignore_nans: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        self.full_path_of_tmpSimCenter_dir = full_path_of_tmpSimCenter_dir
        self.list_of_dir_names_to_copy_files_from = (
            list_of_dir_names_to_copy_files_from
        )
        self.list_of_rv_names = list_of_rv_names
        self.driver_filename = driver_filename
        self.length_of_results = length_of_results
        self.workdir_prefix = workdir_prefix
        self.ignore_nans = ignore_nans

        self.num_rv = len(self.list_of_rv_names)

    def _check_size_of_sample(self, sample_values: NDArray) -> None:
        num_samples = len(sample_values)
        if num_samples > 1:
            msg = f'Do one simulation at a time. There were {num_samples} samples provided in the sample value {sample_values}.'
            raise ModelEvaluationError(msg)

        for i in range(num_samples):
            num_values_in_each_sample = len(sample_values[i])
            if num_values_in_each_sample != self.num_rv:
                msg = f'Expected {self.num_rv} values in each sample, found {num_values_in_each_sample} in {sample_values}.'
                raise ModelEvaluationError(msg)

    def _create_workdir(self, simulation_number: int) -> str:
        workdir = os.path.join(  # noqa: PTH118
            self.full_path_of_tmpSimCenter_dir,
            f'{self.workdir_prefix}.{simulation_number + 1}',
        )
        if os.path.exists(workdir):  # noqa: PTH110
            for root, dirs, files in os.walk(workdir):
                try:
                    for file in files:
                        os.chmod(os.path.join(root, file), 0o777)  # noqa: S103, PTH101, PTH118
                        os.unlink(os.path.join(root, file))  # noqa: PTH108, PTH118
                except Exception as ex:
                    msg = f'Could not remove file {file} from {workdir}.'
                    raise ModelEvaluationError(msg) from ex
                try:
                    for directory in dirs:
                        shutil.rmtree(os.path.join(root, directory))  # noqa: PTH118
                except Exception as ex:
                    msg = f'Could not remove directory {directory} from {workdir}.'
                    raise ModelEvaluationError(msg) from ex

        for src_dir in self.list_of_dir_names_to_copy_files_from:
            src = os.path.join(self.full_path_of_tmpSimCenter_dir, src_dir)  # noqa: PTH118
            msg = _copytree(src, workdir)
            if msg != '0':
                raise ModelEvaluationError(msg)
        return workdir

    def _create_params_file(self, sample_values: NDArray, workdir: str) -> None:
        list_of_strings_to_write = []
        list_of_strings_to_write.append(f'{self.num_rv}')
        for i, rv in enumerate(self.list_of_rv_names):
            list_of_strings_to_write.append(f'{rv} {sample_values[0][i]}')
        try:
            with open(os.path.join(workdir, 'params.in'), 'w') as f:  # noqa: PTH118, PTH123
                f.write('\n'.join(list_of_strings_to_write))
        except Exception as exc:
            msg = f'Failed to create params.in file in {workdir}. The following error occurred: \n{exc}'
            raise ModelEvaluationError(msg) from exc

    def _execute_driver_file(self, workdir: str) -> None:
        command = (
            f'{os.path.join(workdir, self.driver_filename)} 1> model_eval.log 2>&1'  # noqa: PTH118
        )
        os.chdir(workdir)
        completed_process = subprocess.run(command, shell=True, check=False)  # noqa: S602
        try:
            completed_process.check_returncode()
        except subprocess.CalledProcessError as ex:
            returnStringList = [f'Failed to run the model in {workdir}.']  # noqa: N806
            returnStringList.append(f'The command to run the model was {ex.cmd}')
            returnStringList.append(f'The return code was {ex.returncode}')
            returnStringList.append(f'The following error occurred: \n{ex}')
            raise ModelEvaluationError('\n\n'.join(returnStringList))  from ex # noqa: B904

    def _read_outputs_from_results_file(self, workdir: str) -> NDArray:
        if glob.glob('results.out'):  # noqa: PTH207
            outputs = np.loadtxt('results.out', dtype=float).flatten()
        else:
            msg = f"Error running FEM: 'results.out' missing in {workdir}\n"
            msg = _append_msg_in_out_file(msg, out_file_name='ops.out')
            raise ModelEvaluationError(msg)

        if outputs.shape[0] == 0:
            msg = f"Error running FEM: in {workdir}, 'results.out' is empty\n"
            msg = _append_msg_in_out_file(msg, out_file_name='ops.out')
            raise ModelEvaluationError(msg)

        if outputs.shape[0] != self.length_of_results:
            msg = f"Error running FEM: in {workdir}, 'results.out' contains {outputs.shape[0]} values, expected to get {self.length_of_results} values\n"
            msg = _append_msg_in_out_file(msg, out_file_name='ops.out')
            raise ModelEvaluationError(msg)

        if not self.ignore_nans:
            if np.isnan(np.sum(outputs)):
                msg = f"Error running FEM: 'results.out' in {workdir} contains NaN"
                raise ModelEvaluationError(msg)

        return outputs

    def evaluate_model_once(  # noqa: D102
        self, simulation_number: int, sample_values: NDArray
    ) -> Union[str, NDArray]:  # noqa: FA100
        outputs = ''
        try:
            sample_values = np.atleast_2d(sample_values)
            self._check_size_of_sample(sample_values)
            workdir = self._create_workdir(simulation_number)
            self._create_params_file(sample_values, workdir)
            self._execute_driver_file(workdir)
            outputs = self._read_outputs_from_results_file(workdir)
        except ModelEvaluationError as model_err:
            # Custom error handling for ModelEvaluationError
            msg = f'\nIn workdir: {workdir}\n' f'Error: {model_err!s}\n'
            raise ModelEvaluationError(msg) from model_err
        except Exception as exc:
            # Catch all other exceptions, show full traceback and message
            exc_type, exc_value, exc_traceback = sys.exc_info()
            formatted_traceback = ''.join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )
            msg = (
                f'\nIn workdir: {workdir}\n'
                f'Encountered an error:\n{exc}\n\n'
                f'Traceback:\n{formatted_traceback}'
            )
            raise RuntimeError(msg) from exc  # Chain RuntimeError with traceback
        finally:
            os.chdir(self.full_path_of_tmpSimCenter_dir)
        return outputs


class ParallelRunnerMultiprocessing:  # noqa: D101
    def __init__(self, run_type: str = 'runningLocal') -> None:
        self.run_type = run_type
        self.num_processors = self.get_num_processors()
        self.pool = self.get_pool()

    def get_num_processors(self) -> int:  # noqa: D102
        num_processors = os.cpu_count()
        max_num_processors = 32  # max number of processors to use in multiprocessing when running locally
        if num_processors is None:
            num_processors = 1
        elif num_processors < 1:
            raise ValueError(  # noqa: TRY003
                'Number of processes must be at least 1.                     '  # noqa: EM102
                f'         Got {num_processors}'
            )
        elif num_processors > max_num_processors:
            # this is to get past memory problems when running large number processors in a container
            num_processors = max_num_processors

        return num_processors

    def get_pool(self) -> Pool:  # noqa: D102
        context = get_context('spawn')
        self.pool = context.Pool(processes=self.num_processors)
        return self.pool

    def run(self, func, job_args):  # noqa: D102
        return self.pool.starmap(func, job_args)

    def close_pool(self):  # noqa: D102
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None  # optional but safe


def make_ERADist_object(name, opt, val) -> ERADist:  # noqa: N802, D103
    return ERADist(name=name, opt=opt, val=val)


def create_one_marginal_distribution(rv_data) -> ERADist:  # noqa: D103
    try:
        # Access the distribution and input type dynamically without eval
        distribution_model = getattr(
            quoFEM_RV_models, rv_data['distribution'] + rv_data['inputType']
        )
        rv = distribution_model.model_validate(rv_data)

        # Create the ERADist object
        return make_ERADist_object(name=rv.ERAName, opt=rv.ERAOpt, val=rv.ERAVal)

    except AttributeError as e:
        msg = f'Invalid distribution or input type: {e}'
        raise AttributeError(msg) from e
    except TypeError as e:
        msg = f'model_validate does not accept the provided rv_data: {e}'
        raise TypeError(msg) from e


def make_list_of_marginal_distributions(  # noqa: D103
    list_of_random_variables_data,
) -> list[ERADist]:  # noqa: FA102
    marginal_ERAdistribution_objects_list = []  # noqa: N806
    for rv_data in list_of_random_variables_data:
        marginal_ERAdistribution_objects_list.append(  # noqa: PERF401
            create_one_marginal_distribution(rv_data)
        )
    return marginal_ERAdistribution_objects_list


def make_correlation_matrix(correlation_matrix_data, num_rvs) -> NDArray:  # noqa: D103
    return np.atleast_2d(correlation_matrix_data).reshape((num_rvs, num_rvs))


def make_ERANataf_object(list_of_ERADist, correlation_matrix) -> ERANataf:  # noqa: N802, N803, D103
    return ERANataf(M=list_of_ERADist, Correlation=correlation_matrix)


class ERANatafJointDistribution:  # noqa: D101
    def __init__(
        self,
        list_of_random_variables_data: list,
        correlation_matrix_data: NDArray,
    ) -> None:
        self.list_of_random_variables_data = list_of_random_variables_data
        self.correlation_matrix_data = correlation_matrix_data

        self.num_rvs = len(self.list_of_random_variables_data)
        self.correlation_matrix = make_correlation_matrix(
            self.correlation_matrix_data, self.num_rvs
        )
        self.marginal_ERAdistribution_objects_list = (
            make_list_of_marginal_distributions(self.list_of_random_variables_data)
        )
        self.ERANataf_object = make_ERANataf_object(
            self.marginal_ERAdistribution_objects_list, self.correlation_matrix
        )

    def u_to_x(  # noqa: D102
        self,
        u: NDArray,
        jacobian: bool = False,  # noqa: FBT001, FBT002
    ) -> Union[tuple[NDArray[np.float64], Any], NDArray[np.float64]]:  # noqa: FA100, FA102
        return self.ERANataf_object.U2X(U=u, Jacobian=jacobian)

    def x_to_u(  # noqa: D102
        self,
        x: NDArray,
        jacobian: bool = False,  # noqa: FBT001, FBT002
    ) -> Union[  # noqa: FA100
        tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]],  # noqa: FA102
        NDArray[np.floating[Any]],
    ]:
        return self.ERANataf_object.X2U(X=x, Jacobian=jacobian)

    def pdf(self, x: NDArray) -> Union[Any, NDArray[np.float64]]:  # noqa: FA100, D102
        return self.ERANataf_object.pdf(X=x)

    def logpdf(self, x: NDArray) -> NDArray[np.float64]:  # noqa: D102
        return np.log(self.pdf(x))

    def cdf(self, x: NDArray) -> float:  # noqa: D102
        return self.ERANataf_object.cdf(X=x)

    def random(  # noqa: D102
        self,
        list_of_rngs: list[np.random.Generator] = [],  # noqa: B006, FA102
        n: int = 1,
    ) -> Union[tuple[NDArray[np.float64], Any], NDArray[np.float64]]:  # noqa: FA100, FA102
        if list_of_rngs == []:
            list_of_rngs = [
                np.random.default_rng(seed=i)
                for i in range(len(self.marginal_ERAdistribution_objects_list))
            ]
        u = np.zeros((n, len(list_of_rngs)))
        for i, rng in enumerate(list_of_rngs):
            u[:, i] = rng.normal(size=n)
        return self.u_to_x(u)


def get_list_of_pseudo_random_number_generators(entropy, num_spawn):  # noqa: D103
    seed_sequence = np.random.SeedSequence(entropy=entropy).spawn(num_spawn)
    prngs = [np.random.Generator(np.random.PCG64DXSM(s)) for s in seed_sequence]
    return prngs  # noqa: RET504


def get_parallel_pool_instance(run_type: str):  # noqa: D103
    if run_type == 'runningRemote':
        from parallel_runner_mpi4py import ParallelRunnerMPI4PY

        return ParallelRunnerMPI4PY(run_type)
    else:  # noqa: RET505
        return ParallelRunnerMultiprocessing(run_type)


def make_list_of_rv_names(all_rv_data):  # noqa: D103
    list_of_rv_names = []
    for rv_data in all_rv_data:
        list_of_rv_names.append(rv_data['name'])  # noqa: PERF401
    return list_of_rv_names


def get_length_of_results(edp_data):  # noqa: D103
    length_of_results = 0
    for edp in edp_data:
        length_of_results += int(float(edp['length']))
    return length_of_results


def create_default_model(  # noqa: D103
    run_directory,
    list_of_dir_names_to_copy_files_from,
    list_of_rv_names,
    driver_filename,
    length_of_results,
    workdir_prefix,
):
    model = SimCenterWorkflowDriver(
        full_path_of_tmpSimCenter_dir=run_directory,
        list_of_dir_names_to_copy_files_from=list_of_dir_names_to_copy_files_from,
        list_of_rv_names=list_of_rv_names,
        driver_filename=driver_filename,
        length_of_results=length_of_results,
        workdir_prefix=workdir_prefix,
    )
    return model  # noqa: RET504


def get_default_model_evaluation_function(model):  # noqa: D103
    return model.evaluate_model_once


def get_ERANataf_joint_distribution_instance(  # noqa: N802, D103
    list_of_rv_data,
    correlation_matrix_data,
):
    joint_distribution = ERANatafJointDistribution(
        list_of_rv_data, correlation_matrix_data
    )
    return joint_distribution  # noqa: RET504


def get_std_normal_to_rv_transformation_function(joint_distribution):  # noqa: D103
    transformation_function = joint_distribution.u_to_x
    return transformation_function  # noqa: RET504


def get_default_model(  # noqa: D103
    list_of_rv_data,
    edp_data,
    list_of_dir_names_to_copy_files_from,
    run_directory,
    driver_filename='driver',
    workdir_prefix='workdir',
):
    list_of_rv_names = make_list_of_rv_names(list_of_rv_data)
    length_of_results = get_length_of_results(edp_data)
    list_of_dir_names_to_copy_files_from = list_of_dir_names_to_copy_files_from  # noqa: PLW0127
    driver_filename = driver_filename  # noqa: PLW0127
    workdir_prefix = workdir_prefix  # noqa: PLW0127

    model = create_default_model(
        run_directory,
        list_of_dir_names_to_copy_files_from,
        list_of_rv_names,
        driver_filename,
        length_of_results,
        workdir_prefix,
    )
    return model  # noqa: RET504


def model_evaluation_function(  # noqa: D103
    func,
    list_of_iterables,
):
    return func(*list_of_iterables)


def get_random_number_generators(entropy, num_prngs):  # noqa: D103
    return get_list_of_pseudo_random_number_generators(entropy, num_prngs)


def get_standard_normal_random_variates(list_of_prngs, size=1):  # noqa: D103
    return [prng.standard_normal(size=size) for prng in list_of_prngs]


def get_inverse_gamma_random_variate(prng, shape, scale, size=1):  # noqa: D103
    return scipy.stats.invgamma.rvs(shape, scale=scale, size=size, random_state=prng)


def multivariate_normal_logpdf(x, mean, cov):  # noqa: D103
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    logdet = np.sum(np.log(eigenvalues))
    valsinv = 1.0 / eigenvalues
    U = eigenvectors * np.sqrt(valsinv)  # noqa: N806
    dim = len(eigenvalues)
    dev = x - mean
    maha = np.square(dev.T @ U).sum()
    log2pi = np.log(2 * np.pi)
    return -0.5 * (dim * log2pi + maha + logdet)


@dataclass
class NormalInverseWishartParameters:  # noqa: D101
    mu_vector: npt.NDArray
    lambda_scalar: float
    nu_scalar: float
    psi_matrix: npt.NDArray


@dataclass
class InverseGammaParameters:  # noqa: D101
    alpha_scalar: float
    beta_scalar: float

    def _to_shape_and_scale(self):
        return (self.alpha_scalar, 1 / self.beta_scalar)


def _get_tabular_results_file_name_for_dataset(
    tabular_results_file_base_name,
    dataset_number,
):
    tabular_results_parent = tabular_results_file_base_name.parent
    tabular_results_stem = tabular_results_file_base_name.stem
    tabular_results_extension = tabular_results_file_base_name.suffix

    tabular_results_file = (
        tabular_results_parent
        / f'{tabular_results_stem}_dataset_{dataset_number + 1}{tabular_results_extension}'
    )
    return tabular_results_file  # noqa: RET504


def _write_to_tabular_results_file(tabular_results_file, string_to_write):
    with tabular_results_file.open('a') as f:
        f.write(string_to_write)


class Ensure2DOutputShape:
    def __init__(self, func, expected_dim, label="wrapped_function"):
        self.func = func
        self.expected_dim = expected_dim
        self.label = label
        self._last_input_shape = None
        self._last_output_shape = None

    def __call__(self, *args):
        result = self.func(*args)
        result = np.asarray(result)

        # Handle scalar return (e.g., float) → reshape to (1, 1)
        if result.ndim == 0:
            result = result.reshape(1, 1)

        # Treating the last argument of the function call as the sample
        x = np.asarray(args[-1])
        n_samples = x.shape[0] if x.ndim > 1 else 1

        input_shape = x.shape
        output_shape = result.shape
        if input_shape == self._last_input_shape and output_shape == self._last_output_shape:
            return result

        if result.ndim == 1:
            if n_samples == 1 and result.shape[0] == self.expected_dim:
                result = result.reshape(1, self.expected_dim)
            elif self.expected_dim == 1 and result.shape[0] == n_samples:
                result = result.reshape(n_samples, 1)
            else:
                raise ValueError(
                    f"[{self.label}] 1D output shape {result.shape} is incompatible with input {x.shape}. "
                    f"Expected ({n_samples}, {self.expected_dim})"
                )

        elif result.ndim == 2:
            if result.shape != (n_samples, self.expected_dim):
                raise ValueError(
                    f"[{self.label}] 2D output shape {result.shape} does not match expected ({n_samples}, {self.expected_dim})"
                )

        else:
            raise ValueError(
                f"[{self.label}] Output has {result.ndim} dimensions; expected 1D or 2D"
            )

        self._last_input_shape = input_shape
        self._last_output_shape = result.shape
        return result


def safe_evaluate_model_for_gp_ab(sim_number: int, x: np.ndarray, model_callable, logger=None):
    """
    A safe wrapper to evaluate the model. Returns (x, y, msg) where msg is None if success.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None, str | None]
        The input, output (or None), and error message (or None).
    """
    try:
        y = model_callable(sim_number, x)
        return x, y, None
    except (ModelEvaluationError, Exception) as e:
        msg = f"Model evaluation failed at sim {sim_number}, x={x.tolist()}:\n{str(e)}"
        if logger:
            logger.warning(msg)
        return x, None, msg


def log_failed_points_to_file(
    failed: list[tuple[np.ndarray, str]],
    iteration: int,
    logger=None,
    output_dir: Path = Path("results"),
):
    """
    Save failed input points and messages to a JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"failed_model_inputs_iter_{iteration}.json"

    out_data = [{"input": x.tolist(), "message": msg} for x, msg in failed]

    with out_path.open("w") as f:
        json.dump(make_json_serializable(out_data), f, indent=4)

    if logger:
        logger.info(f"Saved {len(failed)} failed inputs to: {out_path}")


def make_json_serializable(obj):
    """
    Recursively convert Python, NumPy, and common custom types to JSON-serializable formats.

    Supports: dict, list, tuple, np.ndarray, int, float, bool, str, None,
              pathlib.Path, enum.Enum, and custom objects via __str__ fallback.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, Path):
        return str(obj)
    elif obj is None:
        return None
    else:
        try:
            return str(obj)
        except Exception:
            msg = f'Object of type {type(obj)} is not JSON serializable and cannot be converted with str(): {obj}'
            raise TypeError(msg)
