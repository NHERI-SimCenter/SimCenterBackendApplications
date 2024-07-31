import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Literal, Union

import numpy as np

path_to_common_uq = Path(__file__).parent.parent / 'common'
sys.path.append(str(path_to_common_uq))
import uq_utilities

InputsType = tuple[
    Path,
    Path,
    Literal['runningLocal', 'runningRemote'],
    Path,
    dict,
]


class CommandLineArguments:
    working_directory_path: Path
    template_directory_path: Path
    run_type: Union[Literal['runningLocal'], Literal['runningRemote']]
    driver_file: Path
    input_file: Path


def _handle_arguments(
    command_line_arguments: CommandLineArguments,
) -> InputsType:
    working_directory_path = command_line_arguments.working_directory_path
    template_directory_path = command_line_arguments.template_directory_path
    run_type = command_line_arguments.run_type
    driver_file = command_line_arguments.driver_file
    input_file = command_line_arguments.input_file
    with open(input_file, 'r') as f:
        inputs = json.load(f)
    return (
        working_directory_path,
        template_directory_path,
        run_type,
        driver_file,
        inputs,
    )


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Preprocess the inputs to the hierarchical Bayesian updating'
            ' algorithm'
        )
    )
    parser.add_argument(
        'working_directory_path',
        help=(
            'path to the working directory where the analysis will be' ' conducted'
        ),
        type=Path,
    )
    parser.add_argument(
        'template_directory_path',
        help=(
            'path to the template directory containing the model, data, and'
            ' any other files required for the analysis'
        ),
        type=Path,
    )
    parser.add_argument(
        'run_type',
        help=(
            'string indicating whether the analysis is being run locally or'
            " remotely on DesignSafe's computing infrastructure"
        ),
        type=str,
    )
    parser.add_argument(
        'driver_file',
        help=(
            'path to the driver file containing the commands to perform one'
            ' evaluation of the model'
        ),
        type=Path,
    )
    parser.add_argument(
        'input_file',
        help=(
            'path to the JSON file containing the user provided inputs to run'
            ' the hierarchical Bayesian analysis'
        ),
        type=Path,
    )
    return parser


def _print_start_message(demarcation_string: str = '=', start_space: str = ''):
    msg = f"'{Path(__file__).name}' started running"
    print()
    print(start_space + demarcation_string * len(msg))
    print(start_space + msg)
    print()


def _print_end_message(demarcation_string: str = '=', start_space: str = ''):
    msg = f"'{Path(__file__).name}' finished running"
    print()
    print(start_space + msg)
    print(start_space + demarcation_string * len(msg))


def main(arguments: InputsType):
    (
        working_directory_path,
        template_directory_path,
        run_type,
        driver_file,
        inputs,
    ) = arguments
    # applications_inputs = inputs["Applications"]
    edp_inputs = inputs['EDP']
    # fem_inputs = inputs["FEM"]
    uq_inputs = inputs['UQ']
    correlation_matrix_inputs = inputs['correlationMatrix']
    # local_applications_directory = inputs["localAppDir"]
    rv_inputs = inputs['randomVariables']
    # remote_applications_directory = inputs["remoteAppDir"]
    # run_type = inputs["runType"]
    # working_directory = inputs["workingDir"]

    # path_to_common_uq = main_script_directory.parent / "common"
    # sys.path.append(str(path_to_common_uq))

    joint_distribution = uq_utilities.ERANatafJointDistribution(
        rv_inputs,
        correlation_matrix_inputs,
    )
    # joint_distribution = uq_utilities.get_ERANataf_joint_distribution_instance(
    #     list_of_rv_data=rv_inputs,
    #     correlation_matrix_data=correlation_matrix_inputs,
    # )

    num_rv = len(rv_inputs)
    num_edp = len(edp_inputs)
    list_of_dataset_subdirs = uq_inputs['List Of Dataset Subdirectories']
    calibration_data_file_name = uq_inputs['Calibration Data File Name']

    list_of_models = []
    list_of_model_evaluation_functions = []
    list_of_datasets = []
    list_of_dataset_lengths = []

    for sample_number, dir_name_string in enumerate(list_of_dataset_subdirs):
        dir_name_string = list_of_dataset_subdirs[sample_number]
        dir_name = Path(dir_name_string).stem
        source_dir_name = template_directory_path / dir_name
        destination_dir_name = working_directory_path / dir_name
        shutil.move(source_dir_name, destination_dir_name)
        list_of_dir_names_to_copy_files_from = [
            str(template_directory_path),
            str(destination_dir_name),
        ]
        edp_data = edp_inputs[sample_number]
        data = np.genfromtxt(
            destination_dir_name / calibration_data_file_name, dtype=float
        )
        list_of_datasets.append(data)
        list_of_dataset_lengths.append(edp_data['length'])

        model = uq_utilities.get_default_model(
            list_of_rv_data=rv_inputs,
            edp_data=[edp_data],
            list_of_dir_names_to_copy_files_from=list_of_dir_names_to_copy_files_from,
            run_directory=working_directory_path,
            driver_filename=str(driver_file),
            workdir_prefix=f'{dir_name}.workdir',
        )

        list_of_models.append(model)
        list_of_model_evaluation_functions.append(
            uq_utilities.get_default_model_evaluation_function(model)
        )

    parallel_pool = uq_utilities.get_parallel_pool_instance(run_type)
    # parallel_evaluation_function = parallel_pool.run
    function_to_evaluate = uq_utilities.model_evaluation_function

    restart_file_name = Path(uq_inputs['Restart File Name']).name
    restart_file_path = template_directory_path / restart_file_name
    if not restart_file_path.is_file():
        restart_file_path = None

    return (
        parallel_pool,
        function_to_evaluate,
        joint_distribution,
        num_rv,
        num_edp,
        list_of_model_evaluation_functions,
        list_of_datasets,
        list_of_dataset_lengths,
        restart_file_path,
    )


def _parse_arguments(args) -> InputsType:
    parser = _create_parser()
    command_line_arguments = CommandLineArguments()
    parser.parse_args(args=args, namespace=command_line_arguments)
    arguments = _handle_arguments(command_line_arguments)
    return arguments


def preprocess_arguments(args):
    arguments = _parse_arguments(args)
    return main(arguments=arguments)


if __name__ == '__main__':
    _print_start_message()
    (
        parallel_evaluation_function,
        function_to_evaluate,
        joint_distribution,
        num_rv,
        num_edp,
        list_of_model_evaluation_functions,
        list_of_datasets,
        list_of_dataset_lengths,
        restart_file_path,
    ) = preprocess_arguments(sys.argv)
    _print_end_message()
