"""authors: Mukesh Kumar Ramancha, Maitreya Manoj Kurumbhati, Prof. J.P. Conte, Aakash Bangalore Satish*
affiliation: University of California, San Diego, *SimCenter, University of California, Berkeley

"""

# ======================================================================================================================
import json
import shlex
import sys
from pathlib import Path

path_to_common_uq = Path(__file__).parent.parent / 'common'
sys.path.append(str(path_to_common_uq))


# ======================================================================================================================
def main(input_args):
    # # Initialize analysis
    # path_to_UCSD_UQ_directory = Path(input_args[2]).resolve().parent
    # path_to_working_directory = Path(input_args[3]).resolve()
    # path_to_template_directory = Path(input_args[4]).resolve()
    # run_type = input_args[5]  # either "runningLocal" or "runningRemote"
    # driver_file_name = input_args[6]
    # input_file_name = input_args[7]

    # Initialize analysis
    path_to_UCSD_UQ_directory = Path(input_args[0]).resolve().parent
    path_to_working_directory = Path(input_args[1]).resolve()
    path_to_template_directory = Path(input_args[2]).resolve()
    run_type = input_args[3]  # either "runningLocal" or "runningRemote"
    driver_file_name = input_args[4]
    input_file_name = input_args[5]

    Path('dakotaTab.out').unlink(missing_ok=True)
    Path('dakotaTabPrior.out').unlink(missing_ok=True)

    input_file_full_path = path_to_template_directory / input_file_name

    with open(input_file_full_path, encoding='utf-8') as f:
        inputs = json.load(f)

    uq_inputs = inputs['UQ']
    if uq_inputs['uqType'] == 'Metropolis Within Gibbs Sampler':
        import mainscript_hierarchical_bayesian

        main_function = mainscript_hierarchical_bayesian.main
    else:
        import mainscript_tmcmc

        main_function = mainscript_tmcmc.main

    command = (
        f'"{path_to_working_directory}" "{path_to_template_directory}" '
        f'{run_type} "{driver_file_name}" "{input_file_full_path}"'
    )
    command_list = shlex.split(command)
    main_function(command_list)


# ======================================================================================================================

if __name__ == '__main__':
    input_args = sys.argv
    main(input_args)

# ======================================================================================================================
