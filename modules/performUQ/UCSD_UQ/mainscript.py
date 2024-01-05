"""
authors: Mukesh Kumar Ramancha, Maitreya Manoj Kurumbhati, Prof. J.P. Conte, Aakash Bangalore Satish*
affiliation: University of California, San Diego, *SimCenter, University of California, Berkeley

"""

# ======================================================================================================================
import sys
import json
import platform
from pathlib import Path
import subprocess


# ======================================================================================================================
def main(input_args):

    # Initialize analysis
    path_to_UCSD_UQ_directory = Path(input_args[0]).resolve().parent
    path_to_working_directory = Path(input_args[1]).resolve()
    path_to_template_directory = Path(input_args[2]).resolve()
    run_type = input_args[3]  # either "runningLocal" or "runningRemote"
    driver_file_name = input_args[4]
    input_file_name = input_args[5]

    Path("dakotaTab.out").unlink(missing_ok=True)
    Path("dakotaTabPrior.out").unlink(missing_ok=True)

    with open(input_file_name, "r") as f:
        inputs = json.load(f)

    uq_inputs = inputs["UQ"]
    if uq_inputs["uqType"] == "Metropolis Within Gibbs Sampler":
        main_script = (
            path_to_UCSD_UQ_directory / "mainscript_hierarchical_bayesian.py"
        )
    else:
        main_script = path_to_UCSD_UQ_directory / "mainscript_tmcmc.py"

    if platform.system() == "Windows":
        python_command = "python"
    else:
        python_command = "python3"

    command = (
        f"'{python_command}' '{main_script}' "
        f"'{path_to_working_directory}' '{path_to_template_directory}' "
        f"{run_type} {driver_file_name} {input_file_name}"
    )
    print(command)
    try:
        result = subprocess.check_output(
            command, stderr=subprocess.STDOUT, shell=True
        )
        returnCode = 0
    except subprocess.CalledProcessError as e:
        result = e.output
        returnCode = e.returncode


# ======================================================================================================================

if __name__ == "__main__":
    input_args = sys.argv
    main(input_args)

# ======================================================================================================================
