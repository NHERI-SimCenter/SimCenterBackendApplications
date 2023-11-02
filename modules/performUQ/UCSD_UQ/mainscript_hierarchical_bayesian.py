import sys
from pathlib import Path
import os
import json


def main(input_args):
    # Initialize analysis
    mainscript_path = Path(input_args[0]).resolve()
    workdir_main = Path(input_args[1]).resolve()
    workdir_template = Path(input_args[2]).resolve()
    run_type = input_args[3]  # either "runningLocal" or "runningRemote"
    workflow_driver = input_args[4]
    input_file = input_args[5]

    try:
        os.remove("dakotaTab.out")
        os.remove("dakotaTabPrior.out")
    except OSError:
        pass

    with open(input_file, "r") as f:
        inputs = json.load(f)
    
    applications_inputs = inputs["Applications"]
    edp_inputs = inputs["EDP"]
    fem_inputs = inputs["FEM"]
    uq_inputs = inputs["UQ"]
    correlation_matrix_inputs = inputs["correlationMatrix"]
    local_applications_directory = inputs["localAppDir"]
    rv_inputs = inputs["randomVariables"]
    remote_applications_directory = inputs["remoteAppDir"]
    # run_type = inputs["runType"]
    working_directory = inputs["workingDir"]


    

    print(
        f"{mainscript_path = }\n{workdir_main = }\n{workdir_template = }\n{run_type = }\n{workflow_driver = }\n{input_file = }"
    )


if __name__ == "__main__":
    input_args = sys.argv
    main(input_args)
