import sys
from pathlib import Path
import os


def main(input_args):
    # Initialize analysis
    mainscript_path = Path(input_args[0]).resolve()
    workdir_main = Path(input_args[1]).resolve()
    workdir_template = Path(input_args[2]).resolve()
    run_type = input_args[3]  # either "runningLocal" or "runningRemote"
    workflow_driver = input_args[4]
    input_file = input_args[5]

    try:
        os.remove('dakotaTab.out')
        os.remove('dakotTabPrior.out')
    except OSError:
        pass

    print(f"{mainscript_path = }\n{workdir_main = }\n{workdir_template = }\n{run_type = }\n{workflow_driver = }\n{input_file = }")

if __name__ == "__main__":
    input_args = sys.argv
    main(input_args)
