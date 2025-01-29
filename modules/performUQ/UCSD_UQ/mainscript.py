"""
module: mainscript.py.

This script executes an uncertainty quantification (UQ) analysis based on an input JSON file.
It dynamically selects the appropriate UQ method in the UCSD_UQ engine and executes the corresponding script.

### Functionality:
- Adds the `common` directory to `sys.path` to allow importing necessary modules.
- Parses command-line arguments for necessary paths and parameters.
- Reads and validates the JSON input file.
- Determines the appropriate UQ script to run based on `uqType` and `useApproximation` flags.
- Constructs the command and executes the selected script.
- Captures errors and logs them to `UCSD_UQ.err`.

### Usage:
```sh
python script.py /path/to/working /path/to/template runningLocal driver input.json
```

### Arguments:
1. `working_dir` (Path): Path to the working directory where computations are performed.
2. `template_dir` (Path): Path to the template directory containing the input JSON file.
3. `run_type` (str): Execution mode (`runningLocal` or `runningRemote`).
4. `driver_file` (str): Name of the driver file.
5. `input_file` (Path): Name of the JSON input file.

### Error Handling:
- If an invalid JSON file is provided, the script exits with an error.
- If an unknown UQ type is encountered, the script exits with an error.
- If an exception occurs during execution, the error is logged in `UCSD_UQ.err`.

"""

import argparse
import json
import sys
import traceback
from pathlib import Path

# Dynamically add 'common' directory to sys.path
path_to_common_uq = Path(__file__).resolve().parent.parent / 'common'
if path_to_common_uq not in sys.path:
    sys.path.insert(0, str(path_to_common_uq))


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Run UQ analysis.')
    parser.add_argument('working_dir', type=Path, help='Path to working directory')
    parser.add_argument('template_dir', type=Path, help='Path to template directory')
    parser.add_argument(
        'run_type', choices=['runningLocal', 'runningRemote'], help='Run type'
    )
    parser.add_argument('driver_file', help='Driver file name')
    parser.add_argument('input_file', type=Path, help='Input JSON file')
    return parser.parse_args()


def load_json(file_path):
    """Load and validate JSON input."""
    try:
        with file_path.open(encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        sys.stderr.write(f'ERROR: Invalid JSON format in {file_path}:\n{e}\n')
        sys.exit(1)

    if 'UQ' not in data:
        sys.stderr.write(f"ERROR: Missing 'UQ' key in input JSON: {file_path}\n")
        sys.exit(1)

    return data


def select_uq_script(uq_inputs):
    """Determine which UQ script to run."""
    uq_methods = {
        'Metropolis Within Gibbs Sampler': 'mainscript_hierarchical_bayesian',
    }

    module_name = uq_methods.get(uq_inputs['uqType'], 'mainscript_tmcmc')
    if uq_inputs.get('useApproximation'):
        module_name = 'gp_ab_algorithm'

    try:
        module = __import__(module_name)
    except ImportError:
        sys.stderr.write(
            f'ERROR: Could not import {module_name}. Ensure the script exists.\n'
        )
        sys.exit(1)

    return module.main


def main():
    """Run UQ analysis."""
    args = parse_args()

    # Ensure output error file exists
    err_file = args.working_dir / 'UCSD_UQ.err'
    err_file.touch(exist_ok=True)

    # Remove any previous output files
    (args.working_dir / 'dakotaTab.out').unlink(missing_ok=True)
    (args.working_dir / 'dakotaTabPrior.out').unlink(missing_ok=True)

    # Load JSON input file
    inputs = load_json(args.template_dir / args.input_file)
    uq_inputs = inputs['UQ']

    # Select the appropriate UQ script
    main_function = select_uq_script(uq_inputs)

    # Construct the command
    command = [
        str(args.working_dir),
        str(args.template_dir),
        args.run_type,
        args.driver_file,
        str(args.template_dir / args.input_file),
    ]

    try:
        main_function(command)
    except Exception:
        err_msg = f'ERROR: An exception occurred:\n{traceback.format_exc()}\n'
        with err_file.open('a') as f:
            f.write(err_msg)
        sys.stderr.write(err_msg)


if __name__ == '__main__':
    main()
