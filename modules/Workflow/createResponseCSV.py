# noqa: D100, INP001

#
# Code to write response.csv file given input and dakotaTab.out files
#

# Written fmk, important code copied from whale/main.py
# date: 08/24

import argparse
import json
import os

import numpy as np
import pandas as pd


def main(input_file, dakota_tab_file):  # noqa: D103
    directory_inputs = os.path.dirname(input_file)  # noqa: PTH120
    os.chdir(directory_inputs)

    try:
        # Attempt to open the file
        with open(input_file) as file:  # noqa: PTH123
            data = json.load(file)

    except FileNotFoundError:
        # Handle the error if the file is not found
        print(f"Error createResponseCSV.py: The file '{input_file}' was not found.")  # noqa: T201
        return
    except OSError:
        # Handle other I/O errors
        print(f"Error createResponseCSV.py: Error reading the file '{input_file}'.")  # noqa: T201
        return

    app_data = data.get('Applications', None)
    if app_data is not None:
        dl_data = app_data.get('DL', None)

        if dl_data is not None:
            dl_app_data = dl_data.get('ApplicationData', None)

            if dl_app_data is not None:
                is_coupled = dl_app_data.get('coupled_EDP', None)

    try:
        # sy, abs - added try-statement because dakota-reliability does not write DakotaTab.out
        dakota_out = pd.read_csv(dakota_tab_file, sep=r'\s+', header=0, index_col=0)

        if is_coupled:
            if 'eventID' in dakota_out.columns:
                events = dakota_out['eventID'].values  # noqa: PD011
                events = [int(e.split('x')[-1]) for e in events]
                sorter = np.argsort(events)
                dakota_out = dakota_out.iloc[sorter, :]
                dakota_out.index = np.arange(dakota_out.shape[0])

        dakota_out.to_csv('response.csv')

    except FileNotFoundError:
        # Handle the error if the file is not found
        print(f"Error createResponseCSV.py: The file '{dakota_tab_file}' not found.")  # noqa: T201
        return

    except OSError:
        # Handle other I/O errors
        print(f"Error createResponseCSV.py: Error reading '{dakota_tab_file}'.")  # noqa: T201
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Add arguments with default values
    parser.add_argument(
        '--inputFile', type=str, default='AIM.json', help='Path to the input file)'
    )
    parser.add_argument(
        '--dakotaTab',
        type=str,
        default='dakotaTab.out',
        help='Path to the dakotaTab file)',
    )

    # Parse the arguments
    args = parser.parse_args()

    # Use the arguments
    main(args.inputFile, args.dakotaTab)
