# Import packages needed for setting up required packages:  # noqa: INP001, D100
import importlib.metadata
import subprocess
import sys

# If not installed, install BRAILS, argparse, and requests:
required = {'BRAILS', 'argparse', 'requests'}
installed = set()
for x in importlib.metadata.distributions():
    try:  # noqa: SIM105
        installed.add(x.name)
    except:  # noqa: S110, PERF203, E722
        pass
missing = required - installed

python = sys.executable
if missing:
    print('\nInstalling packages required for running this widget...')  # noqa: T201
    subprocess.check_call(  # noqa: S603
        [python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL
    )
    print('Successfully installed the required packages')  # noqa: T201

# If BRAILS was previously installed ensure it is the latest version:
import requests  # noqa: E402

latestBrailsVersion = requests.get('https://pypi.org/pypi/BRAILS/json').json()[  # noqa: S113, N816
    'info'
]['version']
if importlib.metadata.version('BRAILS') != latestBrailsVersion:
    print(  # noqa: T201
        '\nAn older version of BRAILS was detected. Updating to the latest BRAILS version..'
    )
    subprocess.check_call(  # noqa: S603
        [python, '-m', 'pip', 'install', 'BRAILS', '-U'], stdout=subprocess.DEVNULL
    )
    print('Successfully installed the latest version of BRAILS')  # noqa: T201

# Import packages required for running the latest version of BRAILS:
import argparse  # noqa: E402
import os  # noqa: E402
from time import gmtime, strftime  # noqa: E402

from brails.TranspInventoryGenerator import TranspInventoryGenerator  # noqa: E402


def str2bool(v):  # noqa: ANN001, ANN201, D103
    # courtesy of Maxim @ stackoverflow

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):  # noqa: RET505
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')  # noqa: EM101, TRY003


# Define a standard way of printing program outputs:
def log_msg(msg):  # noqa: ANN001, ANN201, D103
    formatted_msg = '{} {}'.format(strftime('%Y-%m-%dT%H:%M:%SZ', gmtime()), msg)
    print(formatted_msg)  # noqa: T201


# Define a way to call BRAILS TranspInventoryGenerator:
def runBrails(  # noqa: ANN201, N802, D103, PLR0913
    latMin,  # noqa: ANN001, N803
    latMax,  # noqa: ANN001, N803
    longMin,  # noqa: ANN001, N803
    longMax,  # noqa: ANN001, N803
    minimumHAZUS,  # noqa: ANN001, N803
    maxRoadLength,  # noqa: ANN001, N803
    lengthUnit,  # noqa: ANN001, N803
):
    # Initialize TranspInventoryGenerator:
    invGenerator = TranspInventoryGenerator(  # noqa: N806
        location=(longMin, latMin, longMax, latMax)
    )

    # Run TranspInventoryGenerator to generate an inventory for the entered location:
    invGenerator.generate()

    # Combine and format the generated inventory to SimCenter transportation network inventory json format
    invGenerator.combineAndFormat_HWY(
        minimumHAZUS=minimumHAZUS, maxRoadLength=maxRoadLength, lengthUnit=lengthUnit
    )


def main(args):  # noqa: ANN001, ANN201, D103
    parser = argparse.ArgumentParser()
    parser.add_argument('--latMin', default=None, type=float)
    parser.add_argument('--latMax', default=None, type=float)
    parser.add_argument('--longMin', default=None, type=float)
    parser.add_argument('--longMax', default=None, type=float)
    parser.add_argument('--outputFolder', default=None)
    parser.add_argument(
        '--minimumHAZUS', default=True, type=str2bool, nargs='?', const=True
    )
    parser.add_argument('--maxRoadLength', default=100, type=float)
    parser.add_argument('--lengthUnit', default='m', type=str)

    args = parser.parse_args(args)

    # Change the current directory to the user-defined output folder:
    os.makedirs(args.outputFolder, exist_ok=True)  # noqa: PTH103
    os.chdir(args.outputFolder)

    # Run BRAILS TranspInventoryGenerator with the user-defined arguments:
    runBrails(
        args.latMin,
        args.latMax,
        args.longMin,
        args.longMax,
        args.minimumHAZUS,
        args.maxRoadLength,
        args.lengthUnit,
    )

    log_msg('BRAILS successfully generated the requested transportation inventory')


# Run main:
if __name__ == '__main__':
    main(sys.argv[1:])
