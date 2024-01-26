# Import packages needed for setting up required packages:
import sys
import subprocess
import importlib.metadata

# If not installed, install BRAILS, argparse, and requests:
required = {'BRAILS', 'argparse', 'requests'}
installed = {x.name for x in importlib.metadata.distributions()}
missing = required - installed

python = sys.executable
if missing:
    print('\nInstalling packages required for running this widget...')
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], 
                          stdout=subprocess.DEVNULL)
    print('Successfully installed the required packages')

# If BRAILS was previously installed ensure it is the latest version:
import requests
latestBrailsVersion = requests.get('https://pypi.org/pypi/BRAILS/json').json()['info']['version']
if  importlib.metadata.version('BRAILS')!=latestBrailsVersion:
    print('\nAn older version of BRAILS was detected. Updating to the latest BRAILS version..')    
    subprocess.check_call([python, '-m', 'pip', 'install', 'BRAILS','-U'],
                          stdout=subprocess.DEVNULL)
    print('Successfully installed the latest version of BRAILS') 

# Import packages required for running the latest version of BRAILS:
import argparse
import os
from time import gmtime, strftime
from brails.TranspInventoryGenerator import TranspInventoryGenerator

def str2bool(v):
    # courtesy of Maxim @ stackoverflow

    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
# Define a standard way of printing program outputs:
def log_msg(msg):
    formatted_msg = '{} {}'.format(strftime('%Y-%m-%dT%H:%M:%SZ', gmtime()), msg)
    print(formatted_msg)

# Define a way to call BRAILS TranspInventoryGenerator:
def runBrails(latMin, latMax, longMin, longMax, 
              minimumHAZUS, maxRoadLength, lengthUnit):

    # Initialize TranspInventoryGenerator:
    invGenerator = TranspInventoryGenerator(location=(longMin,latMin,longMax,latMax))

    # Run TranspInventoryGenerator to generate an inventory for the entered location:
    invGenerator.generate()

    #Combine and format the generated inventory to SimCenter transportation network inventory json format
    if combineGeoJSON:
        invGenerator.combineAndFormat_HWY(minimumHAZUS, connectivity,maxRoadLength, lengthUnit)


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--latMin', default=None, type=float)
    parser.add_argument('--latMax', default=None, type=float)
    parser.add_argument('--longMin', default=None, type=float)
    parser.add_argument('--longMax', default=None, type=float)
    parser.add_argument('--outputFolder', default=None)
    parser.add_argument('--minimumHAZUS', default = True, 
                        type = str2bool, nargs='?', const=True)
    parser.add_argument('--maxRoadLength', default = 100, type=float)
    parser.add_argument('--lengthUnit', default="m", type=str)
    
    args = parser.parse_args(args)
    
    # Change the current directory to the user-defined output folder:
    os.makedirs(args.outputFolder, exist_ok=True)
    os.chdir(args.outputFolder)
    
    # Run BRAILS TranspInventoryGenerator with the user-defined arguments:
    runBrails(args.latMin, args.latMax, args.longMin, args.longMax, 
              args.minimumHAZUS, args.maxRoadLength, args.lengthUnit)

    log_msg('BRAILS successfully generated the requested transportation inventory')

# Run main:
if __name__ == '__main__':
    main(sys.argv[1:])    