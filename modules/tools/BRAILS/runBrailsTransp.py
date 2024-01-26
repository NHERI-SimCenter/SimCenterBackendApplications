import sys
import subprocess
import pkg_resources
import os

from time import gmtime, strftime

# If not installed yet, install BRAILS and argparse:
required = {'BRAILS', 'argparse'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

import argparse
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
def runBrails(latMin, latMax, longMin, longMax,\
              combineGeoJSON, minimumHAZUS, connectivity, maxRoadLength,\
                lengthUnit):

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
    parser.add_argument('--outputFile', default=None)
    parser.add_argument('--combineGeoJSON', default = True, \
                        type = str2bool, nargs='?', const=False)
    parser.add_argument('--minimumHAZUS', default = True, \
                        type = str2bool, nargs='?', const=True)
    parser.add_argument('--connectivity', default = True, \
                        type = str2bool, nargs='?', const=False)
    parser.add_argument('--maxRoadLength', default = 100, type=float)
    parser.add_argument('--lengthUnit', default="m", type=str)
    
    args = parser.parse_args(args)
    # Change to output folder
    if os.path.exists(args.outputFile):
        os.chdir(args.outputFile)
    else:
        os.mkdir(args.outputFile)
        os.chdir(args.outputFile)
    # Run BRAILS TranspInventoryGenerator with the user-defined arguments:
    runBrails(args.latMin, args.latMax,args.longMin, args.longMax,\
              args.combineGeoJSON, args.minimumHAZUS, args.connectivity,\
              args.maxRoadLength, args.lengthUnit)

    log_msg('BRAILS successfully generated the requested transportation inventory')

# Run main:
if __name__ == '__main__':
    main(sys.argv[1:])    