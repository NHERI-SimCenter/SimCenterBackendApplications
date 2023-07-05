import sys
import subprocess
import pkg_resources

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

# Define a standard way of printing program outputs:
def log_msg(msg):
    formatted_msg = '{} {}'.format(strftime('%Y-%m-%dT%H:%M:%SZ', gmtime()), msg)
    print(formatted_msg)

# Define a way to call BRAILS TranspInventoryGenerator:
def runBrails(latMin, latMax, longMin, longMax):
    # Initialize TranspInventoryGenerator:
    invGenerator = TranspInventoryGenerator(location=(longMin,latMin,longMax,latMax))

    # Run TranspInventoryGenerator to generate an inventory for the entered location:
    invGenerator.generate()

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--latMin', default=None, type=float)
    parser.add_argument('--latMax', default=None, type=float)
    parser.add_argument('--longMin', default=None, type=float)
    parser.add_argument('--longMax', default=None, type=float)
    parser.add_argument('--outputFile', default=None)

    args = parser.parse_args(args)
    
    # Run BRAILS TranspInventoryGenerator with the user-defined arguments:
    runBrails(args.latMin, args.latMax,args.longMin, args.longMax)

    log_msg('BRAILS successfully generated the requested transportation inventory')

# Run main:
if __name__ == '__main__':
    main(sys.argv[1:])    