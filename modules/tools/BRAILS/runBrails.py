import sys
import subprocess
import pkg_resources

from time import gmtime, strftime

def log_msg(msg):

    formatted_msg = '{} {}'.format(strftime('%Y-%m-%dT%H:%M:%SZ', gmtime()), msg)

    print(formatted_msg)

required = {'BRAILS', 'argparse'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
    
#modulename = 'BRAILS'
#if modulename not in sys.modules:
#    print(sys.modules)
#    python = sys.executable
#    print(python)
#    log_msg("needing to pip install  BRAILS")
#    subprocess.check_call([python, "-m", "pip", "install", "-q", modulename], stdout=subprocess.DEVNULL)

import argparse
from brails.InventoryGenerator import InventoryGenerator


def runBrails(latMin, latMax, longMin, longMax, seed, numBuildings, gKey):
    
    # Initialize InventoryGenerator:
    invGenerator = InventoryGenerator(location=(longMin,latMin,longMax,latMax),
                                      nbldgs=numBuildings, randomSelection=True,
                                      GoogleAPIKey=gKey)

    # Run InventoryGenerator to generate an inventory for the entered location:
    # To run InventoryGenerator for all enabled attributes set attributes='all':
    invGenerator.generate(attributes=['numstories','roofshape','buildingheight'])

def main(args):

    parser = argparse.ArgumentParser()
    parser.add_argument('--latMin', default=None, type=float)
    parser.add_argument('--latMax', default=None, type=float)
    parser.add_argument('--longMin', default=None, type=float)
    parser.add_argument('--longMax', default=None, type=float)
    parser.add_argument('--outputFile', default=None)
    parser.add_argument('--googKey', default=None)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--numBuildings', default=None, type=int)

    args = parser.parse_args(args)

    #print(args)
    runBrails(
        args.latMin, args.latMax,
        args.longMin, args.longMax,
        args.numBuildings,
        args.seed, args.googKey)

    log_msg('brails finished.')

if __name__ == '__main__':

    main(sys.argv[1:])    
    






    
