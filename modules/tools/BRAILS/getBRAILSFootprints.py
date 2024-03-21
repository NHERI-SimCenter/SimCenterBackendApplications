# Import packages needed for setting up required packages:
import sys
import subprocess
from importlib import metadata as importlib_metadata

# If not installed, install BRAILS, argparse, and requests:
required = {'BRAILS', 'argparse', 'requests'}
installed = set()

# Detect installed packages using Python-provided importlib.metadata:
for x in importlib_metadata.distributions():
    try:
        installed.add(x.name)
    except:
        pass

# If installed packages could not be detected, use importlib_metadata backport:
if not installed:
    import importlib_metadata
    for x in importlib_metadata.distributions():
        try:
            installed.add(x.name)
        except:
            pass
missing = required - installed

# Install missing packages:
python = sys.executable
if missing:
    print('\nInstalling packages required for running this widget...')
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], 
                          stdout=subprocess.DEVNULL)
    print('Successfully installed the required packages')

# If requests and BRAILS were previously installed ensure they are at their latest versions:
subprocess.check_call([python, '-m', 'pip', 'install', 'requests','-U'],
                      stdout=subprocess.DEVNULL)
 
import requests
latestBrailsVersion = requests.get('https://pypi.org/pypi/BRAILS/json').json()['info']['version']
if  importlib_metadata.version('BRAILS')!=latestBrailsVersion:
    print('\nAn older version of BRAILS was detected. Updating to the latest BRAILS version..')    
    subprocess.check_call([python, '-m', 'pip', 'install', 'BRAILS','-U'],
                          stdout=subprocess.DEVNULL)
    print('Successfully installed the latest version of BRAILS')   
 
# Import packages required for running the latest version of BRAILS:
import argparse
import os
from time import gmtime, strftime
from brails.workflow.FootprintHandler import FootprintHandler    

# Define a standard way of printing program outputs:
def log_msg(msg):
    formatted_msg = '{} {}'.format(strftime('%Y-%m-%dT%H:%M:%SZ', gmtime()), msg)
    print(formatted_msg)

# Define a way to call BRAILS FootprintHandler:
def runBrails(latMin, latMax, longMin, longMax, locationStr, fpSrc, fpSourceAttrMap, 
              outputfile, lengthunit):      
    # Initialize FootprintHandler:
    fpHandler = FootprintHandler()

    # Format location input based on the GUI input:
    if 'geojson' in fpSrc.lower() or 'csv' in fpSrc.lower():
        location = fpSrc
        fpSrc = 'osm'
    elif locationStr=="":
        location = (longMin,latMin,longMax,latMax)
    else:
        location = locationStr

    print(location)
    # Run FootprintHandler to get GeoJSON file for the footprints of the entered location:
    if fpSourceAttrMap=='':
        fpHandler.fetch_footprint_data(location, fpSource=fpSrc, lengthUnit=lengthunit, outputFile=outputfile)
    else: 
        fpHandler.fetch_footprint_data(location, fpSource=fpSrc, attrmap = fpSourceAttrMap,
                                       lengthUnit=lengthunit, outputFile = outputfile)

# Define a way to collect GUI input:
def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--latMin', default=None, type=float)
    parser.add_argument('--latMax', default=None, type=float)
    parser.add_argument('--longMin', default=None, type=float)
    parser.add_argument('--longMax', default=None, type=float)
    parser.add_argument('--location', default=None, type=str)
    parser.add_argument('--fpSource', default=None, type=str)
    parser.add_argument('--fpSourceAttrMap', default=None, type=str)    
    parser.add_argument('--outputFile', default=None)
    parser.add_argument('--lengthUnit', default="m", type=str)     
    
    args = parser.parse_args(args)

    # Create the folder for the user-defined output directory, if it does not exist:
    outdir = os.path.abspath(args.outputFile).replace(os.path.split(args.outputFile)[-1],'')
    os.makedirs(outdir, exist_ok=True)

    # Run BRAILS FootprintHandler with the user-defined arguments:
    runBrails(
        args.latMin, args.latMax, args.longMin, args.longMax, args.location, 
        args.fpSource, args.fpSourceAttrMap, args.outputFile, args.lengthUnit)

    log_msg('BRAILS successfully obtained the footprints for the entered location')
    
# Run main:
if __name__ == '__main__':
    main(sys.argv[1:])