# -*- coding: utf-8 -*-
#
# Copyright (c) 2024 The Regents of the University of California
#
# This file is a part of SimCenter backend applications.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# BRAILS. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Barbaros Cetiner
#
# Last updated:
# 03-27-2024 

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
def runBrails(latMin, latMax, longMin, longMax, locationStr, outputfile):      
    # Initialize FootprintHandler:
    fpHandler = FootprintHandler()

    # Run FootprintHandler to generate the boundary GeoJSON file for the entered location:
    if locationStr=="":
        fpHandler._FootprintHandler__bbox2poly((longMin,latMin,longMax,latMax),
                                          outfile = outputfile)
    else:
        fpHandler._FootprintHandler__fetch_roi(locationStr, outfile = outputfile)        

# Define a way to collect GUI input:
def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--latMin', default=None, type=float)
    parser.add_argument('--latMax', default=None, type=float)
    parser.add_argument('--longMin', default=None, type=float)
    parser.add_argument('--longMax', default=None, type=float)
    parser.add_argument('--location', default=None, type=str)
    parser.add_argument('--outputFile', default=None)
    
    args = parser.parse_args(args)

    # Create the folder for the user-defined output directory, if it does not exist:
    outdir = os.path.abspath(args.outputFile).replace(os.path.split(args.outputFile)[-1],'')
    os.makedirs(outdir, exist_ok=True)

    # Run BRAILS FootprintHandler with the user-defined arguments:
    runBrails(
        args.latMin, args.latMax, args.longMin, args.longMax, args.location, args.outputFile)

    log_msg('BRAILS successfully generated the requested boundary polygon')
    
# Run main:
if __name__ == '__main__':
    main(sys.argv[1:])