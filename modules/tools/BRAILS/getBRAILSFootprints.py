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
 
# Import packages required for running the latest version of BRAILS:
import sys
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
    if locationStr == "\"\"":
        locationStr = ""

    # Format location input based on the GUI input:
    if 'geojson' in fpSrc.lower() or 'csv' in fpSrc.lower():
        location = fpSrc
        fpSrc = 'osm'
    elif locationStr=="":
        location = (longMin,latMin,longMax,latMax)
    else:
        location = locationStr

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