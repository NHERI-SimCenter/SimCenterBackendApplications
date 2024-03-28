# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 The Regents of the University of California
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
# 03-28-2024 

# Import packages required for running the latest version of BRAILS:
import argparse
import os
from time import gmtime, strftime
import sys
from brails.InventoryGenerator import InventoryGenerator    

# Define a standard way of printing program outputs:
def log_msg(msg):
    formatted_msg = '{} {}'.format(strftime('%Y-%m-%dT%H:%M:%SZ', gmtime()), msg)
    print(formatted_msg)

# Define a way to call BRAILS InventoryGenerator:
def runBrails(latMin, latMax, longMin, longMax, locationStr, lengthUnit, 
              fpSource, fpAttrMap, invInput, invAttributeMap, attrRequested, 
              outputFile, seed, numBuildings, getAllBuildings, gKey):    
    
    if getAllBuildings:
        numBuildings = 'all'

    if attrRequested not in ['all','hazuseq']:
        attrRequested = attrRequested.split(',')

    # Initialize InventoryGenerator:
    if locationStr=="":
        invGenerator = InventoryGenerator(location=(longMin,latMin,longMax,latMax),
                                          nbldgs=numBuildings, randomSelection=seed,
                                          GoogleAPIKey=gKey)
    else:
        invGenerator = InventoryGenerator(location=locationStr,
                                          nbldgs=numBuildings, randomSelection=seed,
                                          GoogleAPIKey=gKey)
    

    # Run InventoryGenerator to generate an inventory for the entered location:
    invGenerator.generate(attributes=attrRequested, outFile=outputFile, 
                          lengthUnit=lengthUnit)

# Define a way to collect GUI input:
def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--latMin', default=None, type=float)
    parser.add_argument('--latMax', default=None, type=float)
    parser.add_argument('--longMin', default=None, type=float)
    parser.add_argument('--longMax', default=None, type=float)
    parser.add_argument('--location', default=None, type=str)
    parser.add_argument('--lengthUnit', default="m", type=str) 
    parser.add_argument('--fpSource', default=None, type=str)
    parser.add_argument('--fpAttrMap', default=None, type=str)
    parser.add_argument('--invInput', default=None, type=str)
    parser.add_argument('--invAttributeMap', default=None, type=str)
    parser.add_argument('--attrRequested', default=None, type=str)        
    parser.add_argument('--outputFile', default=None, type=str)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--numBuildings', default=None, type=int)  
    parser.add_argument('--getAllBuildings', default=None, type=int)  
    parser.add_argument('--googKey', default=None, type=str)

    args = parser.parse_args(args)

    # Create the folder for the user-defined output directory, if it does not exist:
    outdir = os.path.abspath(args.outputFile).replace(os.path.split(args.outputFile)[-1],'')
    os.makedirs(outdir, exist_ok=True)

    # Run BRAILS InventoryGenerator with the user-defined arguments:
    runBrails(
        args.latMin, args.latMax, args.longMin, args.longMax, args.location,
        args.lengthUnit, args.fpSource, args.fpAttrMap, args.invInput, args.invAttributeMap, 
        args.attrRequested, args.outputFile, args.seed, args.numBuildings, 
        args.getAllBuildings, args.googKey)

    log_msg('BRAILS successfully generated the requested building inventory')
    
# Run main:
if __name__ == '__main__':
    main(sys.argv[1:])