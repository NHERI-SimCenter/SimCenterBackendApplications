# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 The Regents of the University of California
# Copyright (c) 2019 Leland Stanford Junior University
#
# This file is part of the SimCenter Backend Applications.
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
# SimCenter Backend Applications. If not, see <http://www.opensource.org/licenses/>.
#
#
# Modified 'cb-cities' code provided by the Soga Research Group UC Berkeley
# Dr. Stevan Gavrilovic

import itertools
import argparse
import os, sys, json, posixpath
import pandas as pd
import numpy as np

from operator import itemgetter

from CBCitiesMethods import *

def main(node_info, pipe_info):


    # Load Data
    
    print('Loading the node json file...')
    
    with open(node_info, 'r') as f:
        node_data = json.load(f)
    
    with open(pipe_info, 'r') as f:
        pipe_data = json.load(f)
        
    
    min_id = int(pipe_data[0]['id'])
    max_id = int(pipe_data[0]['id'])
    
    allPipes = []
    
    for pipe in pipe_data:
    
        AIM_file = pipe['file']
        
        asst_id = pipe['id']
        
        min_id = min(int(asst_id), min_id)
        max_id = max(int(asst_id), max_id)
    
        # Open the AIM file
        with open(AIM_file, 'r') as f:
            pipe = AIM_data = json.load(f)
            
        allPipes.append(pipe)
        

    # read pgv for nodes
#    pgv_csv_files = glob('../data/rupture/rupture62_im/*.csv')

    # Mapping & Saving
    import multiprocessing as mp

    pool = mp.Pool(mp.cpu_count()-1)
    results = pool.map(add_failrate2pipe, [pipe for pipe in allPipes])
    pool.close()
    
    df = pd.DataFrame({'DV':{},'MeanFailureProbability':{}})

    for pipe in results:
    
        failureProbArray = pipe['fail_prob']
        avgFailureProb = np.average(failureProbArray)
        pipe_id = pipe['GeneralInformation']['AIM_id']

        print("pipe_id: ",pipe_id)
#        print("failureProbArray: ",failureProbArray)
        print("avgFailureProb: ",avgFailureProb)
        
        df2 = pd.DataFrame({'DV': pipe_id, 'MeanFailureProbability': avgFailureProb}, index=[0])
        df = pd.concat([df,df2], axis=0)
       
    
    # Get the directory for saving the results, assume it is the same one with the AIM file
    aimDir = os.path.dirname(pipe_info)
    aimFileName = os.path.basename(pipe_info)
    
    saveDir = posixpath.join(aimDir,f'DV_{min_id}-{max_id}.csv')
        
    df.to_csv(saveDir, index = False)
    
    return 0
        #failed_pipes = fail_pipes_number(pipe)
    

if __name__ == '__main__':

    #Defining the command line arguments
    workflowArgParser = argparse.ArgumentParser(
        "Run the CB-cities water distribution damage and loss workflow.",
        allow_abbrev=False)

    workflowArgParser.add_argument("-n", "--nodeInfo",
        default=None,
        help="Node information.")
    workflowArgParser.add_argument("-p", "--pipeInfo",
        default=None,
        help="Pipe Information.")
    workflowArgParser.add_argument("-s", "--save_dir",
        default=None,
        help="Directory where to save the results.")

    #Parsing the command line arguments
    wfArgs = workflowArgParser.parse_args()

    # update the local app dir with the default - if needed
#    if wfArgs.appDir is None:
#        workflow_dir = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
#        wfArgs.appDir = workflow_dir.parents[1]

    #Calling the main workflow method and passing the parsed arguments
    main(node_info = wfArgs.nodeInfo, pipe_info = wfArgs.pipeInfo)
