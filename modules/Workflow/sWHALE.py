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
# Contributors:
# Frank McKenna
# Adam Zsarnóczay
# Wael Elhaddad
# Michael Gardner
# Chaofeng Wang
# Stevan Gavrilovic

import sys, os, json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import whale.main as whale
from whale.main import log_msg, log_div


def runSWhale(inputs,
              WF,
              assetID = None,
              assetAIM = 'AIM.json', 
              prep_app_sequence = ['Event', 'Modeling', 'EDP', 'Simulation'], 
              WF_app_sequence = ['Event', 'Modeling', 'EDP', 'Simulation'], 
              asset_type = None,
              copy_resources = False,
              force_cleanup = False) :

    # update the runDir, if needed
    #    with open(input_file, 'r') as f:
    #        inputs = json.load(f)
    #    runDir = inputs['runDir']
    #
    #    if working_dir is not None:
    #        runDir = working_dir
    #    else:
    #        runDir = inputs['runDir']
    #
    #
    #    whale.log_file = runDir + '/log.txt'
    #
    #    # initialize log file
    #    whale.set_options({
    #        "LogFile": runDir + '/log.txt',
    #        "LogShowMS": False,
    #        "PrintLog": True
    #        })
    #
    
    log_msg('\nStarting sWHALE workflow\n', prepend_timestamp=False, prepend_blank_space=False)

#    whale.print_system_info()

    # echo the inputs
    log_div(prepend_blank_space=False)
    log_div(prepend_blank_space=False)
    log_msg('Running the workflow script')
    log_div()

    if WF.run_type != 'loss_only':

        # initialize the working directory
        #  assetID is a unique asset identifier, assetAIM is the asset information model, e.g., 'AIM.json'
        WF.init_simdir(assetID, assetAIM)

        #Delete the origional AIM and use the templatedir/AIM from here

        # prepare the input files for the simulation
        WF.preprocess_inputs(prep_app_sequence, assetAIM, assetID, asset_type)
        
        # create the workflow driver file
        WF.create_driver_file(WF_app_sequence, assetID, assetAIM)
        
        # gather all Randomvariables and EDP's and place in new input file for UQ
        WF.gather_workflow_inputs(assetID, assetAIM);

        # run uq engine to simulate response
        WF.simulate_response(AIM_file_path = assetAIM, asst_id = assetID)
        
    if WF.run_type != 'set_up':

        # run dl engine to estimate losses
        # Use the templatedir/AIM.json for pelicun
        WF.estimate_losses(AIM_file_path = assetAIM,
                           asst_id = assetID,
                           asset_type = asset_type,
                           input_file = inputs,
                           copy_resources=copy_resources)
    

        # run performance engine to assess asset performance, e.g., recovery
        WF.estimate_performance(AIM_file_path = assetAIM,
                                asst_id = assetID,
                                asset_type = asset_type,
                                input_file = inputs,
                                copy_resources=copy_resources)

    if force_cleanup:
        #clean up intermediate files from the simulation
        WF.cleanup_simdir(assetID)

    log_msg('Workflow completed.')
    log_div(prepend_blank_space=False)
    log_div(prepend_blank_space=False)


def main(run_type, input_file, app_registry, working_dir, app_dir, log_file):

    # update the runDir, if needed
    with open(input_file, 'r') as f:
        inputs = json.load(f)
    runDir = inputs['runDir']

    if working_dir is not None:
        runDir = working_dir
    else:
        runDir = inputs['runDir']


    whale.log_file = runDir + '/log.txt'
    
    # initialize log file
    whale.set_options({
        "LogFile": runDir + '/log.txt',
        "LogShowMS": False,
        "PrintLog": True
        })
    
    log_msg('\nsWHALE workflow\n',
            prepend_timestamp=False, prepend_blank_space=False)

    whale.print_system_info()

    # echo the inputs
    log_div(prepend_blank_space=False)
    log_div(prepend_blank_space=False)
    log_msg('Started running the workflow script')
    log_div()

    # If there is an external EDP file provided, change the run_type to loss_only
    try:
        if inputs['DL']['Demands']['DemandFilePath'] is not None:
            run_type = 'loss_only'
    except:
        pass

    WF = whale.Workflow(run_type, input_file, app_registry,
        app_type_list = ['Event', 'Modeling', 'EDP', 'Simulation', 'UQ', 'DL', 'Performance'],
        working_dir = working_dir,
        app_dir = app_dir)
        
    runSWhale(inputs = input_file,
              WF = WF,
              prep_app_sequence = ['Event', 'Modeling', 'EDP', 'Simulation'],
              WF_app_sequence = ['Event', 'Modeling', 'EDP', 'Simulation'])


if __name__ == '__main__':

    """
    if len(sys.argv) != 4:
        print('\nNeed three arguments, e.g.:\n')
        print('    python %s action workflowinputfile.json workflowapplications.json' % sys.argv[0])
        print('\nwhere: action is either check or run\n')
        exit(1)

    main(run_type=sys.argv[1], input_file=sys.argv[2], app_registry=sys.argv[3])
    """

    #Defining the command line arguments

    workflowArgParser = argparse.ArgumentParser(
        "Run the NHERI SimCenter sWHALE workflow for a single asset.",
        allow_abbrev=False)

    workflowArgParser.add_argument("runType",
        help="Specifies the type of run requested.")
    workflowArgParser.add_argument("inputFile",
        help="Specifies the input file for the workflow.")
    workflowArgParser.add_argument("registry",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "WorkflowApplications.json"),
        help="Path to file containing registered workflow applications")
    workflowArgParser.add_argument("-w", "--workDir",
        default=None,
        help="Absolute path to the working directory.")
    workflowArgParser.add_argument("-a", "--appDir",
        default=None,
        help="Absolute path to the local application directory.")
    workflowArgParser.add_argument("-l", "--logFile",
        default='log.txt',
        help="Path where the log file will be saved.")

    #Parsing the command line arguments
    wfArgs = workflowArgParser.parse_args()

    # update the local app dir with the default - if needed
    if wfArgs.appDir is None:
        workflow_dir = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
        wfArgs.appDir = workflow_dir.parents[1]

    #Calling the main workflow method and passing the parsed arguments
    main(run_type = wfArgs.runType,
         input_file = Path(wfArgs.inputFile).resolve(),
         app_registry = wfArgs.registry,
         working_dir = wfArgs.workDir,
         app_dir = wfArgs.appDir,
         log_file = wfArgs.logFile)
