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
# You should have received a copy of the BSD 3-Clause License along with the
# SimCenter Backend Applications. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Frank McKenna
# Adam Zsarnóczay
# Wael Elhaddad
# Michael Gardner
# Chaofeng Wang
# Kuanshi Zhong
# Stevan Gavrilovic

import sys, os, json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import whale.main as whale
from whale.main import log_msg, log_div

from sWHALE import runSWhale


def main(run_type, input_file, app_registry,
         force_cleanup, bldg_id_filter, reference_dir,
         working_dir, app_dir, log_file, site_response):

    # save the reference dir in the input file
    with open(input_file, 'r') as f:
        inputs = json.load(f)

    # TODO: if the ref dir is needed, do NOT save it to the input file, store it
    # somewhere else in a file that i not shared among processes
    #inputs['refDir'] = reference_dir
    #with open(input_file, 'w') as f:
    #    json.dump(inputs, f, indent=2)

    # TODO: remove the commented section below, I only kept it for now to make
    # sure it is not needed

    #if working_dir is not None:
    #    runDir = working_dir
    #else:
    #    runDir = inputs['runDir']

    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    # initialize log file
    if log_file == 'log.txt':
        log_file_path = working_dir + '/log.txt'
    else:
        log_file_path = log_file

    whale.set_options({
        "LogFile": log_file_path,
        "LogShowMS": False,
        "PrintLog": True
        })
        
    log_msg('\nrWHALE workflow\n', prepend_timestamp=False, prepend_blank_space=False)

    whale.print_system_info()

    # echo the inputs
    log_div(prepend_blank_space=False)
    log_div(prepend_blank_space=False)
    log_msg('Started running the workflow script')
    log_div()

    if force_cleanup:
        log_msg('Forced cleanup turned on.')

    WF = whale.Workflow(run_type, input_file, app_registry,
        app_type_list = ['Assets', 'RegionalEvent', 'RegionalMapping', 
        'Event', 'Modeling', 'EDP', 'Simulation', 'UQ', 'DL'],
        reference_dir = reference_dir,
        working_dir = working_dir,
        app_dir = app_dir)

    if bldg_id_filter is not None:
        print(bldg_id_filter)
        log_msg(f'Overriding simulation scope; running buildings {bldg_id_filter}')

        # If a Min or Max attribute is used when calling the script, we need to
        # update the min and max values in the input file.
        WF.workflow_apps['Building'].pref["filter"] = bldg_id_filter

    # initialize the working directory
    WF.init_workdir()

    # prepare the basic inputs for individual assets
    asset_files = WF.create_asset_files()

    # perform the event simulation (if needed), commented by KZ
    if site_response == 'sequential' and 'RegionalEvent' in WF.workflow_apps.keys():
        # run anlaysis
        WF.perform_regional_event()
        
    for asset_type, assetIt in asset_files.items() :

        # perform the regional mapping
        WF.perform_regional_mapping(assetIt, asset_type)
        
        # TODO: not elegant code, fix later
        with open(assetIt, 'r') as f:
            asst_data = json.load(f)
        
        # Mapping & Saving
        #        import multiprocessing as mp
        #        pool = mp.Pool(mp.cpu_count()-1)
        #        results = pool.starmap(runAssetAnalysis, [[asst,asset_type,WF,force_cleanup] for asst in asst_data])
        #        pool.close()
        
        # Sometimes multiple asset types need to be analyzed together, e.g., pipelines and nodes in a water network
        run_asset_type = asset_type

        if asset_type == 'Buildings' :
            pass
        elif asset_type == 'WaterNetworkNodes' :
            continue # Run the nodes with the pipelines, i.e., the water distribution network
        elif asset_type == 'WaterNetworkPipelines' :
            run_asset_type = 'WaterDistributionNetwork' # Run the pipelines with the entire water distribution network
        else :
            print("No support for asset type: ",asset_type)
            
            
        # The preprocess app sequence (previously get_RV)
        preprocess_app_sequence = ['Event', 'Modeling', 'EDP', 'Simulation']
            
        # The workflow app sequence
        #FMKWF_app_sequence = ['Assets', 'Event', 'Modeling', 'EDP', 'Simulation']
        WF_app_sequence = ['Event', 'Modeling', 'EDP', 'Simulation']        

        # For each asset
        for asst in asst_data:
            
            log_msg('', prepend_timestamp=False)
            log_div(prepend_blank_space=False)
            log_msg(f"{asset_type} id {asst['id']} in file {asst['file']}")
            log_div()

            
            
            # Run sWhale
            runSWhale(
                inputs = None, 
                WF = WF, 
                assetID = asst['id'], 
                assetAIM = asst['file'], 
                prep_app_sequence = preprocess_app_sequence,  
                WF_app_sequence = WF_app_sequence, 
                asset_type = run_asset_type, 
                copy_resources = True, 
                force_cleanup = force_cleanup)
            
        # aggregate results
        if asset_type == 'Buildings' :
        
            WF.aggregate_results(asst_data = asst_data, asset_type = asset_type)
            
        elif asset_type == 'WaterNetworkPipelines' :
        
            # Provide the headers and out types
            headers = dict(DV = [0])
                
            out_types = ['DV']
            
            WF.aggregate_results(asst_data = asst_data,
                asset_type = asset_type, out_types = out_types, headers = headers)


    if force_cleanup:
        # clean up intermediate files from the working directory
        WF.cleanup_workdir()

    log_msg('Workflow completed.')
    log_div(prepend_blank_space=False)
    log_div(prepend_blank_space=False)

if __name__ == '__main__':

    #Defining the command line arguments

    workflowArgParser = argparse.ArgumentParser(
        "Run the NHERI SimCenter rWHALE workflow for a set of assets.",
        allow_abbrev=False)

    workflowArgParser.add_argument("configuration",
        help="Configuration file specifying the applications and data to be "
             "used")
    workflowArgParser.add_argument("-F", "--filter",
        default=None,
        help="Provide a subset of building ids to run")
    workflowArgParser.add_argument("-c", "--check",
        help="Check the configuration file")
    workflowArgParser.add_argument("-r", "--registry",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "WorkflowApplications.json"),
        help="Path to file containing registered workflow applications")
    workflowArgParser.add_argument("-f", "--forceCleanup",
        action="store_true",
        help="Remove working directories after the simulation is completed.")
    workflowArgParser.add_argument("-d", "--referenceDir",
        default=os.path.join(os.getcwd(), 'input_data'),
        help="Relative paths in the config file are referenced to this directory.")
    workflowArgParser.add_argument("-w", "--workDir",
        default=os.path.join(os.getcwd(), 'results'),
        help="Absolute path to the working directory.")
    workflowArgParser.add_argument("-a", "--appDir",
        default=None,
        help="Absolute path to the local application directory.")
    workflowArgParser.add_argument("-l", "--logFile",
        default='log.txt',
        help="Path where the log file will be saved.")
    workflowArgParser.add_argument("-s", "--siteResponse",
        default='sequential',
        help="How site response analysis runs.")

    #Parsing the command line arguments
    wfArgs = workflowArgParser.parse_args()

    # update the local app dir with the default - if needed
    if wfArgs.appDir is None:
        workflow_dir = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
        wfArgs.appDir = workflow_dir.parents[1]

    if wfArgs.check:
        run_type = 'set_up'
    else:
        run_type = 'run'

    #Calling the main workflow method and passing the parsed arguments
    main(run_type = run_type,
         input_file = Path(wfArgs.configuration).resolve(), # to pass the absolute path to the input file
         app_registry = wfArgs.registry,
         force_cleanup = wfArgs.forceCleanup,
         bldg_id_filter = wfArgs.filter,
         reference_dir = wfArgs.referenceDir,
         working_dir = wfArgs.workDir,
         app_dir = wfArgs.appDir,
         log_file = wfArgs.logFile,
         site_response = wfArgs.siteResponse)
