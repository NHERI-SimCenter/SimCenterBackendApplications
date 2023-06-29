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

import sys, os, json
import argparse
from pathlib import Path
from createGM4BIM import createFilesForEventGrid
import importlib

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import whale.main as whale
from whale.main import log_msg, log_div

from sWHALE import runSWhale

def main(run_type, input_file, app_registry,
         force_cleanup, bldg_id_filter, reference_dir,
         working_dir, app_dir, log_file, output_dir,
         parallelType, mpiExec, numPROC):

    numP = 1
    procID = 0
    doParallel = False
    
    mpi_spec = importlib.util.find_spec("mpi4py")
    found = mpi_spec is not None
    if found:
        import mpi4py
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        numP = comm.Get_size()
        procID = comm.Get_rank()
        parallelType = 'parRUN'
        if numP < 2:
            doParallel = False
            numP = 1
            parallelType = 'seqRUN'
            procID = 0
        else:
            doParallel = True;


    print('siteResponse (doParallel, procID, numP):', doParallel, procID, numP, mpiExec, numPROC)

    # save the reference dir in the input file
    with open(input_file, 'r') as f:
        inputs = json.load(f)

    print('WORKING_DIR', working_dir)
    
    if procID == 0:
        if not os.path.exists(working_dir):
            os.mkdir(working_dir)

    if doParallel == True:
        comm.Barrier()
        
    # initialize log file
    if log_file == 'log.txt':
        log_file_path = working_dir + '/log.txt' + '.' + str(procID)
    else:
        log_file_path = log_file + '.' + str(procID)

    whale.set_options({
        "LogFile": log_file_path,
        "LogShowMS": False,
        "PrintLog": True
        })
    log_msg('\nrWHALE workflow\n',
            prepend_timestamp=False, prepend_blank_space=False)

    whale.print_system_info()

    # echo the inputs
    log_div(prepend_blank_space=False)
    log_div(prepend_blank_space=False)
    log_msg('Started running the workflow script')
    log_div()

    if force_cleanup:
        log_msg('Forced cleanup turned on.')

    #
    # parse regionalEventAppData, create new input file 
    # for the rWHALE workflow
    #

    randomVariables = []
    if "randomVariables" in inputs.keys():
        randomVariables = inputs["randomVariables"]

    inputApplications = inputs["Applications"]
    regionalApplication = inputApplications["RegionalEvent"]    
    appData = regionalApplication["ApplicationData"]
    regionalData = inputs["RegionalEvent"]    
    regionalData["eventFile"]=appData["inputEventFilePath"]  + "/" + appData["inputEventFile"]
    regionalData["eventFilePath"]=appData["inputEventFilePath"]

    siteFilter = appData["filter"]

    # KZ: 10/19/2022, adding new attributes for the refactored whale

    remoteAppDir = inputs.get('remoteAppDir', "")
    localAppDir = inputs.get('localAppDir', "")
    if localAppDir == "":
        localAppDir = remoteAppDir
    if remoteAppDir == "":
        remoteAppDir = localAppDir

    siteResponseInput = {
        "units": inputs["units"],
        "outputs": {
            "IM": True,
            "EDP": False,
            "DM": False,
            "AIM": False,            
            "DV": False,
            "every_realization": False
        },        
        "RegionalEvent": regionalData,
        "randomVariables" : randomVariables,
        "Applications": {
            "RegionalMapping": inputApplications["RegionalMapping"],
            "UQ": inputApplications["UQ"],
            "Assets" : {
                "Buildings": {
                    "Application": "CSV_to_AIM",
                    "ApplicationData": {
                        "assetSourceFile": appData["soilGridParametersFilePath"] + "/" +     appData["soilGridParametersFile"],
                        "filter": siteFilter
                    }
                }
            },
            "EDP": {
                "Application": "DummyEDP",
                "ApplicationData": {}
            },
            "Events": [
                {
                    "EventClassification": "Earthquake",
                    "Application": "RegionalSiteResponse",
                    "ApplicationData": {
                        "pathEventData": "inputMotions",
                        "mainScript": appData["siteResponseScript"],
                        "modelPath": appData["siteResponseScriptPath"],
                        "ndm": 3
                    }
                }
            ]
        },
        "UQ": inputs.get('UQ', dict()),
        "localAppDir": localAppDir,
        "remoteAppDir": remoteAppDir,
        "runType": inputs.get('runType', ""),
        "DefaultValues": {
            "driverFile": "driver",
            "edpFiles": [
                "EDP.json"
            ],
            "filenameDL": "BIM.json",
            "filenameEDP": "EDP.json",
            "filenameEVENT": "EVENT.json",
            "filenameSAM": "SAM.json",
            "filenameSIM": "SIM.json",
            "rvFiles": [
                "SAM.json",
                "EVENT.json",
                "SIM.json"
            ],
            "workflowInput": "scInput.json",
            "workflowOutput": "EDP.json"
        }
    }        

    #siteResponseInputFile = 'tmpSiteResponseInput.json'
    #siteResponseInputFile = os.path.join(os.path.dirname(input_file),'tmpSiteResponseInput.json')
    # KZ: 10/19/2022, fixing the json file path
    siteResponseInputFile = os.path.join(os.path.dirname(reference_dir),'tmpSiteResponseInput.json')

    if procID == 0:
        with open(siteResponseInputFile, 'w') as json_file:
            json_file.write(json.dumps(siteResponseInput, indent=2))    

            
    WF = whale.Workflow(run_type,
                        siteResponseInputFile,
                        app_registry,
                        app_type_list = ['Assets', 'RegionalMapping', 'Event', 'EDP', 'UQ'],
                        reference_dir = reference_dir,
                        working_dir = working_dir,
                        app_dir = app_dir,
                        parType = parallelType,
                        mpiExec = mpiExec,
                        numProc = numPROC)

    if bldg_id_filter is not None:
        print(bldg_id_filter)
        log_msg(
            f'Overriding simulation scope; running buildings {bldg_id_filter}')

        # If a Min or Max attribute is used when calling the script, we need to
        # update the min and max values in the input file.
        WF.workflow_apps['Building'].pref["filter"] = bldg_id_filter

    if procID == 0:
        
        # initialize the working directory        
        WF.init_workdir()

        # prepare the basic inputs for individual buildings
        asset_files = WF.create_asset_files()

    if doParallel == True:
        comm.Barrier()

    asset_files = WF.augment_asset_files()        

    if procID == 0:        
        for asset_type, assetIt in asset_files.items() :

            # perform the regional mapping
            #WF.perform_regional_mapping(assetIt)
            # KZ: 10/19/2022, adding the required argument for the new whale
            print('0 STARTING MAPPING')
            # FMK _ PARALLEL WF.perform_regional_mapping(assetIt, asset_type, False)
            # WF.perform_regional_mapping(assetIt, asset_type)             
            WF.perform_regional_mapping(assetIt, asset_type, False)

    # get all other processes to wait till we are here
    if doParallel == True:
        comm.Barrier()        

    print("BARRIER AFTER PERFORM REGIONAL MAPPING")

    count = 0
    for asset_type, assetIt in asset_files.items() :
        
        # TODO: not elegant code, fix later
        with open(assetIt, 'r') as f:
            asst_data = json.load(f)
        
        # The preprocess app sequence (previously get_RV)
        preprocess_app_sequence = ['Event', 'EDP']
        
        # The workflow app sequence
        WF_app_sequence = ['Assets', 'Event', 'EDP']

        # For each asset
        for asst in asst_data:

            if count % numP == procID:            
                log_msg('', prepend_timestamp=False)
                log_div(prepend_blank_space=False)
                log_msg(f"{asset_type} id {asst['id']} in file {asst['file']}")
                log_div()
        
                # Run sWhale
                print('COUNT: ', count, ' ID: ', procID)

                runSWhale(inputs = None,
                          WF = WF,
                          assetID = asst['id'],
                          assetAIM = asst['file'],
                          prep_app_sequence = preprocess_app_sequence,
                          WF_app_sequence = WF_app_sequence,
                          asset_type = asset_type,
                          copy_resources = True,
                          force_cleanup = force_cleanup)
                
            count = count+1

    if doParallel == True:
        comm.Barrier()
        
    if procID == 0:                        
        createFilesForEventGrid(os.path.join(working_dir,'Buildings'),
                                output_dir,
                                force_cleanup)

        # aggregate results
        #WF.aggregate_results(bldg_data = bldg_data)
        # KZ: 10/19/2022, chaning bldg_data to asst_data
        WF.aggregate_results(asst_data= asst_data)


    if doParallel == True:
        comm.Barrier()

    # clean up intermediate files from the working directory
    if force_cleanup:
        if procID == 0:
            WF.cleanup_workdir()

    log_msg('Workflow completed.')
    log_div(prepend_blank_space=False)
    log_div(prepend_blank_space=False)

if __name__ == '__main__':

    pwd1 = os.getcwd()
    if os.path.basename(pwd1) == 'Results':
        os.chdir('..')
        
    #
    # little bit of preprocessing
    #
    
    thisScriptPath = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
    registryFile = thisScriptPath / "WorkflowApplications.json"
    applicationDir = Path(thisScriptPath).parents[1]    
    pwd = os.getcwd()
    currentDir = Path(pwd)
    referenceDir = currentDir / "input_data"
    siteResponseOutputDir = referenceDir / "siteResponseWorkingDir"
    siteResponseAggregatedResultsDir = referenceDir / "siteResponseOutputMotions"

    print('PWD: ', pwd);
    print('currentDir: ', currentDir);
    print('referenceDir: ', referenceDir);
    print('siteResponseOutputDir: ', siteResponseOutputDir);    
    
    
    #
    # parse command line
    #
    
    workflowArgParser = argparse.ArgumentParser(
        "Run the NHERI SimCenter rWHALE workflow for a set of assets.",
        allow_abbrev=False)

    workflowArgParser.add_argument("-i", "--input",
        default=None,
        help="Configuration file specifying the applications and data to be "
             "used")
    workflowArgParser.add_argument("-F", "--filter",
        default=None,
        help="Provide a subset of building ids to run")
    workflowArgParser.add_argument("-c", "--check",
        help="Check the configuration file")
    workflowArgParser.add_argument("-r", "--registry",
        default = registryFile,
        help="Path to file containing registered workflow applications")
    workflowArgParser.add_argument("-f", "--forceCleanup",
        action="store_true",
        help="Remove working directories after the simulation is completed.")
    workflowArgParser.add_argument("-d", "--referenceDir",
        default = str(referenceDir),
        help="Relative paths in the config file are referenced to this directory.")
    workflowArgParser.add_argument("-w", "--workDir",
        default=str(siteResponseOutputDir),
        help="Absolute path to the working directory.")
    workflowArgParser.add_argument("-o", "--outputDir",
        default=str(siteResponseAggregatedResultsDir),
        help="Absolute path to the working directory.")    
    workflowArgParser.add_argument("-a", "--appDir",
        default=None,
        help="Absolute path to the local application directory.")
    workflowArgParser.add_argument("-l", "--logFile",
        default='log.txt',
        help="Path where the log file will be saved.")

    # adding some parallel stuff
    workflowArgParser.add_argument("-p", "--parallelType",
        default='seqRUN',
        help="How parallel runs: options seqRUN, parSETUP, parRUN")
    workflowArgParser.add_argument("-m", "--mpiexec",
        default='mpiexec',
        help="How mpi runs, e.g. ibrun, mpirun, mpiexec")
    workflowArgParser.add_argument("-n", "--numP",
        default='8',
        help="If parallel, how many jobs to start with mpiexec option") 

    #Parsing the command line arguments
    wfArgs = workflowArgParser.parse_args()
    
    # update the local app dir with the default - if needed
    if wfArgs.appDir is None:
        workflow_dir = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
        wfArgs.appDir = workflow_dir.parents[1]

    if wfArgs.check:
        run_type = 'set_up'
    else:
        #run_type = 'run'
        # KZ: 10/19/22, changing to the new run type for the refactored whale
        run_type = 'runningLocal'

    #
    # Calling the main workflow method and passing the parsed arguments
    #
    print('FMK siteResponse main: WORKDIR: ', wfArgs.workDir)
    numPROC = int(wfArgs.numP)
    
    main(run_type = run_type,
         input_file = wfArgs.input,
         app_registry = wfArgs.registry,
         force_cleanup = wfArgs.forceCleanup,
         bldg_id_filter = wfArgs.filter,
         reference_dir = wfArgs.referenceDir,
         working_dir = wfArgs.workDir,
         app_dir = wfArgs.appDir,
         log_file = wfArgs.logFile,
         output_dir = wfArgs.outputDir,
         parallelType = wfArgs.parallelType,
         mpiExec = wfArgs.mpiexec,
         numPROC = numPROC)

    #
    # now create new event file, sites and record files
    #
    
    #createFilesForEventGrid(wfArgs.workDir,
    #                        wfArgs.outputDir,
    #                        wfArgs.forceCleanup)
                            

    # chdir again back to where ADAM starts!
    os.chdir(pwd1)
