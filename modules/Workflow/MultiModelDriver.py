# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 The Regents of the University of California
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
# SimCenter Backend Applications. If not, see <http://www.opensource.org/licences/>.
#
# Contributors:
# Frank McKenna

import sys, os, json
import subprocess
import argparse
from pathlib import Path
from copy import deepcopy


sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import whale.main as whale
from whale.main import log_msg, log_div, _parse_app_registry, create_command, run_command

def main(inputFile,
         driverFile,
         appKey,
         registryFile,
         appDir,
         runType,
         osType,
         runDriver):

    print('inputFile:', inputFile)
    print('driverFile:', driverFile)    
    print('appKey: ', appKey)
    print('registryFile: ', registryFile)                              
    print('appDir: ', appDir)
    print('runDriver: ', runDriver)    
    print('')          

    #
    # get some dir paths, load input file and get data for app, appKey 
    #

    inputDir = os.path.dirname(inputFile)
    inputFileName = os.path.basename(inputFile)
    if inputDir != "":
        os.chdir(inputDir)

    with open(inputFileName, 'r') as f:
        inputs = json.load(f)

    if 'referenceDir' in inputs:
        reference_dir = input_data['referenceDir']
    else:
        reference_dir = inputDir;

    appData=""
    if appKey in inputs:
        appData = inputs[appKey]

    if 'models' not in appData:
        print('NO models in: ', appData)
        
        
    models = appData['models']
    modelToRun = appData['modelToRun']
    
    appsInMultiModel=[]
    appDataInMultiModel=[]
    appRunDataInMultiModel=[]    
    beliefs=[]
    sumBeliefs = 0;
    
    numModels = 0
    
    for model in models:
        belief = model['belief']
        appName = model['Application']
        appData = model['ApplicationData']
        appRunData = model['data']
        beliefs.append(belief)
        sumBeliefs = sumBeliefs + belief;
        appsInMultiModel.append(appName)
        appDataInMultiModel.append(appData)
        appRunDataInMultiModel.append(appRunData)
        numModels = numModels + 1

        
    for i in range(0,numModels):
        beliefs[i] = beliefs[i]/sumBeliefs
        
    appTypes=[appKey]
    
    parsedRegistry = (_parse_app_registry(registryFile, appTypes))
    appsRegistry = parsedRegistry[0][appKey]
    appDefaults=parsedRegistry[1]
    
    if runDriver == "False":

        print("MultiModel - setting up drivers")

        #
        # launch each application to create a driver file, each file seperate driver file & input
        #

        for i in range(0, numModels):
            
            appName = appsInMultiModel[i]
            application = appsRegistry[appName];
            application.set_pref(appDataInMultiModel[i], reference_dir)            

            asset_command_list = application.get_command_list(appDir)
            asset_command_list.append(u'--getRV')
            command = create_command(asset_command_list)
            # thinking to store aplications commands in a file so don't have to repeat this!

            # create input file for application        
            tmpInputFile = "MultiModel." + appKey + "." + str(i) + ".json";
            tmpDriverFile = "MultiModel." + appKey + "." + str(i);

            inputsTmp = deepcopy(inputs)
            inputsTmp[appKey] =  appRunDataInMultiModel[i];
            inputsTmp['Applications'][appKey] =  {
                "Application":appsInMultiModel[i],
                "ApplicationData":appDataInMultiModel[i]
            }
            print(inputsTmp['Applications'][appKey])
                  
            with open(tmpInputFile, "w") as outfile:
                json.dump(inputsTmp, outfile)

            # run the application
            asset_command_list = application.get_command_list(appDir)
            indexInputFile = asset_command_list.index('--workflowInput') + 1;
            asset_command_list[indexInputFile] = tmpInputFile;
            indexInputFile = asset_command_list.index('--driverFile') + 1;
            asset_command_list[indexInputFile] = tmpDriverFile;            
            asset_command_list.append(u'--osType')
            asset_command_list.append(osType)
            asset_command_list.append(u'--runType')
            asset_command_list.append(runType)                                
            command = create_command(asset_command_list)
            run_command(command);
            print('RUNNING:', command)

            
        #
        # update general input file with new random variables
        #

        randomVariables = inputs['randomVariables'];
        rvName = "MultiModel-"+appKey;
        rvValue="Rv.MultiModel-"+appKey;
        
        thisRV = {
            "distribution": "Discrete",
            "inputType": "Parameters",
            "name": rvName,
            "refCount": 1,
            "value": rvValue,
            "createdRun": True,            
            "variableClass": "Uncertain",
            "Weights":beliefs,
            "Values":[i+1 for i in range(0,numModels)]
        }
        randomVariables.append(thisRV);

        with open(inputFile, "w") as outfile:
            json.dump(inputs, outfile)        

        #
        # create driver file to invoke this application that runs the right driver
        #



        args_str = ' '.join(sys.argv[0:])
        command = sys.executable + ' ' + args_str + ' --runDriver True'
        with open(driverFile, "w") as dFile:
            print(command, file=dFile)
        
    else:

        print('NOTHING DOING .. need to run correct driver')
        # open params file to see which model to run

        modelToRun = 1
        
        # run the model
        modelToRun = modelToRun-1
        driverToRun =  "MultiModel." + appKey + "." + str(modelToRun);
        output = subprocess.check_output(driverToRun, shell=True)
        

    print("Finished MultiModelDriver")

    
if __name__ == '__main__':

    print('STARTING MultiModelDriver')
    print(sys.argv[1:])
    
    #Defining the command line arguments
    parser = argparse.ArgumentParser(
        "Run the MultiModel application.",
        allow_abbrev=False)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--driverFile', default=None)
    parser.add_argument('--workflowInput', default=None)
    parser.add_argument("--appKey", default=None)
    parser.add_argument('--runType', default=None)
    parser.add_argument('--osType', default=None)     
    parser.add_argument("--registry",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "WorkflowApplications.json"),
                        help="Path to file containing registered workflow applications")
    parser.add_argument('--runDriver', default="False")    
    parser.add_argument("-a", "--appDir",
                        default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        help="Absolute path to the local application directory.")
    parser.add_argument("-l", "--logFile",
                        default='log.txt',
                        help="Path where the log file will be saved.")

    args = parser.parse_args()        
        
    main(inputFile = args.workflowInput,
         driverFile = args.driverFile,         
         appKey = args.appKey,
         registryFile = args.registry,
         appDir = args.appDir,
         runType = args.runType,
         osType = args.osType,
         runDriver = args.runDriver);      
         
