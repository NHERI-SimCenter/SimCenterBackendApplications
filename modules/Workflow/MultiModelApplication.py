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
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import whale.main as whale
from whale.main import log_msg, log_div, _parse_app_registry, create_command, run_command

def main(inputFile,
         appKey,
         getRV,
         samFile,
         evtFile,
         edpFile,
         simFile,
         registryFile,
         appDir) :
    
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
        reference_dir = inputs['referenceDir']
    else:
        reference_dir = inputDir

    appData={}
    if appKey in inputs:
        appData = inputs[appKey]
    else:
        raise KeyError(f'No data for "{appKey}" application in the input file "{inputFile}"')

    eventApp = False;

    if appKey == "Events":
        eventApp = True;
        appData = appData[0]


    print('appKEY: ', appKey)
    print('appDATA: ', appData)
    print('HELLO ')        
    
    if 'models' not in appData:
        print('NO models in: ', appData)
        raise KeyError(f'"models" not defined in data for "{appKey}" application in the input file "{inputFile}')
        
    if len(appData['models']) < 2:
        raise RuntimeError(f"At least two models must be provided if the multimodel {appKey} application is used")


    models = appData['models']
    modelToRun = appData['modelToRun']

    if not getRV:

        #
        # make sure not still a string, if so try reading from params.in
        # 
        
        if isinstance(modelToRun, str):
            rvName = "MultiModel-"+appKey
            # if not here, try opening params.in and getting var from there
            with open("params.in", 'r') as params:
                # Read the file line by line
                for line in params:
                    values = line.strip().split()
                    print(values)
                    if values[0] == rvName:
                        modelToRun = values[1]
                        
        modelToRun = int(float(modelToRun))
    
    appsInMultiModel=[]
    appDataInMultiModel=[]
    appRunDataInMultiModel=[]    
    beliefs=[]
    sumBeliefs = 0
    
    numModels = 0
    
    for model in models:
        belief = model['belief']
        appName = model['Application']
        appData = model['ApplicationData']
        appRunData = model['data']
        beliefs.append(belief)
        sumBeliefs = sumBeliefs + belief
        appsInMultiModel.append(appName)
        appDataInMultiModel.append(appData)
        appRunDataInMultiModel.append(appRunData)
        numModels = numModels + 1

    for i in range(0,numModels):
        beliefs[i] = beliefs[i]/sumBeliefs

    #
    # parse WorkflowApplications to get possible applications
    # need the 2 ifs, as appKey needs to be Events, but switch in WorkflowApplications needs to be Event!
    #
    
    if appKey == "Events":
        appTypes=["Event"]
    else:
        appTypes=[appKey]

    parsedRegistry = (_parse_app_registry(registryFile, appTypes))
    
    if appKey == "Events":    
        appsRegistry = parsedRegistry[0]["Event"]
    else:
        appsRegistry = parsedRegistry[0][appKey]        

    #
    # now we run the application
    #   if getRV we have to run each & collect the RVs
    #   if !getRV we run the single application chosen
    #
    
    if getRV:
        
        print("MultiModel - getRV")

        #
        # launch each application with getRV and add any new RandomVariable
        # add randomvariable for MultiModel itself, to launch application
        # need to create temp inputfile for just that application, 
        #

        for i in range(0, numModels):
            appName = appsInMultiModel[i]
            print('appsRegistry:', appsRegistry)
            application = appsRegistry[appName]
            application.set_pref(appDataInMultiModel[i], reference_dir)            

            asset_command_list = application.get_command_list(appDir)
            asset_command_list.append(u'--getRV')
            command = create_command(asset_command_list)
            # thinking to store aplications commands in a file so don't have to repeat this!

        #
        # update input file
        #

        #
        # for NOW, add RV to input file
        #

        randomVariables = inputs['randomVariables']
        rvName = "MultiModel-"+appKey
        rvValue="RV.MultiModel-"+appKey
        # nrv = len(randomVariables)
        
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
        randomVariables.append(thisRV)

        #
        # read corr and append row/cols
        #

        # if 'correlationMatrix' in inputs:
        #     corrVec = inputs['correlationMatrix']
        #     corrMat = np.reshape(corrVec, (nrv, nrv))
        #     newCorrMat = np.identity(nrv+1)
        #     newCorrMat[0:nrv,0:nrv] = corrMat
        #     inputs['correlationMatrix'] = newCorrMat.flatten().tolist()


        with open(inputFile, "w") as outfile:
            json.dump(inputs, outfile)        

        print('UPDATING INPUT FILE:', inputFile)
        
        #
        # for now just run the last model (works in sWHALE for all apps that don't create RV, i.e. events)
        #

        # create input file for application
        
        tmpFile = "MultiModel." + appKey + ".json"
        inputs[appKey] =  appRunDataInMultiModel[numModels-1]
        
        with open(tmpFile, "w") as outfile:
            json.dump(inputs, outfile)

        # run the application
        asset_command_list = application.get_command_list(appDir)
        indexInputFile = asset_command_list.index('--filenameAIM') + 1
        asset_command_list[indexInputFile] = tmpFile
        asset_command_list.append(u'--getRV')        
        command = create_command(asset_command_list)
        run_command(command)
        print('RUNNING --getRV:', command)
            
    else:
        
        print("MultiModel - run")
        modelToRun = modelToRun - 1
        # get app data given model
        appName = appsInMultiModel[modelToRun]
        application = appsRegistry[appName]
        application.set_pref(appDataInMultiModel[modelToRun], reference_dir)            

        # create modified input file for app
        tmpFile = "MultiModel." + appKey + ".json"

        #if appKey == "Events":
        #    inputs["Events"][0]=appRunDataInMultiModel[modelToRun]

        #else:
        #    inputs[appKey] =  appRunDataInMultiModel[modelToRun]
        inputs[appKey] =  appRunDataInMultiModel[modelToRun]

        print('model to run:', modelToRun)

        with open(tmpFile, "w") as outfile:
            json.dump(inputs, outfile)

        print("INPUTS", inputs)
        
        # run application
        asset_command_list = application.get_command_list(appDir)
        indexInputFile = asset_command_list.index('--filenameAIM') + 1
        asset_command_list[indexInputFile] = tmpFile     
        command = create_command(asset_command_list)        
        run_command(command)
        print('RUNNING:', command)        

    print("Finished MultiModelApplication")

if __name__ == '__main__':

    #Defining the command line arguments
    parser = argparse.ArgumentParser(
        "Run the MultiModel application.",
        allow_abbrev=False)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameAIM', default=None)
    parser.add_argument('--filenameSAM', default='NA')
    parser.add_argument('--filenameEVENT', default='NA')
    parser.add_argument('--filenameEDP', default='NA')
    parser.add_argument('--filenameSIM', default='NA')
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    parser.add_argument("--appKey", default=None)
    parser.add_argument("--registry",
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "WorkflowApplications.json"),
                        help="Path to file containing registered workflow applications")
    parser.add_argument("-a", "--appDir",
                        default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        help="Absolute path to the local application directory.")
    parser.add_argument("-l", "--logFile",
                        default='log.txt',
                        help="Path where the log file will be saved.")

    args, unknown  = parser.parse_known_args()        
    

    main(inputFile = args.filenameAIM,
         appKey = args.appKey,
         getRV = args.getRV,
         samFile = args.filenameSAM,
         evtFile = args.filenameEVENT,
         edpFile = args.filenameEDP,
         simFile = args.filenameSIM,
         registryFile = args.registry,
         appDir = args.appDir)

         
