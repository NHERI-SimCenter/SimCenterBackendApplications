# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
#
# This file is part of the SimCenter Backend Applications
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
# this file. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors: 
# Sang-ri
#

# Description:
# Read SAM and GI, add it to params.in, run surrogate model, and write the results to EDP.json
#
#


import sys
import os
import subprocess
import importlib
import argparse
import json
from pathlib import Path

def main(aimName,samName, evtName,
         edpName, simName, getRV):
    
    #
    # Find the GI and SAM files
    #

    with open(aimName, 'r') as f:
        root_AIM = json.load(f)
        GI = root_AIM['GeneralInformation']     

    with open(samName, 'r') as f:
        SAM = json.load(f)

    #
    # Get user-uploaded filter script
    #
    # sy - so far works only for single model
    filterFileName = root_AIM['Simulation']['filterFileName']
    filateFilePath = root_AIM['Simulation']['filterFilePath']
    sys.path.insert(0, filateFilePath)
    analysis_script = importlib.__import__(filterFileName[:-3], globals(), locals(), ['model_distributor',], 0)
    model_distributor = analysis_script.model_distributor
    modelName = model_distributor(GI,SAM)

    if getRV:
        runDefault(root_AIM, aimName,samName, evtName, edpName, simName, getRV)
        return

    #
    # Parse filter file
    #

    if modelName.lower() == "none":
        pass
    elif modelName.lower() =="error":
        pass
    elif modelName.lower() =="default":        
        runDefault(root_AIM, aimName,samName, evtName, edpName, simName)
    else:
        runSurrogate(modelName, GI, SAM, root_AIM, aimName, edpName)

def runDefault(root_AIM, aimName,samName, evtName, edpName, simName, getRV=False):

        #
        # Find app name
        #
        mySimAppName = root_AIM['Simulation']['DefaultAnalysis']['Buildings']['Application']

        #
        # overwrite with default AIM.json file  
        #
        root_AIM['Simulation'] = root_AIM['Simulation']['DefaultAnalysis']['Buildings']

        currentDir = os.getcwd()
        newAimName = os.path.join(currentDir,os.path.basename(aimName))
        
        with open(newAimName, 'w') as f:
            json_object = json.dumps(root_AIM)
            f.write(json_object)
        #
        # overwrite with default AIM.json file  
        #
        s=[os.path.dirname( __file__ ),'..','..','Workflow','WorkflowApplications.json']
        workflowAppJsonPath = os.path.join(*s)
        with open(workflowAppJsonPath) as f:
            workflowAppDict = json.load(f)
            appList = workflowAppDict["SimulationApplications"]["Applications"]
            myApp = next(item for item in appList if item["Name"] == mySimAppName)
            s = [os.path.dirname( __file__ ),'..','..','..', os.path.dirname(myApp["ExecutablePath"])]
            mySimAppPath = os.path.join(*s)
            mySimAppName = os.path.basename(myApp["ExecutablePath"])

        #
        # run correct backend app
        #
        # print(newAimName)
        sys.path.insert(0, mySimAppPath)
        sim_module = importlib.__import__(mySimAppName[:-3], globals(), locals(), ['main'], 0)

        if getRV:
            sim_module.main(["--filenameAIM", newAimName, "--filenameSAM", samName, "--filenameEVENT", evtName, "--filenameEDP", edpName, "--filenameSIM", simName, "--getRV"])

        else:
            sim_module.main(["--filenameAIM", newAimName, "--filenameSAM", samName, "--filenameEVENT", evtName, "--filenameEDP", edpName, "--filenameSIM", simName])
        
        return

def runSurrogate(modelName, GI, SAM, root_AIM, aimName, edpName):

        #
        # Augment to params.in file
        #

        GIkeys = ["Latitude","Longitude","NumberOfStories","YearBuilt","OccupancyClass","StructureType","PlanArea","ReplacementCost"]
        SAMkeys_properties = ["dampingRatio","K0","Sy","eta","C","gamma","alpha","beta","omega","eta_soft","a_k"]
        SAMkeys_nodes = ["mass"]

        with open('params.in', 'r') as f:
            paramsStr = f.read()
        nAddParams =0

        for key in GI:
            if key in GIkeys:
                val = GI[key]
                if not isinstance(val, str):
                    paramsStr += "{} {}\n".format(key, val)
                else:
                    paramsStr += "{} \"{}\"\n".format(key, val)
                nAddParams +=1

        # For damping
        for key in SAM["Properties"]:
            if key in SAMkeys_properties:
                val = SAM["Properties"][key]
                if not isinstance(val, str):
                    paramsStr += "{} {}\n".format(key, val)
                else:
                    paramsStr += "{} \"{}\"\n".format(key, val)
                nAddParams +=1

        # For material properties
        for SAM_elem in SAM["Properties"]["uniaxialMaterials"]:
            for key in SAM_elem:
                if key in SAMkeys_properties:
                    val = SAM_elem[key]
                    if not isinstance(val, str):
                        paramsStr += "{}-{} {}\n".format(key, SAM_elem["name"],val)
                    else:
                        paramsStr += "{}-{} \"{}\"\n".format(key, SAM_elem["name"],val)
                    nAddParams +=1

        # For mass
        for SAM_node in SAM["Geometry"]["nodes"]:
            for key in SAM_node:
                if key in SAMkeys_nodes:
                    val = SAM_node[key]
                    if not isinstance(val, str):
                        paramsStr += "{}-{} {}\n".format(key, SAM_node["name"], val)
                    else:
                        paramsStr += "{}-{} \"{}\"\n".format(key, SAM_node["name"], val)
                    nAddParams +=1


        stringList = paramsStr.split("\n")
        stringList.remove(stringList[0]) # remove # params (will be added later)
        stringList = set(stringList) # remove duplicates
        stringList = [i for i in stringList if i] # remove empty
        stringList = [str(len(stringList))]+stringList
        with open('params.in', 'w') as f:
            f.write("\n".join(stringList))

        f.close()

        #
        # get sur model info
        #

        surFileName = None
        for model in root_AIM['Simulation']['Models']:
             if model["modelName"] == modelName:
                surFileName = model["fileName"]

        if surFileName is None:
            print("surrogate model {} is not found".format(modelName))
            exit(-1)

        #
        # find surrogate model prediction app
        #

        s=[os.path.dirname( __file__ ),'..','..','Workflow','WorkflowApplications.json']
        workflowAppJsonPath = os.path.join(*s)
        with open(workflowAppJsonPath) as f:
            workflowAppDict = json.load(f)
            appList = workflowAppDict["SimulationApplications"]["Applications"]
            simAppName = "SurrogateSimulation"
            myApp = next(item for item in appList if item["Name"] == simAppName)
            s = [os.path.dirname( __file__ ),'..','..','..', os.path.dirname(myApp["ExecutablePath"])]
            mySurrogatePath = os.path.join(*s)
            mySurrogateName = os.path.basename(myApp["ExecutablePath"])

        #
        # import surrogate functions
        #

        root_AIM['Applications']['Modeling']['ApplicationData']['MS_Path'] = ""
        root_AIM['Applications']['Modeling']['ApplicationData']['postprocessScript'] = ""
        root_AIM['Applications']['Modeling']['ApplicationData']['mainScript'] = r"..\\..\\..\\..\\input_data\\"+surFileName
        
        currentDir = os.getcwd()
        newAimName = os.path.join(currentDir,os.path.basename(aimName))
        with open(newAimName, 'w') as f:
            json_object = json.dumps(root_AIM)
            f.write(json_object)

        sys.path.insert(0, mySurrogatePath)
        sur_module = importlib.__import__(mySurrogateName[:-3], globals(), locals(), ['run_surrogateGP','write_EDP'], 0)


        #
        # run prediction
        #

        sur_module.run_surrogateGP(newAimName, edpName)

        #
        # write EDP file
        #

        sur_module.write_EDP(newAimName, edpName)

        return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameAIM')
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameSAM')
    parser.add_argument('--filenameEDP')
    parser.add_argument('--filenameSIM')
    #parser.add_argument('--defaultModule', default=None)
    #parser.add_argument('--fileName', default=None)
    #parser.add_argument('--filePath', default=None)
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    sys.exit(main(
        args.filenameAIM, args.filenameSAM,  args.filenameEVENT,
        args.filenameEDP, args.filenameSIM, args.getRV))


