# -*- coding: utf-8 -*-
#
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
# fmk
#

import os, sys
import argparse, json
import subprocess
import shutil

from pathlib import Path

#
# some filePath and python exe stuff
#

thisDir = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
mainDir = thisDir.parents[1]
mainDir = thisDir.parents[1]
currentDir=os.getcwd()

pythonEXE = sys.executable

thisDir = str(thisDir)
mainDir = str(mainDir)
currentDir = str(currentDir)

print(f"thisDir: {thisDir}")
print(f"mainDir: {mainDir}")
print(f"currentDir: {currentDir}")

def runHazardSimulation(inputFILE):

    # log_msg('Startring simulation script...')

    sys.path.insert(0, os.getcwd())

    #
    # open input & parse json
    #
    
    print(f'inputFILE: {inputFILE}')
    with open(inputFILE, 'r') as f:
        inputJSON = json.load(f)

    #
    # read needed input data 
    #
    
    unitData = inputJSON['units']
    inputApplications = inputJSON['Applications']
    hazardApplication = inputApplications['Hazard']
    regionalMappingApplication = inputApplications['RegionalMapping']
    uqApplication = inputApplications['UQ']
    
    hazardAppData = hazardApplication['ApplicationData']

    soilFile = hazardAppData["soilGridParametersFile"]
    soilPath = hazardAppData["soilParametersPath"]
    responseScript = hazardAppData["siteResponseScript"]
    scriptPath = hazardAppData["scriptPath"]
    filters = hazardAppData["filter"]
    eventFile = hazardAppData["inputEventFile"]
    motionDir = hazardAppData["inputMotionDir"]
    outputDir = hazardAppData["outputMotionDir"]    

    # now create an input for siteResponseWHALE
    
    srtFILE = "sc_srt.json"

    outputs = dict(
        EDP =  True,
        DM =  False,
        DV =  False,
        every_realization = False
    )

    edpApplication = dict( 
        Application = 'DummyEDP',
        ApplicationData = dict()
    )

    eventApp = dict(
        EventClassification =  "Earthquake",
        Application = "RegionalSiteResponse",
        ApplicationData = dict(
            pathEventData =  motionDir,
            mainScript = responseScript,
            modelPath =  scriptPath,
            ndm =  3
        )
    )

    regionalMappingAppData = regionalMappingApplication['ApplicationData']
    regionalMappingAppData['filenameEVENTgrid']=eventFile
    
    buildingApplication = dict (
        Application = "CSV_to_BIM",
        ApplicationData = dict (
            buildingSourceFile =  f'{soilPath}{soilFile}',
            filter = filters
        )
    )

    Applications = dict(
        UQ = uqApplication,
        RegionalMapping = regionalMappingApplication,
        Events = [eventApp],
        EDP = edpApplication,
        Building = buildingApplication
    )

    srt = dict(
        units = unitData,
        outputs = outputs,
        Applications = Applications
    )
    
    with open(srtFILE, 'w') as f:
        json.dump(srt, f, indent=2)

    #
    # now invoke siteResponseWHALE
    #

    inputDir = currentDir + "/input_data"
    tmpDir = currentDir + "/input_data/siteResponseRunningDir"    
    
    print(f'RUNNING {pythonEXE} {mainDir}/Workflow/siteResponseWHALE.py ./sc_srt.json --registry {mainDir}/Workflow/WorkflowApplications.json --referenceDir {inputDir} -w {tmpDir}')

    subprocess.run([pythonEXE, mainDir+"/Workflow/siteResponseWHALE.py", "./sc_srt.json","--registry", mainDir+"/Workflow/WorkflowApplications.json", "--referenceDir", inputDir, "-w", tmpDir])

    #
    # gather results, creating new EventGrid file
    # and moving all the motions created
    #

    outputMotionDir = currentDir + "/input_data/" + outputDir;

    print(f'RUNNING {pythonEXE} {mainDir}/createEVENT/siteResponse/createGM4BIM.py -i  {tmpDir} -o  {outputMotionDir} --removeInput')
    
    subprocess.run([pythonEXE, mainDir+"/createEVENT/siteResponse/createGM4BIM.py", "-i", tmpDir, "-o", outputMotionDir])

    #subprocess.run([pythonEXE, mainDir+"/createEVENT/siteResponse/createGM4BIM.py", "-i", tmpDir, "-o", outputMotionDir], "--removeInput")    


    #
    # remove tmp dir
    #

    try:
        shutil.rmtree(tmpDir)
    except OSError as e:
            print("Error: %s : %s" % (tmpDir, e.strerror))
    
    #
    # modify inputFILE to provide new event file for regional mapping 
    #

    regionalMappingAppData = regionalMappingApplication['ApplicationData']
    regionalMappingAppData['filenameEVENTgrid']=f'{outputDir}/EventGrid.csv'

    with open(inputFILE, 'w') as f:
        json.dump(inputJSON, f, indent=2)
    
    #
    # we are done
    #
    
    #log_msg('Simulation script finished.')
    return 0

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input',default=None)
    args = parser.parse_args()

    runHazardSimulation(args.input)
