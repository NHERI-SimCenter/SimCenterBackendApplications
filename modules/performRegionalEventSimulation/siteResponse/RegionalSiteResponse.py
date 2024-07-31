#  # noqa: INP001, D100
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

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

#
# some filePath and python exe stuff
#

thisDir = Path(os.path.dirname(os.path.abspath(__file__))).resolve()  # noqa: PTH100, PTH120, N816
mainDir = thisDir.parents[1]  # noqa: N816
mainDir = thisDir.parents[1]  # noqa: N816
currentDir = os.getcwd()  # noqa: PTH109, N816

pythonEXE = sys.executable  # noqa: N816

thisDir = str(thisDir)  # noqa: N816
mainDir = str(mainDir)  # noqa: N816
currentDir = str(currentDir)  # noqa: N816

print(f'thisDir: {thisDir}')  # noqa: T201
print(f'mainDir: {mainDir}')  # noqa: T201
print(f'currentDir: {currentDir}')  # noqa: T201


def runHazardSimulation(inputFILE):  # noqa: ANN001, ANN201, N802, N803, D103
    # log_msg('Startring simulation script...')  # noqa: ERA001

    sys.path.insert(0, os.getcwd())  # noqa: PTH109

    #
    # open input & parse json
    #

    print(f'inputFILE: {inputFILE}')  # noqa: T201
    with open(inputFILE) as f:  # noqa: PTH123
        inputJSON = json.load(f)  # noqa: N806

    #
    # read needed input data
    #

    unitData = inputJSON['units']  # noqa: N806
    inputApplications = inputJSON['Applications']  # noqa: N806
    hazardApplication = inputApplications['Hazard']  # noqa: N806
    regionalMappingApplication = inputApplications['RegionalMapping']  # noqa: N806
    uqApplication = inputApplications['UQ']  # noqa: N806

    hazardAppData = hazardApplication['ApplicationData']  # noqa: N806

    soilFile = hazardAppData['soilGridParametersFile']  # noqa: N806
    soilPath = hazardAppData['soilParametersPath']  # noqa: N806
    responseScript = hazardAppData['siteResponseScript']  # noqa: N806
    scriptPath = hazardAppData['scriptPath']  # noqa: N806
    filters = hazardAppData['filter']
    eventFile = hazardAppData['inputEventFile']  # noqa: N806
    motionDir = hazardAppData['inputMotionDir']  # noqa: N806
    outputDir = hazardAppData['outputMotionDir']  # noqa: N806

    # now create an input for siteResponseWHALE

    srtFILE = 'sc_srt.json'  # noqa: N806

    outputs = dict(EDP=True, DM=False, DV=False, every_realization=False)  # noqa: C408

    edpApplication = dict(Application='DummyEDP', ApplicationData=dict())  # noqa: C408, N806

    eventApp = dict(  # noqa: C408, N806
        EventClassification='Earthquake',
        Application='RegionalSiteResponse',
        ApplicationData=dict(  # noqa: C408
            pathEventData=motionDir,
            mainScript=responseScript,
            modelPath=scriptPath,
            ndm=3,
        ),
    )

    regionalMappingAppData = regionalMappingApplication['ApplicationData']  # noqa: N806
    regionalMappingAppData['filenameEVENTgrid'] = eventFile

    buildingApplication = dict(  # noqa: C408, N806
        Application='CSV_to_BIM',
        ApplicationData=dict(  # noqa: C408
            buildingSourceFile=f'{soilPath}{soilFile}', filter=filters
        ),
    )

    Applications = dict(  # noqa: C408, N806
        UQ=uqApplication,
        RegionalMapping=regionalMappingApplication,
        Events=[eventApp],
        EDP=edpApplication,
        Building=buildingApplication,
    )

    srt = dict(units=unitData, outputs=outputs, Applications=Applications)  # noqa: C408

    with open(srtFILE, 'w') as f:  # noqa: PTH123
        json.dump(srt, f, indent=2)

    #
    # now invoke siteResponseWHALE
    #

    inputDir = currentDir + '/input_data'  # noqa: N806
    tmpDir = currentDir + '/input_data/siteResponseRunningDir'  # noqa: N806

    print(  # noqa: T201
        f'RUNNING {pythonEXE} {mainDir}/Workflow/siteResponseWHALE.py ./sc_srt.json --registry {mainDir}/Workflow/WorkflowApplications.json --referenceDir {inputDir} -w {tmpDir}'  # noqa: E501
    )

    subprocess.run(  # noqa: S603
        [
            pythonEXE,
            mainDir + '/Workflow/siteResponseWHALE.py',
            './sc_srt.json',
            '--registry',
            mainDir + '/Workflow/WorkflowApplications.json',
            '--referenceDir',
            inputDir,
            '-w',
            tmpDir,
        ],
        check=False,
    )

    #
    # gather results, creating new EventGrid file
    # and moving all the motions created
    #

    outputMotionDir = currentDir + '/input_data/' + outputDir  # noqa: N806
    print(  # noqa: T201
        f'RUNNING {pythonEXE} {mainDir}/createEVENT/siteResponse/createGM4BIM.py -i  {tmpDir} -o  {outputMotionDir} --removeInput'  # noqa: E501
    )

    subprocess.run(  # noqa: S603
        [
            pythonEXE,
            mainDir + '/createEVENT/siteResponse/createGM4BIM.py',
            '-i',
            tmpDir,
            '-o',
            outputMotionDir,
        ],
        check=False,
    )

    # subprocess.run([pythonEXE, mainDir+"/createEVENT/siteResponse/createGM4BIM.py", "-i", tmpDir, "-o", outputMotionDir], "--removeInput")  # noqa: ERA001, E501

    #
    # remove tmp dir
    #

    try:
        shutil.rmtree(tmpDir)
    except OSError as e:
        print('Error: %s : %s' % (tmpDir, e.strerror))  # noqa: T201, UP031

    #
    # modify inputFILE to provide new event file for regional mapping
    #

    regionalMappingAppData = regionalMappingApplication['ApplicationData']  # noqa: N806
    regionalMappingAppData['filenameEVENTgrid'] = f'{outputDir}/EventGrid.csv'

    with open(inputFILE, 'w') as f:  # noqa: PTH123
        json.dump(inputJSON, f, indent=2)

    #
    # we are done
    #

    # log_msg('Simulation script finished.')  # noqa: ERA001
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', default=None)
    args = parser.parse_args()

    runHazardSimulation(args.input)
