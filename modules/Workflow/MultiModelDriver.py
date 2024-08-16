#  # noqa: INP001, D100
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

import argparse
import json
import os
import sys
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))  # noqa: PTH120

from whale.main import (
    _parse_app_registry,  # noqa: PLC2701
    create_command,
    run_command,
)


def main(inputFile, driverFile, appKey, registryFile, appDir, runType, osType):  # noqa: C901, N803, D103
    #
    # get some dir paths, load input file and get data for app, appKey
    #

    inputDir = os.path.dirname(inputFile)  # noqa: PTH120, N806
    inputFileName = os.path.basename(inputFile)  # noqa: PTH119, N806
    if inputDir != '':  # noqa: PLC1901
        os.chdir(inputDir)

    with open(inputFileName) as f:  # noqa: PLW1514, PTH123
        inputs = json.load(f)

    localAppDir = inputs['localAppDir']  # noqa: N806
    remoteAppDir = inputs['remoteAppDir']  # noqa: N806

    appDir = localAppDir  # noqa: N806
    if runType == 'runningRemote':
        appDir = remoteAppDir  # noqa: N806

    if 'referenceDir' in inputs:  # noqa: SIM401
        reference_dir = inputs['referenceDir']
    else:
        reference_dir = inputDir

    appData = {}  # noqa: N806
    if appKey in inputs:
        appData = inputs[appKey]  # noqa: N806

    if 'models' not in appData:
        print('NO models in: ', appData)  # noqa: T201
        raise KeyError(  # noqa: TRY003
            f'"models" not defined in data for "{appKey}" application in the input file "{inputFile}'  # noqa: EM102
        )

    if len(appData['models']) < 2:  # noqa: PLR2004
        raise RuntimeError(  # noqa: TRY003
            f'At least two models must be provided if the multimodel {appKey} application is used'  # noqa: EM102
        )

    models = appData['models']
    modelToRun = appData['modelToRun']  # noqa: N806

    appsInMultiModel = []  # noqa: N806
    appDataInMultiModel = []  # noqa: N806
    appRunDataInMultiModel = []  # noqa: N806
    beliefs = []
    sumBeliefs = 0  # noqa: N806

    numModels = 0  # noqa: N806

    for model in models:
        belief = model['belief']
        appName = model['Application']  # noqa: N806
        appData = model['ApplicationData']  # noqa: N806
        appRunData = model['data']  # noqa: N806
        beliefs.append(belief)
        sumBeliefs = sumBeliefs + belief  # noqa: N806, PLR6104
        appsInMultiModel.append(appName)
        appDataInMultiModel.append(appData)
        appRunDataInMultiModel.append(appRunData)
        numModels = numModels + 1  # noqa: N806, PLR6104

    for i in range(numModels):
        beliefs[i] = beliefs[i] / sumBeliefs  # noqa: PLR6104

    appTypes = [appKey]  # noqa: N806

    parsedRegistry = _parse_app_registry(registryFile, appTypes)  # noqa: N806
    appsRegistry = parsedRegistry[0][appKey]  # noqa: N806

    #
    # add RV to input file
    #

    randomVariables = inputs['randomVariables']  # noqa: N806
    rvName = 'MultiModel-' + appKey  # noqa: N806
    rvValue = 'RV.MultiModel-' + appKey  # noqa: N806

    thisRV = {  # noqa: N806
        'distribution': 'Discrete',
        'inputType': 'Parameters',
        'name': rvName,
        'refCount': 1,
        'value': rvValue,
        'createdRun': True,
        'variableClass': 'Uncertain',
        'Weights': beliefs,
        'Values': [i + 1 for i in range(numModels)],
    }
    randomVariables.append(thisRV)

    with open(inputFile, 'w') as outfile:  # noqa: PLW1514, PTH123
        json.dump(inputs, outfile)

    #
    # create driver file that runs the right driver
    #
    paramsFileName = 'params.in'  # noqa: N806
    multiModelString = 'MultiModel'  # noqa: N806
    exeFileName = 'runMultiModelDriver'  # noqa: N806
    if osType == 'Windows' and runType == 'runningLocal':
        driverFileBat = driverFile + '.bat'  # noqa: N806
        exeFileName = exeFileName + '.exe'  # noqa: N806, PLR6104
        with open(driverFileBat, 'wb') as f:  # noqa: FURB103, PTH123
            f.write(
                bytes(
                    os.path.join(appDir, 'applications', 'Workflow', exeFileName)  # noqa: PTH118
                    + f' {paramsFileName} {driverFileBat} {multiModelString}',
                    'UTF-8',
                )
            )
    elif runType == 'runningRemote':
        with open(driverFile, 'wb') as f:  # noqa: PTH123
            f.write(
                appDir
                + '/applications/Workflow/'
                + exeFileName
                + f' {paramsFileName} {driverFile} {multiModelString}',
                'UTF-8',
            )
    else:
        with open(driverFile, 'wb') as f:  # noqa: FURB103, PTH123
            f.write(
                bytes(
                    os.path.join(appDir, 'applications', 'Workflow', exeFileName)  # noqa: PTH118
                    + f' {paramsFileName} {driverFile} {multiModelString}',
                    'UTF-8',
                )
            )

    for modelToRun in range(numModels):  # noqa: N806
        #
        # run the app to create the driver file for each model
        #

        appName = appsInMultiModel[modelToRun]  # noqa: N806
        application = appsRegistry[appName]
        application.set_pref(appDataInMultiModel[modelToRun], reference_dir)

        #
        # create input file for application
        #

        modelInputFile = f'MultiModel_{modelToRun + 1}_' + inputFile  # noqa: N806
        modelDriverFile = f'MultiModel_{modelToRun + 1}_' + driverFile  # noqa: N806

        inputsTmp = deepcopy(inputs)  # noqa: N806
        inputsTmp[appKey] = appRunDataInMultiModel[modelToRun]
        inputsTmp['Applications'][appKey] = {
            'Application': appsInMultiModel[modelToRun],
            'ApplicationData': appDataInMultiModel[modelToRun],
        }

        with open(modelInputFile, 'w') as outfile:  # noqa: PLW1514, PTH123
            json.dump(inputsTmp, outfile)

        #
        # run the application to create driver file
        #

        asset_command_list = application.get_command_list(localAppDir)
        indexInputFile = asset_command_list.index('--workflowInput') + 1  # noqa: N806
        asset_command_list[indexInputFile] = modelInputFile
        indexInputFile = asset_command_list.index('--driverFile') + 1  # noqa: N806
        asset_command_list[indexInputFile] = modelDriverFile
        asset_command_list.append('--osType')
        asset_command_list.append(osType)
        asset_command_list.append('--runType')
        asset_command_list.append(runType)
        asset_command_list.append('--modelIndex')
        asset_command_list.append(modelToRun + 1)
        command = create_command(asset_command_list)
        run_command(command)


if __name__ == '__main__':
    #
    # Defining the command line arguments
    #

    parser = argparse.ArgumentParser(
        'Run the MultiModel application.', allow_abbrev=False
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--driverFile', default=None)
    parser.add_argument('--workflowInput', default=None)
    parser.add_argument('--appKey', default=None)
    parser.add_argument('--runType', default=None)
    parser.add_argument('--osType', default=None)
    parser.add_argument(
        '--registry',
        default=os.path.join(  # noqa: PTH118
            os.path.dirname(os.path.abspath(__file__)),  # noqa: PTH100, PTH120
            'WorkflowApplications.json',
        ),
        help='Path to file containing registered workflow applications',
    )
    # parser.add_argument('--runDriver', default="False")
    parser.add_argument(
        '-a',
        '--appDir',
        default=os.path.dirname(  # noqa: PTH120
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # noqa: PTH100, PTH120
        ),
        help='Absolute path to the local application directory.',
    )
    parser.add_argument(
        '-l',
        '--logFile',
        default='log.txt',
        help='Path where the log file will be saved.',
    )

    args = parser.parse_args()

    #
    # run the app
    #

    main(
        inputFile=args.workflowInput,
        driverFile=args.driverFile,
        appKey=args.appKey,
        registryFile=args.registry,
        appDir=args.appDir,
        runType=args.runType,
        osType=args.osType,
    )
