#  # noqa: INP001, D100
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


import argparse
import importlib
import json
import os
import sys


def main(aimName, samName, evtName, edpName, simName, getRV):  # noqa: N803, D103
    #
    # Find the GI and SAM files
    #

    with open(aimName, encoding='utf-8') as f:  # noqa: PTH123
        root_AIM = json.load(f)  # noqa: N806
        GI = root_AIM['GeneralInformation']  # noqa: N806

    with open(samName, encoding='utf-8') as f:  # noqa: PTH123
        SAM = json.load(f)  # noqa: N806

    #
    # Get user-uploaded filter script
    #
    # sy - so far works only for single model
    filterFileName = root_AIM['Simulation']['filterFileName']  # noqa: N806
    filateFilePath = root_AIM['Simulation']['filterFilePath']  # noqa: N806
    sys.path.insert(0, filateFilePath)
    analysis_script = importlib.__import__(
        filterFileName[:-3],
        globals(),
        locals(),
        [
            'model_distributor',
        ],
        0,
    )
    model_distributor = analysis_script.model_distributor
    modelName = model_distributor(GI, SAM)  # noqa: N806

    if getRV:
        runDefault(root_AIM, aimName, samName, evtName, edpName, simName, getRV)
        return

    #
    # Parse filter file
    #

    if modelName.lower() == 'none' or modelName.lower() == 'error':
        pass
    elif modelName.lower() == 'default':
        runDefault(root_AIM, aimName, samName, evtName, edpName, simName)
    else:
        runSurrogate(modelName, GI, SAM, root_AIM, aimName, edpName)


def runDefault(root_AIM, aimName, samName, evtName, edpName, simName, getRV=False):  # noqa: FBT002, N802, N803, D103
    #
    # Find app name
    #
    mySimAppName = root_AIM['Simulation']['DefaultAnalysis']['Buildings'][  # noqa: N806
        'Application'
    ]

    #
    # overwrite with default AIM.json file
    #
    root_AIM['Simulation'] = root_AIM['Simulation']['DefaultAnalysis']['Buildings']

    currentDir = os.getcwd()  # noqa: PTH109, N806
    newAimName = os.path.join(currentDir, os.path.basename(aimName))  # noqa: PTH118, PTH119, N806

    with open(newAimName, 'w', encoding='utf-8') as f:  # noqa: FURB103, PTH123
        json_object = json.dumps(root_AIM)
        f.write(json_object)
    #
    # overwrite with default AIM.json file
    #
    s = [
        os.path.dirname(__file__),  # noqa: PTH120
        '..',
        '..',
        'Workflow',
        'WorkflowApplications.json',
    ]
    workflowAppJsonPath = os.path.join(*s)  # noqa: PTH118, N806
    with open(workflowAppJsonPath, encoding='utf-8') as f:  # noqa: PTH123
        workflowAppDict = json.load(f)  # noqa: N806
        appList = workflowAppDict['SimulationApplications']['Applications']  # noqa: N806
        myApp = next(item for item in appList if item['Name'] == mySimAppName)  # noqa: N806
        s = [
            os.path.dirname(__file__),  # noqa: PTH120
            '..',
            '..',
            '..',
            os.path.dirname(myApp['ExecutablePath']),  # noqa: PTH120
        ]
        mySimAppPath = os.path.join(*s)  # noqa: PTH118, N806
        mySimAppName = os.path.basename(myApp['ExecutablePath'])  # noqa: PTH119, N806

    #
    # run correct backend app
    #
    # print(newAimName)
    sys.path.insert(0, mySimAppPath)
    sim_module = importlib.__import__(
        mySimAppName[:-3], globals(), locals(), ['main'], 0
    )

    if getRV:
        sim_module.main(
            [
                '--filenameAIM',
                newAimName,
                '--filenameSAM',
                samName,
                '--filenameEVENT',
                evtName,
                '--filenameEDP',
                edpName,
                '--filenameSIM',
                simName,
                '--getRV',
            ]
        )

    else:
        sim_module.main(
            [
                '--filenameAIM',
                newAimName,
                '--filenameSAM',
                samName,
                '--filenameEVENT',
                evtName,
                '--filenameEDP',
                edpName,
                '--filenameSIM',
                simName,
            ]
        )


def runSurrogate(modelName, GI, SAM, root_AIM, aimName, edpName):  # noqa: C901, N802, N803, D103
    #
    # Augment to params.in file
    #

    GIkeys = [  # noqa: N806
        'Latitude',
        'Longitude',
        'NumberOfStories',
        'YearBuilt',
        'OccupancyClass',
        'StructureType',
        'PlanArea',
        'ReplacementCost',
    ]
    SAMkeys_properties = [  # noqa: N806
        'dampingRatio',
        'K0',
        'Sy',
        'eta',
        'C',
        'gamma',
        'alpha',
        'beta',
        'omega',
        'eta_soft',
        'a_k',
    ]
    SAMkeys_nodes = ['mass']  # noqa: N806

    with open('params.in') as f:  # noqa: FURB101, PLW1514, PTH123
        paramsStr = f.read()  # noqa: N806
    nAddParams = 0  # noqa: N806

    for key in GI:
        if key in GIkeys:
            val = GI[key]
            if not isinstance(val, str):
                paramsStr += f'{key} {val}\n'  # noqa: N806
            else:
                paramsStr += f'{key} "{val}"\n'  # noqa: N806
            nAddParams += 1  # noqa: N806

    # For damping
    for key in SAM['Properties']:
        if key in SAMkeys_properties:
            val = SAM['Properties'][key]
            if not isinstance(val, str):
                paramsStr += f'{key} {val}\n'  # noqa: N806
            else:
                paramsStr += f'{key} "{val}"\n'  # noqa: N806
            nAddParams += 1  # noqa: N806

    # For material properties
    for SAM_elem in SAM['Properties']['uniaxialMaterials']:  # noqa: N806
        for key in SAM_elem:
            if key in SAMkeys_properties:
                val = SAM_elem[key]
                if not isinstance(val, str):
                    paramsStr += '{}-{} {}\n'.format(key, SAM_elem['name'], val)  # noqa: N806
                else:
                    paramsStr += '{}-{} "{}"\n'.format(key, SAM_elem['name'], val)  # noqa: N806
                nAddParams += 1  # noqa: N806

    # For mass
    for SAM_node in SAM['Geometry']['nodes']:  # noqa: N806
        for key in SAM_node:
            if key in SAMkeys_nodes:
                val = SAM_node[key]
                if not isinstance(val, str):
                    paramsStr += '{}-{} {}\n'.format(key, SAM_node['name'], val)  # noqa: N806
                else:
                    paramsStr += '{}-{} "{}"\n'.format(key, SAM_node['name'], val)  # noqa: N806
                nAddParams += 1  # noqa: N806

    stringList = paramsStr.split('\n')  # noqa: N806
    stringList.remove(stringList[0])  # remove # params (will be added later)
    stringList = set(stringList)  # remove duplicates  # noqa: N806
    stringList = [i for i in stringList if i]  # remove empty  # noqa: N806
    stringList = [str(len(stringList))] + stringList  # noqa: N806, RUF005
    with open('params.in', 'w') as f:  # noqa: FURB103, PLW1514, PTH123
        f.write('\n'.join(stringList))

    f.close()

    #
    # get sur model info
    #

    surFileName = None  # noqa: N806
    for model in root_AIM['Simulation']['Models']:
        if model['modelName'] == modelName:
            surFileName = model['fileName']  # noqa: N806

    if surFileName is None:
        print(f'surrogate model {modelName} is not found')  # noqa: T201
        exit(-1)  # noqa: PLR1722

    #
    # find surrogate model prediction app
    #

    s = [
        os.path.dirname(__file__),  # noqa: PTH120
        '..',
        '..',
        'Workflow',
        'WorkflowApplications.json',
    ]
    workflowAppJsonPath = os.path.join(*s)  # noqa: PTH118, N806
    with open(workflowAppJsonPath, encoding='utf-8') as f:  # noqa: PTH123
        workflowAppDict = json.load(f)  # noqa: N806
        appList = workflowAppDict['SimulationApplications']['Applications']  # noqa: N806
        simAppName = 'SurrogateSimulation'  # noqa: N806
        myApp = next(item for item in appList if item['Name'] == simAppName)  # noqa: N806
        s = [
            os.path.dirname(__file__),  # noqa: PTH120
            '..',
            '..',
            '..',
            os.path.dirname(myApp['ExecutablePath']),  # noqa: PTH120
        ]
        mySurrogatePath = os.path.join(*s)  # noqa: PTH118, N806
        mySurrogateName = os.path.basename(myApp['ExecutablePath'])  # noqa: PTH119, N806

    #
    # import surrogate functions
    #

    root_AIM['Applications']['Modeling']['ApplicationData']['MS_Path'] = ''
    root_AIM['Applications']['Modeling']['ApplicationData']['postprocessScript'] = ''
    root_AIM['Applications']['Modeling']['ApplicationData']['mainScript'] = (
        r'..\\..\\..\\..\\input_data\\' + surFileName
    )

    currentDir = os.getcwd()  # noqa: PTH109, N806
    newAimName = os.path.join(currentDir, os.path.basename(aimName))  # noqa: PTH118, PTH119, N806
    with open(newAimName, 'w', encoding='utf-8') as f:  # noqa: FURB103, PTH123
        json_object = json.dumps(root_AIM)
        f.write(json_object)

    sys.path.insert(0, mySurrogatePath)
    sur_module = importlib.__import__(
        mySurrogateName[:-3],
        globals(),
        locals(),
        ['run_surrogateGP', 'write_EDP'],
        0,
    )

    #
    # run prediction
    #

    sur_module.run_surrogateGP(newAimName, edpName)

    #
    # write EDP file
    #

    sur_module.write_EDP(newAimName, edpName)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameAIM')
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameSAM')
    parser.add_argument('--filenameEDP')
    parser.add_argument('--filenameSIM')
    # parser.add_argument('--defaultModule', default=None)
    # parser.add_argument('--fileName', default=None)
    # parser.add_argument('--filePath', default=None)
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    sys.exit(
        main(
            args.filenameAIM,
            args.filenameSAM,
            args.filenameEVENT,
            args.filenameEDP,
            args.filenameSIM,
            args.getRV,
        )
    )
