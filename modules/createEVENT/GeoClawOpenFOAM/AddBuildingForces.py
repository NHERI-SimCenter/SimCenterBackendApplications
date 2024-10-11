#!/usr/bin/env python  # noqa: D100
import argparse
import json
import os


def validateCaseDirectoryStructure(caseDir):  # noqa: N802, N803
    """This method validates that the provided case directory is valid and contains the 0, constant and system directory
    It also checks that system directory contains the controlDict
    """  # noqa: D205, D400, D401, D404
    if not os.path.isdir(caseDir):  # noqa: PTH112
        return False  # noqa: DOC201, RUF100

    caseDirList = os.listdir(caseDir)  # noqa: N806
    necessaryDirs = ['0', 'constant', 'system']  # noqa: N806
    if any(aDir not in caseDirList for aDir in necessaryDirs):
        return False

    controlDictPath = os.path.join(caseDir, 'system/controlDict')  # noqa: PTH118, N806
    if not os.path.exists(controlDictPath):  # noqa: SIM103, PTH110
        return False

    return True


def findFunctionsDictionary(controlDictLines):  # noqa: N802, N803
    """This method will find functions dictionary in the controlDict"""  # noqa: D400, D401, D404
    for line in controlDictLines:
        if line.startswith('functions'):
            return (True, controlDictLines.index(line) + 2)  # noqa: DOC201, RUF100

    return [False, len(controlDictLines)]


def writeForceDictionary(controlDictLines, lineIndex, floorsCount, patches):  # noqa: N802, N803
    """This method will write the force dictionary"""  # noqa: D400, D401, D404
    for line in ['\t\n', '\tbuildingsForces\n', '\t{\n', '\t}\n', '\n']:
        controlDictLines.insert(lineIndex, line)
        lineIndex += 1  # noqa: N806

    forceDictionary = {  # noqa: N806
        'type': 'forces',
        'libs': '("libforces.so")',
        'writeControl': 'timeStep',
        'writeInterval': 1,
        'patches': f'({patches})',
        'rho': 'rhoInf',
        'log': 'true',
        'rhoInf': 1,
        'CofR': '(0 0 0)',
    }

    lineIndex -= 2  # noqa: N806
    for key, value in forceDictionary.items():
        controlDictLines.insert(lineIndex, '\t\t' + key + '\t' + str(value) + ';\n')
        lineIndex += 1  # noqa: N806

    for line in ['\n', '\t\tbinData\n', '\t\t{\n', '\t\t}\n', '\n']:
        controlDictLines.insert(lineIndex, line)
        lineIndex += 1  # noqa: N806

    lineIndex -= 2  # noqa: N806
    binDictionary = {  # noqa: N806
        'nBin': str(floorsCount),
        'direction': '(0 0 1)',
        'cumulative': 'no',
    }

    for key, value in binDictionary.items():
        controlDictLines.insert(
            lineIndex, '\t\t\t' + key + '\t' + str(value) + ';\n'
        )
        lineIndex += 1  # noqa: N806


def AddBuildingsForces(floorsCount, patches):  # noqa: N802, N803
    """First, we need to validate the case directory structure"""  # noqa: D400
    # if not validateCaseDirectoryStructure(caseDir):
    #     print("Invalid OpenFOAM Case Directory!")
    #     sys.exit(-1)

    # controlDictPath = os.path.join(caseDir, "system/controlDict")
    controlDictPath = 'system/controlDict'  # noqa: N806
    with open(controlDictPath) as controlDict:  # noqa: PTH123, N806
        controlDictLines = controlDict.readlines()  # noqa: N806

    [isFound, lineIndex] = findFunctionsDictionary(controlDictLines)  # noqa: N806

    # If we cannot find the function dictionary, we will create one
    if not isFound:
        for line in ['\n', 'functions\n', '{\n', '}\n']:
            controlDictLines.insert(lineIndex, line)
            lineIndex += 1  # noqa: N806

    # Now we can add the building forces
    writeForceDictionary(controlDictLines, lineIndex, floorsCount, patches)

    # Writing updated controlDict
    with open(controlDictPath, 'w') as controlDict:  # noqa: PTH123, N806
        controlDict.writelines(controlDictLines)


def GetFloorsCount(BIMFilePath):  # noqa: N802, N803, D103
    with open(BIMFilePath) as BIMFile:  # noqa: PTH123, N806
        bim = json.load(BIMFile)

    return int(bim['GeneralInformation']['stories'])


if __name__ == '__main__':
    # CLI parser
    parser = argparse.ArgumentParser(
        description='Add forces postprocessing to OpenFOAM controlDict'
    )
    # parser.add_argument('-c', '--case', help="OpenFOAM case directory", required=True)
    parser.add_argument(
        '-f', '--floors', help='Number of Floors', type=int, required=False
    )
    parser.add_argument('-b', '--bim', help='path to BIM file', required=False)
    parser.add_argument(
        '-p',
        '--patches',
        help='Patches used for extracting forces on building',
        required=False,
    )

    # Parsing arguments
    arguments, unknowns = parser.parse_known_args()
    floors = arguments.floors
    if not floors:
        if arguments.bim:
            floors = GetFloorsCount(arguments.bim)
        else:
            floors = 1

    patches = arguments.patches
    if not patches:
        patches = 'Building'

    # Add building forces to post-processing
    # AddBuildingsForces(arguments.case, floors, patches)
    AddBuildingsForces(floors, patches)
