"""authors: Mukesh Kumar Ramancha, Maitreya Manoj Kurumbhati, Prof. J.P. Conte, and Aakash Bangalore Satish*
affiliation: University of California, San Diego, *SimCenter, University of California, Berkeley

"""  # noqa: INP001, D205, D400

import itertools
import json
import os
import sys
from importlib import import_module


class DataProcessingError(Exception):
    """Raised when errors found when processing user-supplied calibration and covariance data.

    Attributes
    ----------
        message -- explanation of the error

    """

    def __init__(self, message):
        self.message = message


def parseDataFunction(dakotaJsonFile, logFile):  # noqa: C901, N802, N803, D103, PLR0915
    # Read in the json object
    logFile.write('\n\tReading the json file')
    with open(dakotaJsonFile) as f:  # noqa: PTH123
        jsonInputs = json.load(f)  # noqa: N806
    logFile.write(' ... Done')

    # Read in the data of the objects within the json file
    logFile.write('\n\tParsing the inputs read in from json file')
    applications = jsonInputs['Applications']
    edpInputs = jsonInputs['EDP']  # noqa: N806
    uqInputs = jsonInputs['UQ']  # noqa: N806
    # femInputs = jsonInputs['FEM']
    rvInputs = jsonInputs['randomVariables']  # noqa: N806
    # localAppDirInputs = jsonInputs['localAppDir']
    # pythonInputs = jsonInputs['python']
    # remoteAppDirInputs = jsonInputs['remoteAppDir']
    # uqResultsInputs = jsonInputs['uqResults']
    # if uqResultsInputs:
    #    resultType = uqResultsInputs['resultType']
    #    if resultType == 'UCSD_Results':
    #        spreadsheet = uqResultsInputs['spreadsheet']
    #        dataValues = spreadsheet['data']
    #        headings = spreadsheet['headings']
    #        numCol = spreadsheet['numCol']
    #        numRow = spreadsheet['numRow']
    #        summary = uqResultsInputs['summary']
    # workingDir = jsonInputs['workingDir']

    # Processing UQ inputs
    logFile.write('\n\t\tProcessing UQ inputs')
    seedValue = uqInputs['seed']  # noqa: N806
    nSamples = uqInputs['numParticles']  # noqa: N806
    # maxRunTime = uqInputs["maxRunTime"]
    # if 'maxRunTime' in uqInputs.keys():
    #     maxRunTime = uqInputs['maxRunTime']
    # else:
    #     maxRunTime = float('inf')
    # logLikelihoodFile = uqInputs['logLikelihoodFile']
    calDataFile = uqInputs['calDataFile']  # noqa: N806

    # parallelizeMCMC = True
    # if 'parallelExecution' in uqInputs:
    #     parallelizeMCMC = uqInputs['parallelExecution']

    # Processing EDP inputs
    logFile.write('\n\n\t\tProcessing EDP inputs')
    edpNamesList = []  # noqa: N806
    edpLengthsList = []  # noqa: N806
    # Get list of EDPs and their lengths
    for edp in edpInputs:
        edpNamesList.append(edp['name'])
        edpLengthsList.append(edp['length'])

    logFile.write('\n\t\t\tThe EDPs defined are:')
    printString = '\n\t\t\t\t'  # noqa: N806
    for i in range(len(edpInputs)):
        printString += (  # noqa: N806
            f"Name: '{edpNamesList[i]}', Length: {edpLengthsList[i]}\n\t\t\t\t"
        )
    logFile.write(printString)
    # logFile.write("\tExpected length of each line in data file: {}".format(lineLength))

    # Processing model inputs
    logFile.write('\n\n\t\tProcessing application inputs')
    # Processing number of models
    # Check if this is a multi-model analysis
    # runMultiModel = False
    modelsDict = {}  # noqa: N806
    modelIndicesList = []  # noqa: N806
    modelRVNamesList = []  # noqa: N806
    applications = jsonInputs['Applications']
    for app, appInputs in applications.items():  # noqa: N806
        logFile.write(f'\n\t\t\tApp: {app}')
        if app.lower() != 'events':
            appl = appInputs['Application'].lower()
        else:
            appl = appInputs[0]['Application'].lower()
        if appl == 'multimodel':
            # runMultiModel = True
            logFile.write(
                f'\n\t\t\t\tFound a multimodel application - {app}: {appInputs["Application"]}'
            )
            modelRVName = jsonInputs[app]['modelToRun'][3:]  # noqa: N806
            appModels = jsonInputs[app]['models']  # noqa: N806
            nM = len(appModels)  # noqa: N806
            logFile.write(f'\n\t\t\t\t\tThere are {nM} {app} models')
            modelData = {}  # noqa: N806
            modelData['nModels'] = nM
            modelData['values'] = [i + 1 for i in range(nM)]
            modelData['weights'] = [model['belief'] for model in appModels]
            modelData['name'] = modelRVName
            modelsDict[app] = modelData
            modelIndicesList.append(modelData['values'])
            modelRVNamesList.append(modelRVName)
        else:
            logFile.write('\n\t\t\t\tNot a multimodel application')
    nModels = 1  # noqa: N806
    for _, data in modelsDict.items():  # noqa: PERF102
        nModels = nModels * data['nModels']  # noqa: N806
    cartesianProductOfModelIndices = list(itertools.product(*modelIndicesList))  # noqa: N806
    # logFile.write("\n\t\t\tNO LONGER Getting the number of models")
    # inputFileList = []
    # nModels = femInputs['numInputs']
    # nModels = 1
    # if nModels > 1:
    #    fileInfo = femInputs['fileInfo']
    #    for m in range(nModels):
    #        inputFileList.append(fileInfo[m]['inputFile'])
    # else:
    #    inputFileList.append(femInputs['inputFile'])
    # logFile.write('\n\t\t\t\tThe number of models is: {}'.format(nModels))
    # writeFEMOutputs = True

    # Variables
    variablesList = []  # noqa: N806
    for _ in range(nModels):
        variablesList.append(  # noqa: PERF401
            {
                'names': [],
                'distributions': [],
                'Par1': [],
                'Par2': [],
                'Par3': [],
                'Par4': [],
            }
        )

    logFile.write('\n\n\t\t\tLooping over the models')
    for ind in range(nModels):
        logFile.write(f'\n\t\t\t\tModel number: {ind}')
        # Processing RV inputs
        logFile.write(f'\n\t\t\t\t\tCreating priors for model number {ind}')
        logFile.write('\n\t\t\t\t\t\tProcessing RV inputs')
        for i, rv in enumerate(rvInputs):
            variablesList[ind]['names'].append(rv['name'])
            variablesList[ind]['distributions'].append(rv['distribution'])
            paramString = ''  # noqa: N806
            if rv['distribution'] == 'Uniform':
                variablesList[ind]['Par1'].append(rv['lowerbound'])
                variablesList[ind]['Par2'].append(rv['upperbound'])
                variablesList[ind]['Par3'].append(None)
                variablesList[ind]['Par4'].append(None)
                paramString = 'params: {}, {}'.format(  # noqa: N806
                    rv['lowerbound'], rv['upperbound']
                )
            elif rv['distribution'] == 'Normal':
                variablesList[ind]['Par1'].append(rv['mean'])
                variablesList[ind]['Par2'].append(rv['stdDev'])
                variablesList[ind]['Par3'].append(None)
                variablesList[ind]['Par4'].append(None)
                paramString = 'params: {}, {}'.format(rv['mean'], rv['stdDev'])  # noqa: N806
            elif rv['distribution'] == 'Half-Normal':
                variablesList[ind]['Par1'].append(rv['Standard Deviation'])
                variablesList[ind]['Par2'].append(rv['Upper Bound'])
                variablesList[ind]['Par3'].append(None)
                variablesList[ind]['Par4'].append(None)
                paramString = 'params: {}, {}'.format(  # noqa: N806
                    rv['Standard Deviation'], rv['Upper Bound']
                )
            elif rv['distribution'] == 'Truncated-Normal':
                variablesList[ind]['Par1'].append(rv['Mean'])
                variablesList[ind]['Par2'].append(rv['Standard Deviation'])
                variablesList[ind]['Par3'].append(rv['a'])
                variablesList[ind]['Par4'].append(rv['b'])
                paramString = 'params: {}, {}, {}, {}'.format(  # noqa: N806
                    rv['Mean'], rv['Standard Deviation'], rv['a'], rv['b']
                )
            elif rv['distribution'] == 'Beta':
                variablesList[ind]['Par1'].append(rv['alphas'])
                variablesList[ind]['Par2'].append(rv['betas'])
                variablesList[ind]['Par3'].append(rv['lowerbound'])
                variablesList[ind]['Par4'].append(rv['upperbound'])
                paramString = 'params: {}, {}, {}, {}'.format(  # noqa: N806
                    rv['alphas'], rv['betas'], rv['lowerbound'], rv['upperbound']
                )
            elif rv['distribution'] == 'Lognormal':
                # meanValue = rv["mean"]
                # stdevValue = rv["stdDev"]
                # mu = np.log(
                #     pow(meanValue, 2) / np.sqrt(pow(stdevValue, 2) + pow(meanValue, 2))
                # )
                # sig = np.sqrt(np.log(pow(stdevValue / meanValue, 2) + 1))
                mu = rv['lambda']
                sigma = rv['zeta']
                variablesList[ind]['Par1'].append(mu)
                variablesList[ind]['Par2'].append(sigma)
                variablesList[ind]['Par3'].append(None)
                variablesList[ind]['Par4'].append(None)
                paramString = f'params: {mu}, {sigma}'  # noqa: N806
            elif rv['distribution'] == 'Gumbel':
                variablesList[ind]['Par1'].append(rv['alphaparam'])
                variablesList[ind]['Par2'].append(rv['betaparam'])
                variablesList[ind]['Par3'].append(None)
                variablesList[ind]['Par4'].append(None)
                paramString = 'params: {}, {}'.format(  # noqa: N806
                    rv['alphaparam'], rv['betaparam']
                )
            elif rv['distribution'] == 'Weibull':
                variablesList[ind]['Par1'].append(rv['shapeparam'])
                variablesList[ind]['Par2'].append(rv['scaleparam'])
                variablesList[ind]['Par3'].append(None)
                variablesList[ind]['Par4'].append(None)
                paramString = 'params: {}, {}'.format(  # noqa: N806
                    rv['shapeparam'], rv['scaleparam']
                )
            elif rv['distribution'] == 'Exponential':
                variablesList[ind]['Par1'].append(rv['lambda'])
                variablesList[ind]['Par2'].append(None)
                variablesList[ind]['Par3'].append(None)
                variablesList[ind]['Par4'].append(None)
                paramString = 'params: {}'.format(rv['lambda'])  # noqa: N806
            elif rv['distribution'] == 'Gamma':
                variablesList[ind]['Par1'].append(rv['k'])
                variablesList[ind]['Par2'].append(rv['lambda'])
                variablesList[ind]['Par3'].append(None)
                variablesList[ind]['Par4'].append(None)
                paramString = 'params: {}, {}'.format(rv['k'], rv['lambda'])  # noqa: N806
            elif rv['distribution'] == 'Chisquare':
                variablesList[ind]['Par1'].append(rv['k'])
                variablesList[ind]['Par2'].append(None)
                variablesList[ind]['Par3'].append(None)
                variablesList[ind]['Par4'].append(None)
                paramString = 'params: {}'.format(rv['k'])  # noqa: N806
            elif rv['distribution'] == 'Truncated exponential':
                variablesList[ind]['Par1'].append(rv['lambda'])
                variablesList[ind]['Par2'].append(rv['a'])
                variablesList[ind]['Par3'].append(rv['b'])
                variablesList[ind]['Par4'].append(None)
                paramString = 'params: {}, {}, {}'.format(  # noqa: N806
                    rv['lambda'], rv['a'], rv['b']
                )
            elif rv['distribution'] == 'Discrete':
                if 'multimodel' in rv['name'].lower():
                    try:
                        index = modelRVNamesList.index(rv['name'])
                        variablesList[ind]['Par1'].append(
                            cartesianProductOfModelIndices[ind][index]
                        )
                        variablesList[ind]['Par2'].append(None)
                        variablesList[ind]['Par3'].append(None)
                        variablesList[ind]['Par4'].append(None)
                        paramString = (  # noqa: N806
                            f'value: {cartesianProductOfModelIndices[ind][index]}'
                        )
                    except ValueError:
                        logFile.write(
                            f"{rv['name']} not found in list of model RV names"
                        )

                else:
                    variablesList[ind]['Par1'].append(rv['Values'])
                    variablesList[ind]['Par2'].append(rv['Weights'])
                    variablesList[ind]['Par3'].append(None)
                    variablesList[ind]['Par4'].append(None)
                    paramString = 'values: {}, weights: {}'.format(  # noqa: N806
                        rv['Values'], rv['Weights']
                    )

            logFile.write(
                '\n\t\t\t\t\t\t\tRV number: {}, name: {}, dist: {}, {}'.format(
                    i, rv['name'], rv['distribution'], paramString
                )
            )
        # if runMultiModel:
        #     variablesList[ind][]

        # Adding one prior distribution per EDP for the error covariance multiplier term
        logFile.write(
            '\n\t\t\t\t\t\tAdding one prior distribution per EDP for the error covariance multiplier '
            'term'
        )
        # logFile.write("\n\t\t\tThe prior on the error covariance multipliers is an inverse gamma distribution \n"
        #       "\t\twith parameters a and b set to 100. This corresponds to a variable whose mean \n"
        #       "\t\tand mode are approximately 1.0 and whose standard deviation is approximately 0.1.")
        # a = 100
        # b = 100
        # The prior with parameters = 100 is too narrow. Using these values instead:
        a = 3
        b = 2
        for i, edp in enumerate(edpInputs):
            name = edp['name'] + '.CovMultiplier'
            variablesList[ind]['names'].append(name)
            variablesList[ind]['distributions'].append('InvGamma')
            variablesList[ind]['Par1'].append(a)
            variablesList[ind]['Par2'].append(b)
            variablesList[ind]['Par3'].append(None)
            variablesList[ind]['Par4'].append(None)
            paramString = f'params: {a}, {b}'  # noqa: N806
            logFile.write(
                '\n\t\t\t\t\t\t\tEDP number: {}, name: {}, dist: {}, {}'.format(
                    i, name, 'InvGamma', paramString
                )
            )

    logFile.write('\n\n\tCompleted parsing the inputs')
    logFile.write('\n\n==========================')
    logFile.flush()
    os.fsync(logFile.fileno())
    return (
        nSamples,
        seedValue,
        calDataFile,
        variablesList,
        edpNamesList,
        edpLengthsList,
        modelsDict,
        nModels,
    )
