"""authors: Dr. Frank McKenna*, Aakash Bangalore Satish*, Mukesh Kumar Ramancha, Maitreya Manoj Kurumbhati,
and Prof. J.P. Conte
affiliation: SimCenter*; University of California, San Diego

"""  # noqa: INP001, D205, D400

import os
import shutil
import subprocess

import numpy as np


def copytree(src, dst, symlinks=False, ignore=None):  # noqa: FBT002, D103
    if not os.path.exists(dst):  # noqa: PTH110
        os.makedirs(dst)  # noqa: PTH103
    for item in os.listdir(src):
        s = os.path.join(src, item)  # noqa: PTH118
        d = os.path.join(dst, item)  # noqa: PTH118
        if os.path.isdir(s):  # noqa: PTH112
            copytree(s, d, symlinks, ignore)
        else:
            try:
                if (
                    not os.path.exists(d)  # noqa: PTH110
                    or os.stat(s).st_mtime - os.stat(d).st_mtime > 1  # noqa: PTH116
                ):
                    shutil.copy2(s, d)
            except Exception as ex:  # noqa: BLE001
                msg = f'Could not copy {s}. The following error occurred: \n{ex}'
                return msg  # noqa: RET504
    return '0'


def runFEM(  # noqa: N802
    particleNumber,  # noqa: N803
    parameterSampleValues,  # noqa: N803
    variables,
    workdirMain,  # noqa: N803
    log_likelihood_function,
    calibrationData,  # noqa: ARG001, N803
    numExperiments,  # noqa: ARG001, N803
    covarianceMatrixList,  # noqa: ARG001, N803
    edpNamesList,  # noqa: N803
    edpLengthsList,  # noqa: N803
    scaleFactors,  # noqa: ARG001, N803
    shiftFactors,  # noqa: ARG001, N803
    workflowDriver,  # noqa: N803
):
    """This function runs FE model (model.tcl) for each parameter value (par)
    model.tcl should take parameter input
    model.tcl should output 'output$PN.txt' -> column vector of size 'Ny'
    """  # noqa: D205, D400, D401, D404
    workdirName = 'workdir.' + str(particleNumber + 1)  # noqa: N806
    analysisPath = os.path.join(workdirMain, workdirName)  # noqa: PTH118, N806

    if os.path.isdir(analysisPath):  # noqa: PTH112
        os.chmod(os.path.join(analysisPath, workflowDriver), 0o777)  # noqa: S103, PTH101, PTH118
        shutil.rmtree(analysisPath)

    os.mkdir(analysisPath)  # noqa: PTH102

    # copy templatefiles
    templateDir = os.path.join(workdirMain, 'templatedir')  # noqa: PTH118, N806
    copytree(templateDir, analysisPath)

    # change to analysis directory
    os.chdir(analysisPath)

    # write input file and covariance multiplier values list
    covarianceMultiplierList = []  # noqa: N806
    parameterNames = variables['names']  # noqa: N806
    with open('params.in', 'w') as f:  # noqa: PTH123
        f.write(f'{len(parameterSampleValues) - len(edpNamesList)}\n')
        for i in range(len(parameterSampleValues)):
            name = str(parameterNames[i])
            value = str(parameterSampleValues[i])
            if name.split('.')[-1] != 'CovMultiplier':
                f.write(f'{name} {value}\n')
            else:
                covarianceMultiplierList.append(parameterSampleValues[i])

    # subprocess.run(workflowDriver, stderr=subprocess.PIPE, shell=True)

    returnCode = subprocess.call(  # noqa: S602, N806, F841
        os.path.join(analysisPath, workflowDriver),  # noqa: PTH118
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )  # subprocess.check_call(workflow_run_command, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

    # Read in the model prediction
    if os.path.exists('results.out'):  # noqa: PTH110
        with open('results.out') as f:  # noqa: PTH123
            prediction = np.atleast_2d(np.genfromtxt(f)).reshape((1, -1))
        preds = prediction.copy()
        os.chdir('../')
        ll = log_likelihood_function(prediction, covarianceMultiplierList)
    else:
        os.chdir('../')
        preds = np.atleast_2d([-np.inf] * sum(edpLengthsList)).reshape((1, -1))
        ll = -np.inf

    return (ll, preds)
