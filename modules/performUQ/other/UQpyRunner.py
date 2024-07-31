# written: Michael Gardner @ UNR
# updated Aakash Bangalore Satish, June 11 2024

import os
from uqRunner import UqRunnerFactory
from uqRunner import UqRunner

# THIS IS FOR WHEN MESSING AROUND WITH UQpy SOURCE
# import sys
# sys.path.append(os.path.abspath("/home/michael/UQpy/src"))

from UQpy.sampling.MonteCarloSampling import MonteCarloSampling as MCS
from UQpy.run_model.RunModel import RunModel
from UQpy.run_model.model_execution.ThirdPartyModel import ThirdPartyModel
from UQpy.distributions.collection.Uniform import Uniform
from createTemplate import createTemplate
import time
import csv
import json
import shutil


class UQpyRunner(UqRunner):
    def runUQ(
        self,
        uqData,
        simulationData,
        randomVarsData,
        demandParams,
        workingDir,
        runType,
        localAppDir,
        remoteAppDir,
    ):
        """
        This function configures and runs a UQ simulation using UQpy based on the
        input UQ configuration, simulation configuration, random variables,
        and requested demand parameters

        Input:
        uqData:         JsonObject that UQ options as input into the quoFEM GUI
        simulationData: JsonObject that contains information on the analysis package to run and its
                    configuration as input in the quoFEM GUI
        randomVarsData: JsonObject that specifies the input random variables, their distributions,
                    and associated parameters as input in the quoFEM GUI
        demandParams:   JsonObject that specifies the demand parameters as input in the quoFEM GUI
        workingDir:     Directory in which to run simulations and store temporary results
        runType:        Specifies whether computations are being run locally or on an HPC cluster
        localAppDir:    Directory containing apps for local run
        remoteAppDir:   Directory containing apps for remote run
        """

        # There is still plenty of configuration that can and should be added here. This currently does MCS sampling with Uniform
        # distributions only, though this is easily expanded

        # Copy required python files to template directory
        shutil.copyfile(
            os.path.join(
                localAppDir, 'applications/performUQ/other/runWorkflowDriver.py'
            ),
            os.path.join(workingDir, 'runWorkflowDriver.py'),
        )
        shutil.copyfile(
            os.path.join(
                localAppDir, 'applications/performUQ/other/createTemplate.py'
            ),
            os.path.join(workingDir, 'createTemplate.py'),
        )
        shutil.copyfile(
            os.path.join(
                localAppDir, 'applications/performUQ/other/processUQpyOutput.py'
            ),
            os.path.join(workingDir, 'processUQpyOutput.py'),
        )

        # Parse configuration for UQ
        distributionNames = []
        distributionParams = []
        variableNames = []
        distributionObjects = []
        samples = []
        samplingMethod = ''
        numberOfSamples = 0
        modelScript = 'runWorkflowDriver.py'
        inputTemplate = 'params.template'
        # outputObjectName = 'OutputProcessor'
        outputObjectName = 'output_function'
        outputScript = 'processUQpyOutput.py'
        numberOfTasks = 1
        numberOfNodes = 1
        coresPerTask = 1
        clusterRun = False
        resumeRun = False
        seed = 1

        # If computations are being executed on HPC, enable UQpy to start computations using srun
        if runType == 'runningRemote':
            clusterRun = True

        for val in randomVarsData:
            if val['distribution'] == 'Uniform':
                distributionNames.append('Uniform')
                variableNames.append(val['name'])
                distributionParams.append([val['lowerbound'], val['upperbound']])
            else:
                raise IOError(
                    "ERROR: You'll need to update UQpyRunner.py to run your"
                    + ' specified RV distribution!'
                )

        for val in uqData['Parameters']:
            if val['name'] == 'Sampling Method':
                samplingMethod = val['value']

            if val['name'] == 'Number of Samples':
                numberOfSamples = int(val['value'])

            if val['name'] == 'Number of Concurrent Tasks':
                numberOfTasks = val['value']

            if val['name'] == 'Number of Nodes':
                numberOfNodes = val['value']

            if val['name'] == 'Cores per Task':
                coresPerTask = val['value']

            if val['name'] == 'Seed':
                seed = int(val['value'])

        # Create distribution objects
        for index, val in enumerate(distributionNames, 0):
            distributionObjects.append(
                Uniform(
                    distributionParams[index][0],
                    distributionParams[index][1] - distributionParams[index][0],
                )
            )

        createTemplate(variableNames, inputTemplate)

        # Generate samples
        if samplingMethod == 'MCS':
            samples = MCS(
                distributionObjects, nsamples=numberOfSamples, random_state=seed
            )
        else:
            raise IOError(
                "ERROR: You'll need to update UQpyRunner.py to run your specified"
                + ' sampling method!'
            )

        # Change workdir to the template directory
        os.chdir(workingDir)

        # Run model based on input config
        startTime = time.time()
        # model = RunModel(samples=samples.samples, model_script=modelScript,
        #                  input_template=inputTemplate, var_names=variableNames,
        #                  output_script=outputScript, output_object_name=outputObjectName,
        #                  verbose=True, ntasks=numberOfTasks,
        #                  nodes=numberOfNodes, cores_per_task=coresPerTask,
        #                  cluster=clusterRun, resume=resumeRun)
        model = ThirdPartyModel(
            model_script=modelScript,
            input_template=inputTemplate,
            var_names=variableNames,
            output_script=outputScript,
            output_object_name=outputObjectName,
        )
        m = RunModel(ntasks=numberOfTasks, model=model)
        m.run(samples.samples)

        runTime = time.time() - startTime
        print('\nTotal time for all experiments: ', runTime)

        with open(os.path.join(workingDir, '..', 'tabularResults.out'), 'w') as f:
            f.write('%eval_id\t interface\t')

            for val in variableNames:
                f.write('%s\t' % val)

            for val in demandParams:
                f.write('%s\t' % val['name'])

            f.write('\n')

            for i in range(numberOfSamples):
                string = f'{i+1} \tcustom\t'
                for sample in samples.samples[i]:
                    string += f'{sample}\t'
                for qoi in m.qoi_list[i]:
                    for val in qoi:
                        string += f'{val}\t'
                string += '\n'
                f.write(string)

    # Factory for creating UQpy runner
    class Factory:
        def create(self):
            return UQpyRunner()
