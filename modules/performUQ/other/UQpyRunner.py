# written: Michael Gardner @ UNR  # noqa: INP001, D100
# updated Aakash Bangalore Satish, June 11 2024

import os
import shutil
import time

from createTemplate import createTemplate
from UQpy.distributions.collection.Uniform import Uniform
from UQpy.run_model.model_execution.ThirdPartyModel import ThirdPartyModel
from UQpy.run_model.RunModel import RunModel

# THIS IS FOR WHEN MESSING AROUND WITH UQpy SOURCE
# import sys  # noqa: ERA001
# sys.path.append(os.path.abspath("/home/michael/UQpy/src"))  # noqa: ERA001
from UQpy.sampling.MonteCarloSampling import MonteCarloSampling as MCS  # noqa: N817
from uqRunner import UqRunner


class UQpyRunner(UqRunner):  # noqa: D101
    def runUQ(  # noqa: ANN201, C901, N802, PLR0912, PLR0913, PLR0915
        self,
        uqData,  # noqa: ANN001, N803
        simulationData,  # noqa: ANN001, ARG002, N803
        randomVarsData,  # noqa: ANN001, N803
        demandParams,  # noqa: ANN001, N803
        workingDir,  # noqa: ANN001, N803
        runType,  # noqa: ANN001, N803
        localAppDir,  # noqa: ANN001, N803
        remoteAppDir,  # noqa: ANN001, ARG002, N803
    ):
        """This function configures and runs a UQ simulation using UQpy based on the
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
        """  # noqa: E501, D205, D400, D401, D404, D415
        # There is still plenty of configuration that can and should be added here. This currently does MCS sampling with Uniform  # noqa: E501
        # distributions only, though this is easily expanded

        # Copy required python files to template directory
        shutil.copyfile(
            os.path.join(  # noqa: PTH118
                localAppDir, 'applications/performUQ/other/runWorkflowDriver.py'
            ),
            os.path.join(workingDir, 'runWorkflowDriver.py'),  # noqa: PTH118
        )
        shutil.copyfile(
            os.path.join(  # noqa: PTH118
                localAppDir, 'applications/performUQ/other/createTemplate.py'
            ),
            os.path.join(workingDir, 'createTemplate.py'),  # noqa: PTH118
        )
        shutil.copyfile(
            os.path.join(  # noqa: PTH118
                localAppDir, 'applications/performUQ/other/processUQpyOutput.py'
            ),
            os.path.join(workingDir, 'processUQpyOutput.py'),  # noqa: PTH118
        )

        # Parse configuration for UQ
        distributionNames = []  # noqa: N806
        distributionParams = []  # noqa: N806
        variableNames = []  # noqa: N806
        distributionObjects = []  # noqa: N806
        samples = []
        samplingMethod = ''  # noqa: N806
        numberOfSamples = 0  # noqa: N806
        modelScript = 'runWorkflowDriver.py'  # noqa: N806
        inputTemplate = 'params.template'  # noqa: N806
        # outputObjectName = 'OutputProcessor'  # noqa: ERA001
        outputObjectName = 'output_function'  # noqa: N806
        outputScript = 'processUQpyOutput.py'  # noqa: N806
        numberOfTasks = 1  # noqa: N806
        numberOfNodes = 1  # noqa: N806
        coresPerTask = 1  # noqa: N806
        clusterRun = False  # noqa: N806
        resumeRun = False  # noqa: N806, F841
        seed = 1

        # If computations are being executed on HPC, enable UQpy to start computations using srun  # noqa: E501
        if runType == 'runningRemote':
            clusterRun = True  # noqa: N806, F841

        for val in randomVarsData:
            if val['distribution'] == 'Uniform':
                distributionNames.append('Uniform')
                variableNames.append(val['name'])
                distributionParams.append([val['lowerbound'], val['upperbound']])
            else:
                raise OSError(
                    "ERROR: You'll need to update UQpyRunner.py to run your"  # noqa: ISC003
                    + ' specified RV distribution!'
                )

        for val in uqData['Parameters']:
            if val['name'] == 'Sampling Method':
                samplingMethod = val['value']  # noqa: N806

            if val['name'] == 'Number of Samples':
                numberOfSamples = int(val['value'])  # noqa: N806

            if val['name'] == 'Number of Concurrent Tasks':
                numberOfTasks = val['value']  # noqa: N806

            if val['name'] == 'Number of Nodes':
                numberOfNodes = val['value']  # noqa: N806, F841

            if val['name'] == 'Cores per Task':
                coresPerTask = val['value']  # noqa: N806, F841

            if val['name'] == 'Seed':
                seed = int(val['value'])

        # Create distribution objects
        for index, val in enumerate(distributionNames, 0):  # noqa: B007
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
            raise OSError(
                "ERROR: You'll need to update UQpyRunner.py to run your specified"  # noqa: ISC003
                + ' sampling method!'
            )

        # Change workdir to the template directory
        os.chdir(workingDir)

        # Run model based on input config
        startTime = time.time()  # noqa: N806
        # model = RunModel(samples=samples.samples, model_script=modelScript,
        #                  input_template=inputTemplate, var_names=variableNames,  # noqa: ERA001
        #                  output_script=outputScript, output_object_name=outputObjectName,  # noqa: ERA001, E501
        #                  verbose=True, ntasks=numberOfTasks,
        #                  nodes=numberOfNodes, cores_per_task=coresPerTask,  # noqa: ERA001
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

        runTime = time.time() - startTime  # noqa: N806
        print('\nTotal time for all experiments: ', runTime)  # noqa: T201

        with open(os.path.join(workingDir, '..', 'tabularResults.out'), 'w') as f:  # noqa: PTH118, PTH123
            f.write('%eval_id\t interface\t')

            for val in variableNames:
                f.write('%s\t' % val)  # noqa: UP031

            for val in demandParams:
                f.write('%s\t' % val['name'])  # noqa: UP031

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
    class Factory:  # noqa: D106
        def create(self):  # noqa: ANN201, D102
            return UQpyRunner()
