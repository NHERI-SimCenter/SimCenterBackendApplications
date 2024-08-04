# written: Aakash Bangalore Satish @ NHERI SimCenter, UC Berkeley  # noqa: CPY001, D100, INP001

import importlib
import json
import os
import sys
import time

from uqRunner import UqRunner


class HeirBayesRunner(UqRunner):  # noqa: D101
    def __init__(self) -> None:
        super().__init__()
        self.n_samples = 0
        self.n_burn_in = 0
        self.tuning_interval = 0
        self.seed = 0

    def storeUQData(self, uqData):  # noqa: N802, N803, D102
        for val in uqData['Parameters']:
            if val['name'] == 'File To Run':
                self.file_to_run = val['value']
            elif val['name'] == '# Samples':
                self.n_samples = int(val['value'])
            elif val['name'] == '# Burn-in':
                self.n_burn_in = int(val['value'])
            elif val['name'] == 'Tuning Interval':
                self.tuning_interval = int(val['value'])
            elif val['name'] == 'Seed':
                self.seed = int(val['value'])

    def performHeirBayesSampling(self):  # noqa: N802, D102
        self.dir_name = os.path.dirname(self.file_to_run)  # noqa: PTH120
        sys.path.append(self.dir_name)
        module_name = os.path.basename(self.file_to_run)  # noqa: PTH119
        module = importlib.import_module(module_name[:-3])
        self.heir_code = module.HeirBayesSampler()

        self.trace, self.time_taken, self.inf_object, self.num_coupons = (
            self.heir_code.perform_sampling(
                n_samples=self.n_samples,
                n_burn_in=self.n_burn_in,
                tuning_interval=self.tuning_interval,
                seed=self.seed,
            )
        )

    def saveResultsToPklFile(self):  # noqa: N802, D102
        self.saved_pickle_filename = self.heir_code.save_results(
            self.trace, self.time_taken, self.inf_object, prefix='synthetic_data'
        )

    def createHeadingStringsList(self):  # noqa: N802, D102
        self.params = ['fy', 'E', 'b', 'cR1', 'cR2', 'a1', 'a3']
        self.num_params = len(self.params)

        self.heading_list = ['Sample#', 'interface']
        for i in range(self.num_coupons):
            for j in range(self.num_params):
                self.heading_list.append(
                    ''.join(['Coupon_', str(i + 1), '_', self.params[j]])
                )

        for row in range(self.num_params):
            for col in range(row + 1):
                self.heading_list.append(
                    ''.join(['Cov_', str(row + 1), str(col + 1)])
                )

        for par in self.params:
            self.heading_list.append(''.join(['Mean_', par]))  # noqa: FLY002

        for sig in range(self.num_coupons):
            self.heading_list.append(''.join(['ErrorVariance_', str(sig + 1)]))

    def makeHeadingRow(self, separator='\t'):  # noqa: N802, D102
        self.headingRow = separator.join([item for item in self.heading_list])  # noqa: C416

    def makeOneRowString(self, sample_num, sample, separator='\t'):  # noqa: N802, D102
        initial_string = separator.join([str(sample_num), '1'])
        coupon_string = separator.join(
            [
                str(sample[i][j])
                for i in range(self.num_coupons)
                for j in range(self.num_params)
            ]
        )
        cov_string = separator.join(
            [
                str(sample[self.num_coupons][row][col])
                for row in range(self.num_params)
                for col in range(row + 1)
            ]
        )
        mean_string = separator.join(
            [
                str(sample[self.num_coupons + 1][par_num])
                for par_num in range(self.num_params)
            ]
        )
        error_string = separator.join(
            [str(sample[-1][coupon_num]) for coupon_num in range(self.num_coupons)]
        )
        row_string = separator.join(
            [initial_string, coupon_string, cov_string, mean_string, error_string]
        )
        return row_string  # noqa: RET504

    def makeTabularResultsFile(  # noqa: N802, D102
        self,
        save_file_name='tabularResults.out',
        separator='\t',
    ):
        self.createHeadingStringsList()
        self.makeHeadingRow(separator=separator)

        cwd = os.getcwd()  # noqa: PTH109
        save_file_dir = os.path.dirname(cwd)  # noqa: PTH120
        save_file_full_path = os.path.join(save_file_dir, save_file_name)  # noqa: PTH118
        with open(save_file_full_path, 'w') as f:  # noqa: PLW1514, PTH123
            f.write(self.headingRow)
            f.write('\n')
            for sample_num, sample in enumerate(self.trace):
                row = self.makeOneRowString(
                    sample_num=sample_num, sample=sample, separator=separator
                )
                f.write(row)
                f.write('\n')

    def startTimer(self):  # noqa: N802, D102
        self.startingTime = time.time()

    def computeTimeElapsed(self):  # noqa: N802, D102
        self.timeElapsed = time.time() - self.startingTime

    def printTimeElapsed(self):  # noqa: N802, D102
        self.computeTimeElapsed()
        print(f'Time elapsed: {self.timeElapsed / 60:0.2f} minutes')  # noqa: T201

    def startSectionTimer(self):  # noqa: N802, D102
        self.sectionStartingTime = time.time()

    def resetSectionTimer(self):  # noqa: N802, D102
        self.startSectionTimer()

    def computeSectionTimeElapsed(self):  # noqa: N802, D102
        self.sectionTimeElapsed = time.time() - self.sectionStartingTime

    def printSectionTimeElapsed(self):  # noqa: N802, D102
        self.computeSectionTimeElapsed()
        print(f'Time elapsed: {self.sectionTimeElapsed / 60:0.2f} minutes')  # noqa: T201

    @staticmethod
    def printEndMessages():  # noqa: N802, D102
        print('Heirarchical Bayesian estimation done!')  # noqa: T201

    def runUQ(  # noqa: N802
        self,
        uqData,  # noqa: N803
        simulationData,  # noqa: ARG002, N803
        randomVarsData,  # noqa: ARG002, N803
        demandParams,  # noqa: ARG002, N803
        workingDir,  # noqa: N803
        runType,  # noqa: ARG002, N803
        localAppDir,  # noqa: ARG002, N803
        remoteAppDir,  # noqa: ARG002, N803
    ):
        """This function configures and runs hierarchical Bayesian estimation based on the
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
        """  # noqa: D205, D400, D401, D404
        self.startTimer()
        self.storeUQData(uqData=uqData)
        os.chdir(workingDir)
        self.performHeirBayesSampling()
        self.saveResultsToPklFile()
        self.makeTabularResultsFile()
        self.printTimeElapsed()
        self.printEndMessages()


class testRunUQ:  # noqa: D101
    def __init__(self, json_file_path_string) -> None:
        self.json_file_path_string = json_file_path_string
        self.getUQData()
        self.createRunner()
        self.runTest()

    def getUQData(self):  # noqa: N802, D102
        with open(os.path.abspath(self.json_file_path_string)) as f:  # noqa: PLW1514, PTH100, PTH123
            input_data = json.load(f)

        self.ApplicationData = input_data['Applications']
        self.uqData = input_data['UQ']
        self.simulationData = self.ApplicationData['FEM']
        self.randomVarsData = input_data['randomVariables']
        self.demandParams = input_data['EDP']
        self.localAppDir = input_data['localAppDir']
        self.remoteAppDir = input_data['remoteAppDir']
        self.workingDir = input_data['workingDir']
        self.workingDir = os.path.join(  # noqa: PTH118
            self.workingDir, 'tmp.SimCenter', 'templateDir'
        )
        self.runType = 'runningLocal'

    def createRunner(self):  # noqa: N802, D102
        self.runner = HeirBayesRunner()

    def runTest(self):  # noqa: N802, D102
        self.runner.runUQ(
            uqData=self.uqData,
            simulationData=self.simulationData,
            randomVarsData=self.randomVarsData,
            demandParams=self.demandParams,
            workingDir=self.workingDir,
            runType=self.runType,
            localAppDir=self.localAppDir,
            remoteAppDir=self.remoteAppDir,
        )


def main():  # noqa: D103
    filename = os.path.abspath(  # noqa: PTH100
        os.path.join(  # noqa: PTH118
            os.path.dirname(__file__),  # noqa: PTH120
            'test_CustomUQ/HeirBayesSyntheticData/templatedir/scInput.json',
        )
    )
    if os.path.exists(filename):  # noqa: PTH110
        testRunUQ(filename)
    else:
        print(f'Test input json file {filename} not found. Not running the test.')  # noqa: T201


if __name__ == '__main__':
    main()
