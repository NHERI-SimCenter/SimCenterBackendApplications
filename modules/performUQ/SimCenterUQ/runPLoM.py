# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Leland Stanford Junior University
# Copyright (c) 2021 The Regents of the University of California
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
# This module is modified from the surrogateBuild.py to use PLoM for surrogate
# modeling while maintaining similar input/output formats compatable with the current workflow
#
# Contributors:
# Kuanshi Zhong
# Sang-ri Yi
# Frank Mckenna
#
import shutil
import os
import sys
import json
from scipy.stats import lognorm, norm
import numpy as np
from multiprocessing import Pool
from PLoM.PLoM import *
import pandas as pd
import subprocess

# ==========================================================================================

class runPLoM:

    """
    runPLoM: class for run a PLoM job
    methods:
        __init__: initialization
        _create_variables: create variable name lists
        _parse_plom_parameters: parse PLoM modeling parameters
        _set_up_parallel: set up paralleling configurations
        _load_variables: load training data
        train_model: model training
        save_model: model saving
    """

    def __init__(self, work_dir, run_type, os_type, job_config, errlog, input_file, workflow_driver):

        """
        __init__
        input: 
            work_dir: working directory
            run_type: job type
            os_type: operating system type
            job_config: configuration (dtype = dict)
            errlog: error log object
        """
        
        # read inputs
        self.work_dir = work_dir
        self.run_type = run_type
        self.os_type = os_type
        self.errlog = errlog
        self.job_config = job_config
        self.input_file = input_file
        self.workflow_driver = workflow_driver

        # initialization
        self.rv_name = list()
        self.g_name = list()
        self.x_dim = 0
        self.y_dim = 0

        # read variable names
        # self.x_dim, self.y_dim, self.rv_name, self.g_name = self._create_variables(job_config)

        # read PLoM parameters
        surrogateInfo = job_config["UQ_Method"]["surrogateMethodInfo"]
        if self._parse_plom_parameters(surrogateInfo):
            msg = 'runPLoM.__init__: Error in reading PLoM parameters.'
            self.errlog.exit(msg)

        # parallel setup
        self.do_parallel = surrogateInfo.get("parallelExecution", False)
        if self.do_parallel:
            if self._set_up_parallel():
                msg = 'runPLoM.__init__: Error in setting up parallel.'
                self.errlog.exit(msg)
        else:
            self.pool = 0
            self.cal_interval = 5

        # prepare training data
        if surrogateInfo["method"] == "Import Data File":
            do_sampling = False
            do_simulation = not surrogateInfo["outputData"]
            self.doe_method = "None"  # default
            do_doe = False
            self.inpData = os.path.join(work_dir, "templatedir/inpFile.in")
            if not do_simulation:
                self.outData = os.path.join(work_dir, "templatedir/outFile.in")
            self._create_variables_from_input()
        elif surrogateInfo["method"] == "Sampling and Simulation":
            # run simulation first to generate training data
            do_sampling = False
            do_simulation = False
            self._run_simulation()
        else:
            msg = 'Error reading json: only supporting "Import Data File"'
            errlog.exit(msg)

        # read variable names
        #self.x_dim, self.y_dim, self.rv_name, self.g_name = self._create_variables(surrogateInfo["method"])

        # load variables
        if self._load_variables(do_sampling, do_simulation):
            msg = 'runPLoM.__init__: Error in loading variables.'
            self.errlog.exit(msg)


    def _run_simulation(self):

        """
        _run_simulation: running simulation to get training data
        input:
            job_config: job configuration dictionary
        output:
            None
        """
        import platform

        job_config = self.job_config

        # get python instance
        runType = job_config.get('runType','runningLocal')
        if (sys.platform == 'darwin' or sys.platform == "linux" or sys.platform == "linux2"):
            pythonEXE = 'python3'
        else:
            pythonEXE = 'python'
        if runType == 'runningLocal' and platform.system() == 'Windows':
            localAppDir = job_config.get('localAppDir',None)
            if localAppDir is None:
                # no local app directory is found, let's try to use system python
                pass
            else:
                #pythonEXE = os.path.join(localAppDir,'applications','python','python.exe')
                pythonEXE = '\"' + sys.executable + '\"'
        else:
            # for remote run and macOS, let's use system python
            pass

        # move into the templatedir
        run_dir = job_config.get('runDir',os.getcwd())
        os.chdir(run_dir)
        # training is done for single building (for now)
        bldg_id = None
        if bldg_id is not None:
            os.chdir(bldg_id)
        os.chdir('templatedir')

        # dakota script path
        dakotaScript = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'dakota','DakotaUQ.py')

        print('dakotaScript = ',dakotaScript)

        # write a new dakota.json for forward propogation
        ## KZ modified 0331
        with open(self.input_file,'r') as f:
            tmp = json.load(f)
        tmp['UQ_Method']['uqType'] = 'Forward Propagation'
        tmp['UQ_Method']['parallelExecution'] = True
        samplingObj = tmp['UQ_Method']['surrogateMethodInfo']['samplingMethod']
        tmp['UQ_Method']['samplingMethodData']=dict()
        ## KZ modified 0331
        tmp['UQ_Method']['uqEngine'] = 'Dakota'
        tmp['Applications']['UQ']['Application'] = 'Dakota-UQ'
        for key, item in samplingObj.items():
            tmp['UQ_Method']['samplingMethodData'][key] = item
        with open('sc_dakota_plom.json','w') as f:
            json.dump(tmp, f, indent=2)

        # command line
        ## KZ modified 0331
        command_line = f"{pythonEXE} {dakotaScript} --workflowInput sc_dakota_plom.json --driverFile {os.path.splitext(self.workflow_driver)[0]} --workflowOutput EDP.json --runType {runType}"
        print(command_line)
        # run command

        try:
            #os.system(command_line)
            subprocess.call(command_line)
        except:
            print('runPLoM._run_simulation: error in running dakota to generate the initial sample.')
            print('runPLoM._run_simulation: please check if the dakota is installed correctly on the system.')

        # remove the new dakota.json
        #os.remove('sc_dakota_plom.json')

        if runType in ['run', 'runningLocal']:
            # create the response.csv file from the dakotaTab.out file
            os.chdir(run_dir)
            if bldg_id is not None:
                os.chdir(bldg_id)
            dakota_out = pd.read_csv('dakotaTab.out', sep=r'\s+', header=0, index_col=0)
            # save to csv
            dakota_out.to_csv('response.csv')
            # create a IM.csv file
            self._compute_IM(run_dir, pythonEXE)
            # collect IMs and RVs to the PLoM_variables.csv and EDPs to PLoM_responses.csv
            self.inpData, self.outData = self._prepare_training_data(run_dir)
            # update job_config['randomVariables']
            cur_rv_list = [x.get('name') for x in job_config['randomVariables']]
            for curRV in self.rv_name:
                if curRV not in cur_rv_list:
                    job_config['randomVariables'].append({'distribution': 'Normal', 'name': curRV})
            self.job_config = job_config

        elif self.run_type in ['set_up', 'runningRemote']:
            pass

    
    def _prepare_training_data(self, run_dir):

        # load IM.csv if exists
        df_IM = pd.DataFrame()
        if os.path.exists(os.path.join(run_dir,'IM.csv')):
            df_IM = pd.read_csv(os.path.join(run_dir,'IM.csv'),index_col=None)
        else:
            msg = 'runPLoM._prepare_training_data: no IM.csv in {}.'.format(run_dir)
            print(msg)

        # load response.csv if exists
        df_SIMU = pd.DataFrame()
        if os.path.exists(os.path.join(run_dir,'response.csv')):
            df_SIMU = pd.read_csv(os.path.join(run_dir,'response.csv'),index_col=None)
        else:
            msg = 'runPLoM._prepare_training_data: response.csv not found in {}.'.format(run_dir)
            self.errlog.exit(msg)

        # read BIM to get RV names
        # KZ modified 0331
        with open(os.path.join(run_dir, 'templatedir', self.input_file)) as f:
            tmp = json.load(f)
        rVs = tmp.get('randomVariables', None)
        if rVs is None:
            rv_names = []
        else:
            rv_names = [x.get('name') for x in rVs]
        
        # collect rv columns from df_SIMU
        df_RV = pd.DataFrame()
        if len(rv_names) > 0:
            df_RV = df_SIMU[rv_names]
            for cur_rv in rv_names:
                df_SIMU.pop(cur_rv)
        if '%eval_id' in list(df_SIMU.columns):
            df_SIMU.pop('%eval_id')
        if 'interface' in list(df_SIMU.columns):
            df_SIMU.pop('interface')
        if 'MultipleEvent' in list(df_SIMU.columns):
            self.multipleEvent = df_SIMU.pop('MultipleEvent')
        else:
            self.multipleEvent = None
        
        # concat df_RV and df_IM
        if not df_IM.empty:
            df_X = pd.concat([df_IM, df_RV], axis=1)
        else:
            df_X = df_RV
        if not df_X.empty and '%eval_id' in list(df_X.columns):
            df_X.pop('%eval_id')
        if not df_X.empty and '%MultipleEvent' in list(df_X.columns):
            self.multipleEvent = df_X.pop('%MultipleEvent')
        elif not df_X.empty and 'MultipleEvent' in list(df_X.columns):
            self.multipleEvent = df_X.pop('MultipleEvent')

        # make the first column name start with %
        if not df_X.empty:
            df_X = df_X.rename({list(df_X.columns)[0]:'%'+list(df_X.columns)[0]}, axis='columns')
        df_SIMU = df_SIMU.rename({list(df_SIMU.columns)[0]:'%'+list(df_SIMU.columns)[0]}, axis='columns')

        # save to csvs
        inpData = os.path.join(run_dir,'PLoM_variables.csv')
        outData = os.path.join(run_dir,'PLoM_responses.csv')
        df_X.to_csv(inpData,index=False)
        df_SIMU.to_csv(outData,index=False)

        # set rv_names, g_name, x_dim, y_dim
        self.rv_name = list(df_X.columns)
        self.g_name = list(df_SIMU.columns)
        self.x_dim = len(self.rv_name)
        self.y_dim = len(self.g_name)

        return inpData, outData


    def _compute_IM(self, run_dir, pythonEXE):

        # find workdirs
        workdir_list = [x for x in os.listdir(run_dir) if x.startswith('workdir')]

        # intensity measure app
        computeIM = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    'createEVENT','groundMotionIM','IntensityMeasureComputer.py')
            
        # compute IMs
        for cur_workdir in workdir_list:
            os.chdir(cur_workdir)
            if os.path.exists('EVENT.json') and os.path.exists('BIM.json'):
                os.system(f"{pythonEXE} {computeIM} --filenameBIM BIM.json --filenameEVENT EVENT.json --filenameIM IM.json")
            os.chdir(run_dir)

        # collect IMs from different workdirs
        for i, cur_workdir in enumerate(workdir_list):
            cur_id = int(cur_workdir.split('.')[-1])
            if os.path.exists(os.path.join(cur_workdir,'IM.csv')):
                try:
                    tmp1 = pd.read_csv(os.path.join(cur_workdir,'IM.csv'),index_col=None)
                except:
                    return
                if tmp1.empty:
                    return
                tmp2 = pd.DataFrame({'%eval_id': [cur_id for x in range(len(tmp1.index))]})
                if i == 0:
                    im_collector = pd.concat([tmp2, tmp1], axis=1)
                else:
                    tmp3 = pd.concat([tmp2, tmp1], axis=1)
                    im_collector = pd.concat([im_collector, tmp3])
            else:
                return
        im_collector = im_collector.sort_values(by=['%eval_id'])
        im_collector.to_csv('IM.csv',index=False)


    def _create_variables(self, training_data):

        """
        create_variables: creating X and Y variables
        input:
            training_data: training data source
        output:
            x_dim: dimension of X data
            y_dim: dimension of Y data
            rv_name: random variable name (X data)
            g_name: variable name (Y data)
        """

        job_config = self.job_config

        # initialization
        rv_name = self.rv_name
        g_name = self.g_name
        x_dim = self.x_dim
        y_dim = self.y_dim

        # check if training data source from simulation
        if training_data == 'Sampling and Simulation':
            return x_dim, y_dim, rv_name, g_name

        # read X and Y variable names
        for rv in job_config['randomVariables']:
            rv_name = rv_name + [rv['name']]
            x_dim += 1
        if x_dim == 0:
            msg = 'Error reading json: RV is empty'
            self.errlog.exit(msg)
        for g in job_config['EDP']:
            if g['length']==1: # scalar
                g_name = g_name + [g['name']]
                y_dim += 1
            else: # vector
                for nl in range(g['length']):
                    g_name = g_name + ["{}_{}".format(g['name'],nl+1)]
                    y_dim += 1
        if y_dim == 0:
            msg = 'Error reading json: EDP(QoI) is empty'
            self.errlog.exit(msg)

        # return
        return x_dim, y_dim, rv_name, g_name


    def _create_variables_from_input(self):

        df_variables = pd.read_csv(self.inpData, header=0)
        df_responses = pd.read_csv(self.outData, header=0)

        self.rv_name = list(df_variables.columns)
        self.g_name = list(df_responses.columns)
        self.x_dim = len(self.rv_name)
        self.y_dim = len(self.g_name)

        if 'MultipleEvent' in list(df_variables.columns):
            self.multipleEvent = df_variables.pop('MultipleEvent')
        else:
            self.multipleEvent = None

        if 'MultipleEvent' in list(df_responses.columns):
            self.multipleEvent = df_responses.pop('MultipleEvent')
        else:
            self.multipleEvent = None


    def _parse_plom_parameters(self, surrogateInfo):

        """
        _parse_plom_parameters: parse PLoM parameters from surrogateInfo
        input:
            surrogateInfo: surrogate information dictionary
        output:
            run_flag: 0 - sucess, 1: failure
        """

        run_flag = 0
        try:
            self.n_mc = int(surrogateInfo['newSampleRatio'])
            self.epsilonPCA = surrogateInfo.get("epsilonPCA",1e-6)
            self.smootherKDE = surrogateInfo.get("smootherKDE",25)
            self.randomSeed = surrogateInfo.get("randomSeed",None)
            self.diffMap = surrogateInfo.get("diffusionMaps",True)
            self.logTransform = surrogateInfo.get("logTransform",False)
            self.constraintsFlag = surrogateInfo.get("constraints",False)
            self.kdeTolerance = surrogateInfo.get("kdeTolerance",0.1)
            if self.constraintsFlag:
                self.constraintsFile = os.path.join(work_dir, "templatedir/plomConstraints.py")
            self.numIter = surrogateInfo.get("numIter",50)
            self.tolIter = surrogateInfo.get("tolIter",0.02)
            self.preTrained = surrogateInfo.get("preTrained",False)
            if self.preTrained:
                self.preTrainedModel = os.path.join(work_dir, "templatedir/surrogatePLoM.h5")
        except:
            run_flag = 1

        # return
        return run_flag


    def _set_up_parallel(self):

        """
        _set_up_parallel: set up modules and variables for parallel jobs
        input:
            none
        output:
            run_flag: 0 - sucess, 1 - failure
        """

        run_flag = 0
        try:
            if self.run_type.lower() == 'runninglocal':
                self.n_processor = os.cpu_count()
                from multiprocessing import Pool
                self.pool = Pool(self.n_processor)
            else:
                from mpi4py import MPI
                from mpi4py.futures import MPIPoolExecutor
                self.world = MPI.COMM_WORLD
                self.pool = MPIPoolExecutor()
                self.n_processor = self.world.Get_size()
            print("nprocessor :")
            print(self.n_processor)
            self.cal_interval = self.n_processor
        except:
            run_flag = 1

        # return
        return run_flag


    def _load_variables(self, do_sampling, do_simulation):
        
        """
        _load_variables: load variables
        input:
            do_sampling: sampling flag
            do_simulation: simulation flag
            job_config: job configuration dictionary
        output:
            run_flag: 0 - sucess, 1 - failure
        """
        job_config = self.job_config

        run_flag = 0
        #try:
        if do_sampling:
            pass
        else:
            X = read_txt(self.inpData, self.errlog)
            print('X = ', X)
            print(X.columns)
            if len(X.columns) != self.x_dim:
                msg = 'Error importing input data: Number of dimension inconsistent: have {} RV(s) but {} column(s).' \
                    .format(self.x_dim, len(X.columns))
                errlog.exit(msg)
            if self.logTransform:
                X = np.log(X)

        if do_simulation:
            pass
        else:
            Y = read_txt(self.outData, self.errlog)
            if Y.shape[1] != self.y_dim:
                msg = 'Error importing input data: Number of dimension inconsistent: have {} QoI(s) but {} column(s).' \
                    .format(self.y_dim, len(Y.columns))
                errlog.exit(msg)
            if self.logTransform:
                Y = np.log(Y)

            if X.shape[0] != Y.shape[0]:
                msg = 'Warning importing input data: numbers of samples of inputs ({}) and outputs ({}) are inconsistent'.format(len(X.columns), len(Y.columns))
                print(msg)

            n_samp = Y.shape[0]
            # writing a data file for PLoM input
            self.X = X.to_numpy()
            self.Y = Y.to_numpy()
            inputXY = os.path.join(work_dir, "templatedir/inputXY.csv")
            X_Y = pd.concat([X,Y], axis=1)
            X_Y.to_csv(inputXY, sep=',', header=True, index=False)
            self.inputXY = inputXY
            self.n_samp = n_samp

            self.do_sampling = do_sampling
            self.do_simulation = do_simulation
            self.rvName = []
            self.rvDist = []
            self.rvVal = []
            try:
                for nx in range(self.x_dim):
                    rvInfo = job_config["randomVariables"][nx]
                    self.rvName = self.rvName + [rvInfo["name"]]
                    self.rvDist = self.rvDist + [rvInfo["distribution"]]
                    if do_sampling:
                        self.rvVal = self.rvVal + [(rvInfo["upperbound"] + rvInfo["lowerbound"]) / 2]
                    else:
                        self.rvVal = self.rvVal + [np.mean(self.X[:, nx])]
            except:
                msg = 'Warning: randomVariables attributes in configuration file are not consistent with x_dim'
                print(msg)
        #except:
        #    run_flag = 1

        # return
        return run_flag


    def train_model(self, model_name='SurrogatePLoM'):
        db_path = os.path.join(self.work_dir, 'templatedir')
        if not self.preTrained:
            self.modelPLoM = PLoM(model_name=model_name, data=self.inputXY, separator=',', col_header=True, db_path=db_path, 
                tol_pca = self.epsilonPCA, epsilon_kde = self.smootherKDE, runDiffMaps = self.diffMap, plot_tag = True)
        else:
            self.modelPLoM = PLoM(model_name=model_name, data=self.preTrainedModel, db_path=db_path, 
                tol_pca = self.epsilonPCA, epsilon_kde = self.smootherKDE, runDiffMaps = self.diffMap)
        if self.constraintsFlag:
            self.modelPLoM.add_constraints(self.constraintsFile)
        if self.n_mc > 0:
            tasks = ['DataNormalization','RunPCA','RunKDE','ISDEGeneration']
        else:
            tasks = ['DataNormalization','RunPCA','RunKDE']
        self.modelPLoM.ConfigTasks(task_list=tasks)
        self.modelPLoM.RunAlgorithm(n_mc=self.n_mc, tol = self.tolIter, max_iter = self.numIter, seed_num=self.randomSeed, tolKDE=self.kdeTolerance)
        if self.n_mc > 0:
            self.modelPLoM.export_results(data_list=['/X0','/X_new'])
        else:
            self.modelPLoM.export_results(data_list=['/X0'])
        self.pcaEigen = self.modelPLoM.mu
        self.pcaError = self.modelPLoM.errPCA
        self.pcaComp = self.modelPLoM.nu
        self.kdeEigen = self.modelPLoM.eigenKDE
        self.kdeComp = self.modelPLoM.m
        self.Errors = []
        if self.constraintsFlag:
            self.Errors = self.modelPLoM.errors


    def save_model(self):

        # copy the h5 model file to the main work dir
        shutil.copy2(os.path.join(self.work_dir,'templatedir','SurrogatePLoM','SurrogatePLoM.h5'),self.work_dir)
        if self.n_mc > 0:
            shutil.copy2(os.path.join(self.work_dir,'templatedir','SurrogatePLoM','DataOut','X_new.csv'),self.work_dir)

        if self.X.shape[0] > 0:
            header_string_x = ' ' + ' '.join([str(elem).replace('%','') for elem in self.rv_name]) + ' '
        else:
            header_string_x = ' '
        header_string_y = ' ' + ' '.join([str(elem).replace('%','') for elem in self.g_name])
        header_string = header_string_x[:-1] + header_string_y

        #xy_data = np.concatenate((np.asmatrix(np.arange(1, self.n_samp + 1)).T, self.X, self.Y), axis=1)
        #np.savetxt(self.work_dir + '/dakotaTab.out', xy_data, header=header_string, fmt='%1.4e', comments='%')
        #np.savetxt(self.work_dir + '/inputTab.out', self.X, header=header_string_x[1:-1], fmt='%1.4e', comments='%')
        #np.savetxt(self.work_dir + '/outputTab.out', self.Y, header=header_string_y[1:], fmt='%1.4e', comments='%')
        df_inputTab = pd.DataFrame(data=self.X, columns=self.rv_name)
        df_outputTab = pd.DataFrame(data=self.Y, columns=self.g_name)
        df_inputTab.to_csv(os.path.join(self.work_dir,'inputTab.out'),index=False)
        df_outputTab.to_csv(os.path.join(self.work_dir,'outputTab.out'),index=False)

        results = {}

        results["valSamp"] = self.n_samp
        results["xdim"] = self.x_dim
        results["ydim"] = self.y_dim
        results["xlabels"] = self.rv_name
        results["ylabels"] = self.g_name
        results["yExact"] = {}
        results["xPredict"] = {}
        results["yPredict"] = {}
        results["valNRMSE"] = {}
        results["valR2"] = {}
        results["valCorrCoeff"] = {}
        for ny in range(self.y_dim):
            results["yExact"][self.g_name[ny]] = self.Y[:, ny].tolist()

        results["inpData"] = self.inpData
        if not self.do_simulation:
            results["outData"] = self.outData

        results["logTransform"] = self.logTransform

        rv_list = []
        try:
            for nx in range(self.x_dim):
                rvs = {}
                rvs["name"] = self.rvName[nx]
                rvs["distribution"] = self.rvDist[nx]
                rvs["value"] = self.rvVal[nx]
                rv_list = rv_list + [rvs]
            results["randomVariables"] = rv_list
        except:
            msg = 'Warning: randomVariables attributes in configuration file are not consistent with x_dim'
            print(msg)
        results["dirPLoM"] = os.path.join(os.path.dirname(os.path.abspath(__file__)),'PLoM')

        results["pcaEigen"] = self.pcaEigen.tolist()
        results["pcaError"] = self.pcaError
        results["pcaComp"] = self.pcaComp
        results["kdeEigen"] = self.kdeEigen.tolist()
        results["kdeComp"] = self.kdeComp
        results["Errors"] = self.Errors
        
        if self.n_mc > 0:
            Xnew = pd.read_csv(self.work_dir + '/X_new.csv', header=0, index_col=0)
            if self.logTransform:
                Xnew = np.exp(Xnew)
            for nx in range(self.x_dim):
                results["xPredict"][self.rv_name[nx]] = Xnew.iloc[:, nx].tolist()

            for ny in range(self.y_dim):
                results["yPredict"][self.g_name[ny]] = Xnew.iloc[:, self.x_dim+ny].tolist()

        if self.X.shape[0]>0:
            xy_data = np.concatenate((np.asmatrix(np.arange(1, self.Y.shape[0] + 1)).T, self.X, self.Y), axis=1)
        else:
            xy_data = np.concatenate((np.asmatrix(np.arange(1, self.Y.shape[0] + 1)).T, self.Y), axis=1)
        np.savetxt(self.work_dir + '/dakotaTab.out', xy_data, header=header_string, fmt='%1.4e', comments='%')
        
        # KZ: adding MultipleEvent if any
        if self.multipleEvent is not None:
            tmp = pd.read_csv(os.path.join(self.work_dir,'dakotaTab.out'),index_col=None,sep=' ')
            tmp = pd.concat([tmp,self.multipleEvent],axis=1)
            tmp.to_csv(os.path.join(self.work_dir,'dakotaTab.out'),index=False,sep=' ')

            #if not self.do_logtransform:
            #results["yPredict_CI_lb"][self.g_name[ny]] = norm.ppf(0.25, loc = results["yPredict"][self.g_name[ny]] , scale = np.sqrt(self.Y_loo_var[:, ny])).tolist()
            #results["yPredict_CI_ub"][self.g_name[ny]] = norm.ppf(0.75, loc = results["yPredict"][self.g_name[ny]] , scale = np.sqrt(self.Y_loo_var[:, ny])).tolist()
            #else:
            #    mu = np.log(self.Y_loo[:, ny] )
            #    sig = np.sqrt(np.log(self.Y_loo_var[:, ny]/pow(self.Y_loo[:, ny] ,2)+1))
            #    results["yPredict_CI_lb"][self.g_name[ny]] =  lognorm.ppf(0.25, s = sig, scale = np.exp(mu)).tolist()
            #    results["yPredict_CI_ub"][self.g_name[ny]] =  lognorm.ppf(0.75, s = sig, scale = np.exp(mu)).tolist()
        
        # over-write the data with Xnew if any
        if self.n_mc > 0:
            Xnew.insert(0,'%',[x+1 for x in list(Xnew.index)])
            Xnew.to_csv(self.work_dir + '/dakotaTab.out', index=False, sep=' ')

        if os.path.exists('dakota.out'):
            os.remove('dakota.out')

        with open('dakota.out', 'w') as fp:
            json.dump(results, fp, indent=2)

        print("Results Saved")


def read_txt(text_dir, errlog):

    if not os.path.exists(text_dir):
        msg = "Error: file does not exist: " + text_dir
        errlog.exit(msg)

    header_line = []
    with open(text_dir) as f:
        # Iterate through the file until the table starts
        header_count = 0
        for line in f:
            if line.startswith('%'):
                header_count = header_count + 1
                header_line = line[1:] # remove '%'
        try:
            with open(text_dir) as f:
                X = np.loadtxt(f, skiprows=header_count)
        except ValueError:
            try:
                with open(text_dir) as f:
                    X = np.genfromtxt(f, skip_header=header_count, delimiter=',')
                # if there are extra delimiter, remove nan
                if np.isnan(X[-1, -1]):
                    X = np.delete(X, -1, 1)
            except ValueError:
                msg = "Error: file format is not supported " + text_dir
                errlog.exit(msg)

    if X.ndim == 1:
        X = np.array([X]).transpose()

    print('X = ', X)

    #df_X = pd.DataFrame(data=X, columns=["V"+str(x) for x in range(X.shape[1])])
    if len(header_line) > 0:
        df_X = pd.DataFrame(data=X, columns=header_line.replace('\n','').split(','))
    else:
        df_X = pd.DataFrame()

    print('df_X = ',df_X)

    return df_X
    

class errorLog(object):

    def __init__(self, work_dir):
        self.file = open('{}/dakota.err'.format(work_dir), "w")

    def exit(self, msg):
        print(msg)
        self.file.write(msg)
        self.file.close()
        exit(-1)


def build_surrogate(work_dir, os_type, run_type, input_file, workflow_driver):
    
    """
    build_surrogate: built surrogate model
    input:
        work_dir: working directory
        run_type: job type
        os_type: operating system type
    """

    # t_total = time.process_time()
    # default filename
    filename = 'PLoM_Model'
    # read the configuration file
    f = open(work_dir + '/templatedir/' + input_file)
    try:
        job_config = json.load(f)
    except ValueError:
        msg = 'invalid json format - ' + input_file
        errlog.exit(msg)
    f.close()

    # check the uq type
    if job_config['UQ_Method']['uqType'] != 'PLoM Model':
        msg = 'UQ type inconsistency : user wanted <' + job_config['UQ_Method']['uqType'] + \
            '> but called <PLoM Model> program'
        errlog.exit(msg)

    # initializing runPLoM
    model = runPLoM(work_dir, run_type, os_type, job_config, errlog, input_file, workflow_driver)
    # training the model
    model.train_model()
    # save the model
    model.save_model()


if __name__ == "__main__":

    """
    shell command: PYTHON runPLoM.py work_dir run_type os_type
    work_dir: working directory
    run_type: job type
    os_type: operating system type
    """

    # collect arguments
    inputArgs = sys.argv
    # working diretory
    work_dir = inputArgs[1].replace(os.sep, '/')
    print('work_dir = {}'.format(work_dir))
    # print the work_dir
    errlog = errorLog(work_dir)
    # job type
    run_type = inputArgs[5]
    # operating system type
    os_type = inputArgs[4]
    # default output file: results.out
    result_file = "results.out"
    # input file name
    input_file = os.path.basename(inputArgs[2])
    print('input_file = {}'.format(input_file))
    # workflowDriver
    workflow_driver = inputArgs[3]
    # start build the surrogate
    build_surrogate(work_dir, os_type, run_type, input_file, workflow_driver)    