#  # noqa: INP001, D100
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
# This module is to create a Gaussian Process surrogate model for quoFEM
#
# Initial Writer:
# Sang-ri Yi
#

# Jan 31, 2023: let's not use GPy calibration parallel for now, because it seems to give local maxima

import copy  # noqa: I001
import json
import os
import pickle
import random
import sys
import time
import warnings
import numpy as np
import random  # noqa: F811

warnings.filterwarnings('ignore')


file_dir = os.path.dirname(__file__)  # noqa: PTH120
sys.path.append(file_dir)

from UQengine import UQengine  # noqa: E402

# import pip installed modules


try:
    moduleName = 'numpy'  # noqa: N816
    import numpy as np

    moduleName = 'GPy'  # noqa: N816
    import GPy as GPy  # noqa: PLC0414

    moduleName = 'scipy.stats'  # noqa: N816
    from scipy.stats import cramervonmises, lognorm, norm, qmc  # noqa: I001
    import scipy

    moduleName = 'sklearn.linear_model'  # noqa: N816
    from sklearn.linear_model import LinearRegression

    moduleName = 'UQengine'  # noqa: N816

    # from utilities import run_FEM_batch, errorLog
    error_tag = False  # global variable
except:  # noqa: E722
    error_tag = True
    print('Failed to import module:' + moduleName)  # type: ignore # noqa: T201


print('Initializing error log file..')  # noqa: T201
print(f'Current working dir (getcwd): {os.getcwd()}')  # noqa: T201, PTH109

work_dir_tmp = sys.argv[1].replace(os.sep, '/')
errFileName = os.path.join(work_dir_tmp, 'dakota.err')  # noqa: N816, PTH118

develop_mode = len(sys.argv) == 8  # a flag for develeopmode  # noqa: PLR2004
if develop_mode:
    # import matplotlib
    # matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    print('Developer mode')  # noqa: T201
else:
    with open(errFileName, 'w') as f:  # noqa: PTH123
        f.write('')
    sys.stderr = open(errFileName, 'w')  # noqa: SIM115, PTH123
    print(f'Error file created at: {errFileName}')  # noqa: T201


#
# Modify GPy package
#

if error_tag == False:  # noqa: E712

    def monkeypatch_method(cls):  # noqa: D103
        def decorator(func):
            setattr(cls, func.__name__, func)
            return func

        return decorator

    @monkeypatch_method(GPy.models.gp_regression.GPRegression)
    def randomize(self, rand_gen=None, *args, **kwargs):  # noqa: D103
        if rand_gen is None:
            rand_gen = np.random.normal
        # first take care of all parameters (from N(0,1))
        x = rand_gen(size=self._size_transformed(), *args, **kwargs)  # noqa: B026
        updates = self.update_model()
        self.update_model(False)  # Switch off the updates  # noqa: FBT003
        self.optimizer_array = x  # makes sure all of the tied parameters get the same init (since there's only one prior object...)
        # now draw from prior where possible
        x = self.param_array.copy()
        [
            np.put(x, ind, p.rvs(ind.size))
            for p, ind in self.priors.items()
            if p is not None
        ]
        unfixlist = np.ones((self.size,), dtype=bool)
        from paramz.transformations import __fixed__

        unfixlist[self.constraints[__fixed__]] = False
        self.param_array.flat[unfixlist] = x.view(np.ndarray).ravel()[unfixlist]
        self.update_model(updates)

# Main function


def main(inputArgs):  # noqa: N803, D103
    gp = surrogate(inputArgs)  # noqa: F841


class surrogate(UQengine):  # noqa: D101
    def __init__(self, inputArgs):  # noqa: N803
        super(surrogate, self).__init__(inputArgs)  # noqa: UP008
        t_init = time.time()

        #
        # Check if there was error in importing python packages
        #
        self.create_errLog()
        self.check_packages(error_tag, moduleName)
        self.cleanup_workdir()

        #
        # Read Json File
        #

        self.readJson()

        #
        # Create GP wrapper
        #

        self.create_gp_model()

        #
        # run DoE
        #
        self.train_surrogate(t_init)

        #
        # save model as
        #

        self.save_model('SimGpModel')

    def check_packages(self, error_tag, moduleName):  # noqa: N803, D102
        if error_tag == True and moduleName == 'GPy':  # noqa: E712
            if self.os_type.lower().startswith('darwin'):
                msg = 'Surrogate modeling module uses GPy python package which is facing a version compatibility issue at this moment (01.05.2024). To use the surrogate module, one needs to update manually the GPy version to 1.13. The instruction can be found in the the documentation: https://nheri-simcenter.github.io/quoFEM-Documentation/common/user_manual/usage/desktop/SimCenterUQSurrogate.html#lblsimsurrogate'
                self.exit(msg)

        if error_tag == True:  # noqa: E712
            if self.os_type.lower().startswith('win'):
                msg = (
                    'Failed to load python module ['
                    + moduleName
                    + ']. Go to File-Preference-Python and reset the path.'
                )
            else:
                msg = (
                    'Failed to load python module ['
                    + moduleName
                    + ']. Did you forget <pip3 install nheri_simcenter --upgrade>?'
                )
            self.exit(msg)

    def readJson(self):  # noqa: C901, N802, D102, PLR0912, PLR0915
        # self.nopt = max([20, self.n_processor])
        self.nopt = 3
        self.is_paralle_opt_safe = False

        try:
            jsonPath = self.inputFile  # for EEUQ  # noqa: N806
            if not os.path.isabs(jsonPath):  # noqa: PTH117
                jsonPath = (  # noqa: N806
                    self.work_dir + '/templatedir/' + self.inputFile
                )  # for quoFEM

            with open(jsonPath, encoding='utf-8') as f:  # noqa: PTH123
                dakotaJson = json.load(f)  # noqa: N806

        except ValueError:
            msg = 'invalid json format - dakota.json'
            self.exit(msg)
        if dakotaJson['UQ']['uqType'] != 'Train GP Surrogate Model':
            msg = (
                'UQ type inconsistency : user wanted <'
                + dakotaJson['UQ']['uqType']
                + '> but we called <Global Surrogate Modeling> program'
            )
            self.exit(msg)

        surrogateJson = dakotaJson['UQ']['surrogateMethodInfo']  # noqa: N806

        if surrogateJson['method'] == 'Sampling and Simulation':
            self.global_seed = surrogateJson['seed']
        else:
            self.global_seed = 42

        random.seed(self.global_seed)
        np.random.seed(self.global_seed)
        #
        # EE-UQ
        #
        # TODO: multihazards?  # noqa: TD002
        self.isEEUQ = False
        self.isWEUQ = False
        if dakotaJson['Applications'].get('Events') != None:  # noqa: E711
            Evt = dakotaJson['Applications']['Events']  # noqa: N806
            if Evt[0].get('EventClassification') != None:  # noqa: E711
                if Evt[0]['EventClassification'] == 'Earthquake':
                    self.isEEUQ = True
                elif Evt[0]['EventClassification'] == 'Wind':
                    self.isWEUQ = True

        self.rv_name_ee = []
        if surrogateJson.get('IntensityMeasure') != None and self.isEEUQ:  # noqa: E711
            self.intensityMeasure = surrogateJson['IntensityMeasure']
            self.intensityMeasure['useGeoMean'] = surrogateJson['useGeoMean']
            self.unitInfo = dakotaJson['GeneralInformation']['units']
            for imName, imChar in surrogateJson['IntensityMeasure'].items():  # noqa: B007, N806, PERF102
                # if imChar.get("Periods") != None:
                #     for pers in imChar["Periods"]:
                #         self.rv_name_ee += [imName+str(pers)]
                # else:
                #     self.rv_name_ee += [imName]
                self.rv_name_ee += [imName]
        else:
            self.IntensityMeasure = {}
            self.unitInfo = {}

        if self.isEEUQ or self.isWEUQ:
            self.checkWorkflow(dakotaJson)
        #
        #  common for all surrogate options
        #

        self.rv_name = list()  # noqa: C408
        x_dim = 0

        for rv in dakotaJson['randomVariables']:
            self.rv_name += [rv['name']]
            x_dim += 1

        self.g_name = list()  # noqa: C408
        y_dim = 0

        for g in dakotaJson['EDP']:
            # scalar
            if not g['name']:
                msg = 'QoI name cannot be an empty string'
                self.exit(msg)

            if g['length'] == 1:
                self.g_name += [g['name']]
                y_dim += 1
            # vector
            else:
                for nl in range(g['length']):
                    self.g_name += ['{}_{}'.format(g['name'], nl + 1)]
                    y_dim += 1

        if x_dim == 0:
            msg = 'Error reading json: RV is empty'
            self.exit(msg)

        if y_dim == 0:
            msg = 'Error reading json: EDP(QoI) is empty'
            self.exit(msg)

        do_predictive = False
        automate_doe = False  # noqa: F841

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.do_predictive = do_predictive

        try:
            self.do_parallel = surrogateJson['parallelExecution']
        except:  # noqa: E722
            self.do_parallel = True

        if self.do_parallel:
            self.n_processor, self.pool = self.make_pool(self.global_seed)
            self.cal_interval = self.n_processor
        else:
            self.n_processor = 1
            self.pool = 0
            self.cal_interval = 5
        print(f'self.cal_interval : {self.cal_interval}')  # noqa: T201

        #
        #  Advanced
        #

        self.heteroscedastic = False
        # if surrogateJson["advancedOpt"]:

        self.do_logtransform = surrogateJson['logTransform']
        self.kernel = surrogateJson['kernel']
        self.do_linear = surrogateJson['linear']

        self.nugget_opt = surrogateJson['nuggetOpt']
        # self.heteroscedastic = surrogateJson["Heteroscedastic"]

        if (self.nugget_opt == 'Fixed Values') or (  # noqa: PLR1714
            self.nugget_opt == 'Fixed Bounds'
        ):
            try:
                self.nuggetVal = np.array(
                    json.loads('[{}]'.format(surrogateJson['nuggetString']))
                )
            except json.decoder.JSONDecodeError:
                msg = 'Error reading json: improper format of nugget values/bounds. Provide nugget values/bounds of each QoI with comma delimiter'
                self.exit(msg)

            if (
                self.nuggetVal.shape[0] != self.y_dim
                and self.nuggetVal.shape[0] != 0
            ):
                msg = f'Error reading json: Number of nugget quantities ({self.nuggetVal.shape[0]}) does not match # QoIs ({self.y_dim})'
                self.exit(msg)
        else:
            self.nuggetVal = 1

        if self.nugget_opt == 'Heteroscedastic':
            self.stochastic = [True] * y_dim
        else:
            self.stochastic = [False] * y_dim

        if self.nugget_opt == 'Fixed Values':
            for Vals in self.nuggetVal:  # noqa: N806
                if not np.isscalar(Vals):
                    msg = 'Error reading json: provide nugget values of each QoI with comma delimiter'
                    self.exit(msg)

        elif self.nugget_opt == 'Fixed Bounds':
            for Bous in self.nuggetVal:  # noqa: N806
                if np.isscalar(Bous):
                    msg = 'Error reading json: provide nugget bounds of each QoI in brackets with comma delimiter, e.g. [0.0,1.0],[0.0,2.0],...'
                    self.exit(msg)
                elif isinstance(Bous, list):
                    msg = 'Error reading json: provide both lower and upper bounds of nugget'
                    self.exit(msg)
                elif Bous.shape[0] != 2:  # noqa: PLR2004
                    msg = 'Error reading json: provide nugget bounds of each QoI in brackets with comma delimiter, e.g. [0.0,1.0],[0.0,2.0],...'
                    self.exit(msg)
                elif Bous[0] > Bous[1]:
                    msg = 'Error reading json: the lower bound of a nugget value should be smaller than its upper bound'
                    self.exit(msg)
        # else:
        #     # use default
        #     self.do_logtransform = False
        #     self.kernel = 'Matern 5/2'
        #     self.do_linear = False
        #     self.nugget_opt = "Optimize"
        #     self.nuggetVal= 1
        #     self.stochastic =[False] * y_dim

        if self.stochastic[0]:

            @monkeypatch_method(GPy.likelihoods.Gaussian)
            def gaussian_variance(self, Y_metadata=None):  # noqa: N803
                if Y_metadata is None:
                    return self.variance
                else:  # noqa: RET505
                    return self.variance * Y_metadata['variance_structure']

            @monkeypatch_method(GPy.core.GP)
            def set_XY2(self, X=None, Y=None, Y_metadata=None):  # noqa: N802, N803
                if Y_metadata is not None:
                    if self.Y_metadata is None:
                        self.Y_metadata = Y_metadata
                    else:
                        self.Y_metadata.update(Y_metadata)
                        print('metadata_updated')  # noqa: T201
                self.set_XY(X, Y)

            @monkeypatch_method(GPy.core.GP)
            def subsample_XY(self, idx):  # noqa: N802, N803, RUF100
                if self.Y_metadata is not None:
                    new_meta = self.Y_metadata
                    new_meta['variance_structure'] = self.Y_metadata[
                        'variance_structure'
                    ][idx]
                    self.Y_metadata.update(new_meta)
                    print('metadata_updated')  # noqa: T201
                self.set_XY(self.X[idx, :], self.Y[idx, :])

        # Save model information
        if (surrogateJson['method'] == 'Sampling and Simulation') or (
            surrogateJson['method'] == 'Import Data File'
        ):
            self.do_mf = False
            self.modelInfoHF = model_info(
                surrogateJson,
                dakotaJson['randomVariables'],
                self.work_dir,
                x_dim,
                y_dim,
                self.n_processor,
                self.global_seed,
                idx=0,
            )
            self.modelInfoLF = model_info(
                surrogateJson,
                dakotaJson['randomVariables'],
                self.work_dir,
                x_dim,
                y_dim,
                self.n_processor,
                self.global_seed,
                idx=-1,
            )  # NONE model
        elif surrogateJson['method'] == 'Import Multi-fidelity Data File':
            self.do_mf = True
            self.modelInfoHF = model_info(
                surrogateJson['highFidelity'],
                dakotaJson['randomVariables'],
                self.work_dir,
                x_dim,
                y_dim,
                self.n_processor,
                self.global_seed,
                idx=1,
            )
            self.modelInfoLF = model_info(
                surrogateJson['lowFidelity'],
                dakotaJson['randomVariables'],
                self.work_dir,
                x_dim,
                y_dim,
                self.n_processor,
                self.global_seed,
                idx=2,
            )
        else:
            msg = 'Error reading json: select among "Import Data File", "Sampling and Simulation" or "Import Multi-fidelity Data File"'
            self.exit(msg)

        if self.do_mf:
            try:
                moduleName = 'emukit'  # noqa: N806

                error_tag = False  # global variable
            except:  # noqa: E722
                error_tag = True  # noqa: F841
                print('Failed to import module:' + moduleName)  # noqa: T201

        if self.modelInfoHF.is_model:
            self.ll = self.modelInfoHF.ll
            self.doe_method = self.modelInfoHF.doe_method
            self.thr_NRMSE = self.modelInfoHF.thr_NRMSE
            self.thr_t = self.modelInfoHF.thr_t
        elif self.modelInfoLF.is_model:
            self.ll = self.modelInfoLF.ll
            self.doe_method = self.modelInfoLF.doe_method
            self.thr_NRMSE = self.modelInfoLF.thr_NRMSE
            self.thr_t = self.modelInfoLF.thr_t
        elif self.modelInfoHF.is_data:
            self.ll = self.modelInfoHF.ll
            self.doe_method = self.modelInfoLF.doe_method  # whatever.
            self.thr_NRMSE = self.modelInfoLF.thr_NRMSE  # whatever.
            self.thr_t = self.modelInfoLF.thr_t  # whatever.
        else:
            self.ll = self.modelInfoLF.ll  # whatever.
            self.doe_method = self.modelInfoLF.doe_method  # whatever.
            self.thr_NRMSE = self.modelInfoLF.thr_NRMSE  # whatever.
            self.thr_t = self.modelInfoLF.thr_t  # whatever.

        self.modelInfoHF.runIdx = 0
        self.modelInfoLF.runIdx = 0
        if self.modelInfoHF.is_model and self.modelInfoLF.is_model:
            self.doeIdx = 'HFLF'  # HFHF is for multi-fidelity GPy
            self.modelInfoHF.runIdx = 1
            self.modelInfoLF.runIdx = 2
            self.cal_interval = 1
        elif not self.modelInfoHF.is_model and self.modelInfoLF.is_model:
            self.doeIdx = 'LF'
        elif self.modelInfoHF.is_model and not self.modelInfoLF.is_model:
            self.doeIdx = 'HF'
        else:
            self.doeIdx = 'HF'  # whatever.

        #
        # For later use..
        #
        # self.femInfo = dakotaJson["fem"]
        # surrogateJson["fem"] = dakotaJson["fem"]

        self.rvName = []
        self.rvDist = []
        self.rvVal = []
        self.rvDiscStr = []
        self.rvDiscIdx = []
        for nx in range(x_dim):
            rvInfo = dakotaJson['randomVariables'][nx]  # noqa: N806
            self.rvName = self.rvName + [rvInfo['name']]  # noqa: RUF005
            self.rvDist = self.rvDist + [rvInfo['distribution']]  # noqa: RUF005
            if self.modelInfoHF.is_model:
                if rvInfo['distribution'] == 'Uniform':
                    self.rvVal += [(rvInfo['upperbound'] + rvInfo['lowerbound']) / 2]
                    self.rvDiscStr += [[]]
                elif rvInfo['distribution'] == 'discrete_design_set_string':
                    self.rvVal += [1]
                    self.rvDiscStr += [rvInfo['elements']]
                    self.rvDiscIdx = [nx]

            elif self.modelInfoHF.is_data:
                self.rvVal = self.rvVal + [  # noqa: RUF005
                    np.mean(self.modelInfoHF.X_existing[:, nx])
                ]
            else:
                self.rvVal = [0] * self.x_dim

    def checkWorkflow(self, dakotaJson):  # noqa: N802, N803, D102
        if dakotaJson['Applications']['EDP']['Application'] == 'SurrogateEDP':
            msg = 'Error in SurrogateGP engine: Do not select [None] in the EDP tab. [None] is used only when using pre-trained surrogate, i.e. when [Surrogate] is selected in the SIM Tab.'
            self.exit(msg)

        if (
            dakotaJson['Applications']['Simulation']['Application']
            == 'SurrogateSimulation'
        ):
            msg = 'Error in SurrogateGP engine: Do not select [None] in the FEM tab. [None] is used only when using pre-trained surrogate, i.e. when [Surrogate] is selected in the SIM Tab.'
            self.exit(msg)

        maxSampSize = float('Inf')  # noqa: N806
        for rv in dakotaJson['randomVariables']:
            if rv['distribution'] == 'discrete_design_set_string':
                maxSampSize = len(rv['elements'])  # noqa: N806

        if (maxSampSize < dakotaJson['UQ']['surrogateMethodInfo']['samples']) and (
            'IntensityMeasure' in dakotaJson['UQ']['surrogateMethodInfo'].keys()  # noqa: SIM118
        ):
            # if #sample is smaller than #GM & IM is used as input
            msg = 'Error in SurrogateGP engine: The number of samples ({}) should NOT be greater than the number of ground motions ({}). Using the same number is highly recommended.'.format(
                dakotaJson['UQ']['surrogateMethodInfo']['samples'], maxSampSize
            )
            self.exit(msg)

    def create_kernel(self, x_dim):  # noqa: D102
        kernel = self.kernel
        if kernel == 'Radial Basis':
            kr = GPy.kern.RBF(input_dim=x_dim, ARD=True)
        elif kernel == 'Exponential':
            kr = GPy.kern.Exponential(input_dim=x_dim, ARD=True)
        elif kernel == 'Matern 3/2':
            kr = GPy.kern.Matern32(input_dim=x_dim, ARD=True)
        elif kernel == 'Matern 5/2':
            kr = GPy.kern.Matern52(input_dim=x_dim, ARD=True)
        else:
            msg = f'Error running SimCenterUQ - Kernel name <{kernel}> not supported'
            self.exit(msg)

        # if self.do_linear:
        #    kr = kr + GPy.kern.Linear(input_dim=x_dim, ARD=True)

        if self.do_mf:
            kr = emf.kernels.LinearMultiFidelityKernel([kr.copy(), kr.copy()])  # type: ignore # noqa: F821

        return kr

    def create_gpy_model(self, X_dummy, Y_dummy, kr):  # noqa: N803, D102
        if not self.do_mf:
            if not self.heteroscedastic:
                m_tmp = GPy.models.GPRegression(
                    X_dummy,
                    Y_dummy,
                    kernel=kr.copy(),
                    normalizer=self.set_normalizer,
                )
            else:
                self.set_normalizer = False
                m_tmp = GPy.models.GPHeteroscedasticRegression(
                    X_dummy, Y_dummy, kernel=kr.copy()
                )

            # for parname in m_tmp.parameter_names():
            #    if parname.endswith("lengthscale"):
            #        exec("m_tmp." + parname + "=self.ll")

            # for multi fidelity case
        else:
            X_list, Y_list = emf.convert_lists_to_array.convert_xy_lists_to_arrays(  # noqa: N806, F821 # type: ignore
                [X_dummy, X_dummy], [Y_dummy, Y_dummy]
            )

            for i in range(y_dim):  # type: ignore # noqa: B007, F821
                m_tmp = GPyMultiOutputWrapper(  # noqa: F821 # type: ignore
                    emf.models.GPyLinearMultiFidelityModel(  # noqa: F821 # type: ignore
                        X_list, Y_list, kernel=kr.copy(), n_fidelities=2
                    ),
                    2,
                    n_optimization_restarts=self.nopt,
                )

        return m_tmp

    def create_gp_model(self):  # noqa: D102
        x_dim = self.x_dim
        y_dim = self.y_dim

        # choose kernel
        kr = self.create_kernel(x_dim)

        X_dummy = np.zeros((1, x_dim))  # noqa: N806
        Y_dummy = np.zeros((1, y_dim))  # noqa: N806
        # for single fidelity case

        self.set_normalizer = True

        self.normMeans = [0] * y_dim
        self.normVars = [1] * y_dim
        self.m_list = [0] * self.y_dim
        if self.do_linear:
            self.lin_list = [0] * self.y_dim
        self.m_var_list, self.var_str, self.Y_mean = (
            [0] * self.y_dim,
            [1] * self.y_dim,
            [0] * self.y_dim,
        )
        # below to facilitate better optimization bounds
        self.init_noise_var = [None] * y_dim
        self.init_process_var = [None] * y_dim

        for i in range(y_dim):
            self.m_list[i] = self.create_gpy_model(X_dummy, Y_dummy, kr)

        self.x_dim = x_dim
        self.y_dim = y_dim

    def predict(self, m_tmp, X, noise=0):  # noqa: ARG002, N803, D102
        if not self.do_mf:
            if all(np.mean(m_tmp.Y, axis=0) == m_tmp.Y):
                return m_tmp.Y[
                    0
                ], 0  # if response is constant - just return constant
            elif self.heteroscedastic:  # noqa: RET505
                return m_tmp.predict_noiseless(X)
            else:
                return m_tmp.predict_noiseless(X)
        else:
            idxHF = np.argwhere(m_tmp.gpy_model.X[:, -1] == 0)  # noqa: N806
            if all(
                np.mean(m_tmp.gpy_model.Y[idxHF, :], axis=0) == m_tmp.gpy_model.Y
            ):
                return (
                    m_tmp.gpy_model.Y[0],
                    0,
                )  # if high-fidelity response is constant - just return constant
            else:  # noqa: RET505
                X_list = convert_x_list_to_array([X, X])  # type: ignore # noqa: N806, F821
                X_list_h = X_list[X.shape[0] :]  # noqa: N806
                return m_tmp.predict(X_list_h)

    def set_XY(  # noqa: C901, N802, D102
        self,
        m_tmp,
        ny,
        X_hf,  # noqa: N803
        Y_hf,  # noqa: N803
        X_lf=float('nan'),  # noqa: N803
        Y_lf=float('nan'),  # noqa: N803
        enforce_hom=False,  # noqa: FBT002
    ):
        #
        # check if X dimension has changed...
        #
        x_current_dim = self.x_dim
        for parname in m_tmp.parameter_names():
            if parname.endswith('lengthscale'):
                exec('x_current_dim = len(m_tmp.' + parname + ')')  # noqa: S102

        if x_current_dim != X_hf.shape[1]:
            kr = self.create_kernel(X_hf.shape[1])
            X_dummy = np.zeros((1, X_hf.shape[1]))  # noqa: N806
            Y_dummy = np.zeros((1, 1))  # noqa: N806
            m_new = self.create_gpy_model(X_dummy, Y_dummy, kr)
            m_tmp = m_new.copy()
            # m_tmp.optimize()

        if self.do_logtransform:
            if np.min(Y_hf) < 0:
                msg = 'Error running SimCenterUQ - Response contains negative values. Please uncheck the log-transform option in the UQ tab'
                self.exit(msg)

            Y_hfs = np.log(Y_hf)  # noqa: N806
        else:
            Y_hfs = Y_hf  # noqa: N806

        if self.do_logtransform and self.do_mf:
            if np.min(Y_lf) < 0:
                msg = 'Error running SimCenterUQ - Response contains negative values. Please uncheck the log-transform option in the UQ tab'
                self.exit(msg)

            Y_lfs = np.log(Y_lf)  # noqa: N806
        else:
            Y_lfs = Y_lf  # noqa: N806

        if self.do_linear:
            self.linear_list = [True] * X_hf.shape[1]  # SY - TEMPORARY
        else:
            self.linear_list = [False] * X_hf.shape[1]  # SY - TEMPORARY

        if sum(self.linear_list) > 0:
            self.lin_list[ny] = LinearRegression().fit(
                X_hf[:, self.linear_list], Y_hfs
            )
            y_pred = self.lin_list[ny].predict(X_hf[:, self.linear_list])
            Y_hfs = Y_hfs - y_pred  # noqa: N806

        # # below is dummy
        # if np.all(np.isnan(X_lf)) and np.all(np.isnan(Y_lf)):
        #     X_lf = self.X_lf
        #     Y_lfs = self.Y_lf

        if not self.do_mf:
            # if self.heteroscedastic:
            #     m_tmp = GPy.models.GPHeteroscedasticRegression(
            #         X_hf, Y_hfs, kernel=self.kg.copy()
            #     )
            #     # TODO: temporary... need to find a way to not calibrate but update the variance  # noqa: TD002
            #     m_tmp.optimize()
            #     self.var_str[ny] = np.ones((m_tmp.Y.shape[0], 1))

            X_new, X_idx, indices, counts = np.unique(  # noqa: N806
                X_hf,
                axis=0,
                return_index=True,
                return_counts=True,
                return_inverse=True,
            )
            n_unique = X_new.shape[0]

            # if n_unique == X_hf.shape[0]: # unique set p - just to homogeneous GP
            if not self.stochastic[ny] or enforce_hom:
                # just homogeneous GP

                m_tmp.set_XY(X_hf, Y_hfs)
                self.var_str[ny] = np.ones((m_tmp.Y.shape[0], 1))
                self.Y_mean[ny] = Y_hfs
                self.indices_unique = range(Y_hfs.shape[0])
                self.n_unique_hf = X_hf.shape[0]

            elif n_unique == X_hf.shape[0]:  # no repl
                # Y_mean=Y_hfs[X_idx]
                # Y_mean1, nugget_mean1 = self.predictStoMeans(X_new, Y_mean)
                (
                    Y_mean1,  # noqa: N806
                    nugget_mean1,
                    initial_noise_variance,
                    initial_process_variance,
                ) = self.predictStoMeans(X_hf, Y_hfs)

                if np.max(nugget_mean1) < 1.0e-10:  # noqa: PLR2004
                    self.set_XY(m_tmp, ny, X_hf, Y_hfs, enforce_hom=True)
                    return None
                else:  # noqa: RET505
                    Y_metadata, m_var, norm_var_str = self.predictStoVars(  # noqa: N806
                        X_hf, (Y_hfs - Y_mean1) ** 2, X_hf, Y_hfs, counts
                    )
                    m_tmp.set_XY2(X_hf, Y_hfs, Y_metadata=Y_metadata)

                    self.m_var_list[ny] = m_var
                    self.var_str[ny] = norm_var_str
                    self.indices_unique = range(Y_hfs.shape[0])
                    self.n_unique_hf = X_new.shape[0]
                    self.Y_mean[ny] = Y_hfs
                    self.init_noise_var[ny] = initial_noise_variance
                    self.init_process_var[ny] = initial_process_variance
            else:
                # nonunique set - check if nugget is zero
                Y_mean, Y_var = np.zeros((n_unique, 1)), np.zeros((n_unique, 1))  # noqa: N806

                for idx in range(n_unique):
                    Y_subset = Y_hfs[[i for i in np.where(indices == idx)[0]], :]  # noqa: C416, N806
                    Y_mean[idx, :] = np.mean(Y_subset, axis=0)
                    Y_var[idx, :] = np.var(Y_subset, axis=0)

                idx_repl = [i for i in np.where(counts > 1)[0]]  # noqa: C416

                if np.max(Y_var) / np.var(Y_mean) < 1.0e-10:  # noqa: PLR2004
                    # NUGGET IS ZERO - no need for stochastic kriging

                    if self.do_logtransform:
                        Y_mean = np.exp(Y_mean)  # noqa: N806

                    m_tmp = self.set_XY(
                        m_tmp, ny, X_new, Y_mean, X_lf, Y_lf
                    )  # send only unique to nonstochastic

                    self.indices_unique = indices
                    return m_tmp

                elif self.nugget_opt == 'Heteroscedastic':  # noqa: RET505
                    (
                        dummy,
                        dummy,  # noqa: PLW0128
                        initial_noise_variance,
                        initial_process_variance,
                    ) = self.predictStoMeans(
                        X_hf,
                        Y_hfs,
                    )  # noqa: N806, RUF100
                    #
                    # Constructing secondary GP model - can we make use of the "variance of sample variance"
                    #
                    # TODO: log-variance  # noqa: TD002

                    Y_metadata, m_var, norm_var_str = self.predictStoVars(  # noqa: N806
                        X_new[idx_repl, :],
                        Y_var[idx_repl],
                        X_new,
                        Y_mean,
                        counts,
                    )
                    """
                        kernel_var = GPy.kern.Matern52(input_dim=self.x_dim, ARD=True)
                        log_vars = np.log(Y_var[idx_repl])
                        m_var = GPy.models.GPRegression(X_new[idx_repl, :], log_vars, kernel_var, normalizer=True,
                                                        Y_metadata=None)
                        print("Optimizing variance field of ny={}".format(ny))
                        m_var.optimize(messages=True, max_f_eval=1000)
                        m_var.optimize_restarts(20,parallel=True, num_processes=self.n_processor)
                        log_var_pred, dum = m_var.predict(X_new)
                        var_pred = np.exp(log_var_pred)
                        #
                        #

                        # norm_var_str = (var_pred.T[0]/counts) / max(var_pred.T[0]/counts)
                        if self.set_normalizer:
                            norm_var_str = (var_pred.T[0]) / np.var(Y_mean)  # if normalization was used..
                        else:
                            norm_var_str = (var_pred.T[0])  # if normalization was used..

                        # norm_var_str = (X_new+2)**2/max((X_new+2)**2)
                        Y_metadata = {'variance_structure': norm_var_str / counts}
                        """
                    m_tmp.set_XY2(X_new, Y_mean, Y_metadata=Y_metadata)

                    self.m_var_list[ny] = m_var
                    self.var_str[ny] = norm_var_str
                    self.indices_unique = indices
                    self.n_unique_hf = X_new.shape[0]
                    self.Y_mean[ny] = Y_mean
                    self.init_noise_var[ny] = initial_noise_variance
                    self.init_process_var[ny] = initial_process_variance

                else:
                    # still nonstochastic gp
                    m_tmp.set_XY(X_hf, Y_hfs)
                    self.var_str[ny] = np.ones((m_tmp.Y.shape[0], 1))
                    self.indices_unique = range(Y_hfs.shape[0])
                    self.Y_mean[ny] = Y_hfs
                    self.n_unique_hf = X_hf.shape[0]
                    self.stochastic[ny] = False

        else:
            (
                X_list_tmp,  # noqa: N806
                Y_list_tmp,  # noqa: N806
            ) = emf.convert_lists_to_array.convert_xy_lists_to_arrays(  # noqa: F821 # type: ignore
                [X_hf, X_lf], [Y_hfs, Y_lfs]
            )
            m_tmp.set_data(X=X_list_tmp, Y=Y_list_tmp)
            self.n_unique_hf = X_hf.shape[0]

        if self.set_normalizer:
            if not self.do_mf:
                self.normMeans[ny] = m_tmp.normalizer.mean
                self.normVars[ny] = m_tmp.normalizer.std**2
            else:
                self.normMeans[ny] = 0
                self.normVars[ny] = 1

        return m_tmp

    def predictStoVars(self, X_repl, Y_var_repl, X_new, Y_mean, counts):  # noqa: ARG002, D102, N802, N803
        my_x_dim = X_repl.shape[1]
        # kernel_var = GPy.kern.Matern52(
        #    input_dim=my_x_dim, ARD=True
        # ) + GPy.kern.Linear(input_dim=my_x_dim, ARD=True)

        kernel_var = GPy.kern.Matern52(input_dim=my_x_dim, ARD=True)
        log_vars = np.log(Y_var_repl)
        m_var = GPy.models.GPRegression(
            X_repl, log_vars, kernel_var, normalizer=True, Y_metadata=None
        )
        # m_var.Gaussian_noise.constrain_bounded(0.2, 2.0, warning=False)
        m_var.Gaussian_noise.constrain_bounded(0.5, 2.0, warning=False)
        m_var.Gaussian_noise = 1  # initial points

        for parname in m_var.parameter_names():
            if parname.endswith('lengthscale'):
                for nx in range(X_repl.shape[1]):
                    myrange = np.max(X_repl, axis=0) - np.min(X_repl, axis=0)
                    m_var.Mat52.lengthscale[[nx]].constrain_bounded(
                        myrange[nx] / X_repl.shape[0], myrange[nx] * 5, warning=False
                    )
                    m_var.Mat52.lengthscale[[nx]] = myrange[nx]  # initial points
                    # m_var.Gaussian_noise.value = 0.05
                    # m_var.Gaussian_noise.constrain_bounded(0.1/np.var(log_vars), 0.8/np.var(log_vars), warning=False)
                    # m_var.Mat52.lengthscale[[nx]].constrain_bounded(
                    #    myrange[nx] / X_repl.shape[0],
                    #    myrange[nx] * 10,
                    #    warning=False,
                    # )
                    # m_var.sum.Mat52.lengthscale[[nx]].constrain_bounded(
                    #    myrange[nx] / X_repl.shape[0] * 10,
                    #    myrange[nx] * 100,
                    #    warning=False,
                    # )
                    # TODO change the kernel  # noqa: TD002, TD004
        # m_var.optimize(max_f_eval=1000)
        # m_var.optimize_restarts(
        #    self.nopt, parallel=self.is_paralle_opt_safe, num_processes=self.n_processor, verbose=False
        # )
        print('Calibrating Secondary surrogate')  # noqa: T201
        m_var = my_optimize_restart(m_var, self.nopt)

        # print(m_var)  # noqa: RUF100, T201

        log_var_pred, dum = m_var.predict(X_new)
        var_pred = np.exp(log_var_pred)

        norm_var_str = (var_pred.T[0] / counts) / max(var_pred.T[0] / counts)
        self.var_norm = 1
        if self.set_normalizer:
            self.var_norm = np.mean(var_pred.T[0])
            # norm_var_str = (var_pred.T[0]) / np.var(Y_mean)
            norm_var_str = var_pred.T[0] / self.var_norm
            # if normalization was used..
        else:
            norm_var_str = var_pred.T[0]  # if normalization was not used..


        # norm_var_str = (X_new+2)**2/max((X_new+2)**2)
        Y_metadata = {'variance_structure': norm_var_str / counts}  # noqa: N806

        if develop_mode:
            plt.figure(3)
            nx = 0
            plt.scatter(X_repl[:, nx], log_vars, alpha=0.1)
            plt.scatter(X_new[:, nx], log_var_pred, alpha=0.1)
            plt.show()
            plt.figure(1)
            dum1, dum2 = m_var.predict(X_repl)

            plt.title('Sto Log-var QoI')
            plt.scatter(log_vars, dum1, alpha=0.5)
            plt.scatter(log_vars, log_vars, alpha=0.5)
            plt.xlabel('exact')
            plt.ylabel('pred')
            plt.grid()
            print(m_var)  # noqa: T201
            print(m_var.Mat52.lengthscale)  # noqa: T201
            plt.show()
        return Y_metadata, m_var, norm_var_str

    def predictStoMeans(self, X, Y):  # noqa: N802, N803, D102
        # under homoscedasticity
        my_x_dim = X.shape[1]
        myrange = np.max(X, axis=0) - np.min(X, axis=0)
        kernel_mean = GPy.kern.Matern52(
            input_dim=my_x_dim, ARD=True, lengthscale=myrange
        )
        # kernel_mean = GPy.kern.Matern52(input_dim=my_x_dim, ARD=True) + GPy.kern.Linear(input_dim=my_x_dim, ARD=True)
        # if self.do_linear and not (self.isEEUQ or self.isWEUQ):
        #    kernel_mean = kernel_mean + GPy.kern.Linear(input_dim=my_x_dim, ARD=True)
        #
        # if sum(self.linear_list)>0:
        #
        #     lin_tmp = LinearRegression().fit(X[:, self.linear_list], Y)
        #     y_pred = lin_tmp.predict(X[:, self.linear_list])
        #
        #     # Set GP
        #
        #     m_mean = GPy.models.GPRegression(
        #         X, Y - y_pred, kernel_mean, normalizer=True, Y_metadata=None
        #     )
        #
        # else:
        #     m_mean = GPy.models.GPRegression(
        #         X, Y, kernel_mean, normalizer=True, Y_metadata=None
        #     )
        m_mean = GPy.models.GPRegression(
            X, Y, kernel_mean, normalizer=True, Y_metadata=None
        )

        """
        for parname in m_mean.parameter_names():
            if parname.endswith('lengthscale'):
                for nx in range(X.shape[1]):
                    # m_mean.kern.Mat52.lengthscale[[nx]]=  myrange[nx]*100
                    # m_mean.kern.Mat52.lengthscale[[nx]].constrain_bounded(myrange[nx]/X.shape[0]*50, myrange[nx]*100)
                    # if self.isEEUQ:
                    #     # m_mean.kern.lengthscale[[nx]] = myrange[nx] * 100
                    #     # m_mean.kern.lengthscale[[nx]].constrain_bounded(
                    #     #     myrange[nx] / X.shape[0] * 50,
                    #     #     myrange[nx] * 100,
                    #     #     warning=False,
                    #     # )
                    #     # m_mean.kern.lengthscale[[nx]] = myrange[nx]
                    #     m_mean.kern.lengthscale[[nx]].constrain_bounded(
                    #         myrange[nx] / X.shape[0],
                    #         myrange[nx] * 10000,
                    #         warning=False,
                    #     )
                    if self.do_linear:
                        # m_mean.kern.Mat52.lengthscale[[nx]] = myrange[nx] * 5000
                        m_mean.kern.Mat52.lengthscale[[nx]].constrain_bounded(
                            myrange[nx] / X.shape[0] * 50,
                            myrange[nx] * 10000,
                            warning=False,
                        )
                    else:

                        m_mean.Gaussian_noise.constrain_bounded(0.1,0.5,warning=False)
                        # m_mean.kern.lengthscale[[nx]] = myrange[nx]
                        m_mean.kern.lengthscale[[nx]].constrain_bounded(
                          myrange[nx]/ X.shape[0]*10,
                          myrange[nx] * 1,
                          warning=False
                        )
        """

        # m_mean.optimize(messages=True, max_f_eval=1000)
        # # m_mean.Gaussian_noise.variance = np.var(Y) # First calibrate parameters
        # m_mean.optimize_restarts(
        #     self.nopt, parallel=self.is_paralle_opt_safe, num_processes=self.n_processor, verbose=True
        # )  # First calibrate parameters
        print('calibrating tertiary surrogate')  # noqa: T201
        m_mean = my_optimize_restart(m_mean, self.nopt)

        # m_mean.optimize(messages=True, max_f_eval=1000)

        # if self.do_linear:

        # m_mean.Gaussian_noise.variance=m_mean.Mat52.variance+m_mean.Gaussian_noise.variance
        # else:
        # m_mean.Gaussian_noise.variance=m_mean.RBF.variance+m_mean.Gaussian_noise.variance.variance
        # m_mean.optimize_restarts(10,parallel=True)

        mean_pred, mean_var = m_mean.predict(X)

        initial_noise_variance = float(m_mean.Gaussian_noise.variance)
        initial_process_variance = float(m_mean.Mat52.variance)

        # if np.sum(self.linear_list) > 0:
        #    mean_pred = mean_pred + lin_tmp.predict(X[:, self.linear_list])

        if develop_mode:
            print(m_mean)  # noqa: T201
            plt.scatter(X[:, 0], Y, alpha=0.2)
            plt.plot(X[:, 0], mean_pred, 'rx')
            plt.errorbar(
                X[:, 0],
                mean_pred.T[0],
                yerr=np.sqrt(mean_var.T)[0],
                fmt='rx',
                alpha=0.1,
            )
            plt.title('Sto Log-mean QoI (initial)')
            plt.show()

            plt.plot(m_mean.Y, mean_pred, 'rx')
            plt.scatter(m_mean.Y, m_mean.Y, alpha=0.2)
            plt.title('Sto Log-mean QoI (initial)')
            plt.show()

        return mean_pred, mean_var, initial_noise_variance, initial_process_variance

    def calibrate(self):  # noqa: C901, D102, RUF100
        print('Calibrating in parallel', flush=True)  # noqa: T201
        warnings.filterwarnings('ignore')
        t_opt = time.time()
        nugget_opt_tmp = self.nugget_opt
        nopt = self.nopt

        parallel_calib = True
        # parallel_calib = self.do_parallel

        if parallel_calib:
            iterables = (
                (
                    copy.deepcopy(self.m_list[ny]),
                    nugget_opt_tmp,
                    self.nuggetVal,
                    self.normVars[ny],
                    self.do_mf,
                    self.heteroscedastic,
                    nopt,
                    ny,
                    self.n_processor,
                    self.is_paralle_opt_safe,
                    self.init_noise_var[ny],
                    self.init_process_var[ny],
                )
                for ny in range(self.y_dim)
            )
            result_objs = list(self.pool.starmap(calibrating, iterables))
            for m_tmp, msg, ny in result_objs:
                self.m_list[ny] = m_tmp
                if msg != '':
                    self.exit(msg)

            # TODO: terminate it gracefully....  # noqa: TD002
            # see https://stackoverflow.com/questions/21104997/keyboard-interrupt-with-pythons-multiprocessing
            if develop_mode:
                for ny in range(self.y_dim):
                    print(self.m_list[ny])  # noqa: T201
                    # print(m_tmp.rbf.lengthscale)
                    tmp = self.m_list[ny].predict(self.m_list[ny].X)[0]
                    # this one has a noise
                    plt.title('Original Mean QoI')
                    plt.scatter(self.m_list[ny].Y, tmp, alpha=0.1)
                    plt.scatter(self.m_list[ny].Y, self.m_list[ny].Y, alpha=0.1)
                    plt.xlabel('exact')
                    plt.ylabel('pred')
                    plt.show()

                    plt.title('Original Mean QoI')
                    plt.scatter(self.m_list[ny].X[:, 0], tmp, alpha=0.1)
                    plt.scatter(
                        self.m_list[ny].X[:, 0], self.m_list[ny].Y, alpha=0.1
                    )
                    plt.xlabel('exact')
                    plt.ylabel('pred')
                    plt.show()
        else:
            for ny in range(self.y_dim):
                self.m_list[ny], msg, ny = calibrating(  # noqa: PLW2901
                    copy.deepcopy(self.m_list[ny]),
                    nugget_opt_tmp,
                    self.nuggetVal,
                    self.normVars[ny],
                    self.do_mf,
                    self.heteroscedastic,
                    self.nopt,
                    ny,
                    self.n_processor,
                    self.is_paralle_opt_safe,
                    self.init_noise_var[ny],
                    self.init_process_var[ny],
                )
                if msg != '':
                    self.exit(msg)
            ####
            if develop_mode:
                print(self.m_list[ny])  # noqa: T201
                # print(m_tmp.rbf.lengthscale)
                tmp = self.m_list[ny].predict(self.m_list[ny].X)
                plt.title('Final Mean QoI')
                plt.scatter(self.m_list[ny].Y, tmp[0], alpha=0.1)
                plt.scatter(self.m_list[ny].Y, self.m_list[ny].Y, alpha=0.1)
                plt.xlabel('exact')
                plt.ylabel('pred')
                plt.show()

        # because EE-UQ results are more likely to have huge nugget.
        # if False:
        # if self.isEEUQ:
        #     if self.heteroscedastic:
        #         variance_keyword = 'het_Gauss.variance'
        #     else:
        #         variance_keyword = 'Gaussian_noise.variance'
        #
        #     for ny in range(self.y_dim):
        #         for parname in self.m_list[ny].parameter_names():
        #             if parname.endswith('variance') and ('Gauss' not in parname):
        #                 exec(  # noqa: RUF100, S102
        #                     'my_new_var = max(self.m_list[ny].'
        #                     + variance_keyword
        #                     + ', 10*self.m_list[ny].'
        #                     + parname
        #                     + ')'
        #                 )
        #                 exec('self.m_list[ny].' + variance_keyword + '= my_new_var')  # noqa: RUF100, S102
        #
        #         self.m_list[ny].optimize()

        self.calib_time = time.time() - t_opt
        print(f'     Calibration time: {self.calib_time:.2f} s', flush=True)  # noqa: T201
        Y_preds, Y_pred_vars, Y_pred_vars_w_measures, e2 = (  # noqa: N806
            self.get_cross_validation_err()
        )

        return Y_preds, Y_pred_vars, Y_pred_vars_w_measures, e2

    def train_surrogate(self, t_init):  # noqa: C901, D102, PLR0915
        self.seed = 43
        np.random.seed(43)

        self.nc1 = min(200 * self.x_dim, 2000)  # candidate points
        self.nq = min(200 * self.x_dim, 2000)  # integration points
        # FEM index
        self.id_sim_hf = 0
        self.id_sim_lf = 0
        self.time_hf_tot = 0
        self.time_lf_tot = 0
        self.time_hf_avg = float('Inf')
        self.time_lf_avg = float('Inf')
        self.time_ratio = 1

        x_dim = self.x_dim  # noqa: F841
        y_dim = self.y_dim

        #
        # Generate initial Samples
        #

        model_hf = self.modelInfoHF
        model_lf = self.modelInfoLF

        self.set_FEM(
            self.rv_name, self.do_parallel, self.y_dim, t_init, model_hf.thr_t
        )

        def FEM_batch_hf(X, id_sim):  # noqa: N802, N803
            # DiscStr: Xstr will be replaced with the string
            Xstr = X.astype(str)  # noqa: N806

            for nx in self.rvDiscIdx:
                for ns in range(X.shape[0]):
                    Xstr[ns][nx] = '"' + self.rvDiscStr[nx][int(X[ns][nx] - 1)] + '"'

            tmp = time.time()
            if model_hf.is_model or model_hf.model_without_sampling:
                res = self.run_FEM_batch(
                    Xstr, id_sim, runIdx=model_hf.runIdx, alterInput=self.rvDiscIdx
                )
            else:
                res = np.zeros((0, self.x_dim)), np.zeros((0, self.y_dim)), id_sim
            self.time_hf_tot += time.time() - tmp
            self.time_hf_avg = (
                np.float64(self.time_hf_tot) / res[2]
            )  # so that it gives inf when divided by zero
            self.time_ratio = self.time_hf_avg / self.time_lf_avg
            return res

        def FEM_batch_lf(X, id_sim):  # noqa: N802, N803
            # DiscStr: Xstr will be replaced with the string
            Xstr = X.astype(str)  # noqa: N806

            for nx in self.rvDiscIdx:
                for ns in range(X.shape[0]):
                    Xstr[ns][nx] = self.rvDiscStr[nx][int(X[ns][nx] - 1)]

            tmp = time.time()
            if model_lf.is_model:
                res = self.run_FEM_batch(
                    Xstr, id_sim, runIdx=model_lf.runIdx, alterInput=self.rvDiscIdx
                )
            else:
                res = np.zeros((0, self.x_dim)), np.zeros((0, self.y_dim)), id_sim
            self.time_lf_tot += time.time() - tmp
            if res[2] > 0:
                self.time_lf_avg = (
                    float(self.time_lf_tot) / res[2]
                )  # so that it gives inf when divided by zero
            else:
                self.time_lf_avg = float('Inf')
            self.time_ratio = self.time_lf_avg / self.time_lf_avg
            return res

        tmp = time.time()  # noqa: F841

        #
        # get initial samples for high fidelity modeling
        #

        X_hf_tmp = model_hf.sampling(max([model_hf.n_init - model_hf.n_existing, 0]))  # noqa: N806

        #
        # if X is from a data file & Y is from simulation
        #

        if model_hf.model_without_sampling:
            X_hf_tmp, model_hf.X_existing = model_hf.X_existing, X_hf_tmp  # noqa: N806
        X_hf_tmp, Y_hf_tmp, self.id_sim_hf = FEM_batch_hf(X_hf_tmp, self.id_sim_hf)  # noqa: N806

        if model_hf.X_existing.shape[0] == 0:
            self.X_hf, self.Y_hf = X_hf_tmp, Y_hf_tmp
        else:
            if model_hf.X_existing.shape[1] != X_hf_tmp.shape[1]:
                msg = f'Error importing input dimension specified {model_hf.X_existing.shape[1]} is different from the written {X_hf_tmp.shape[1]}.'
                self.exit(msg)

            self.X_hf, self.Y_hf = (
                np.vstack([model_hf.X_existing, X_hf_tmp]),
                np.vstack([model_hf.Y_existing, Y_hf_tmp]),
            )

        X_lf_tmp = model_lf.sampling(max([model_lf.n_init - model_lf.n_existing, 0]))  # noqa: N806

        # Design of experiments - Nearest neighbor sampling
        # Giselle FernÃ¡ndez-Godino, M., Park, C., Kim, N. H., & Haftka, R. T. (2019). Issues in deciding whether to use multifidelity surrogates. AIAA Journal, 57(5), 2039-2054.
        self.n_LFHFoverlap = 0
        new_x_lf_tmp = np.zeros((0, self.x_dim))
        X_tmp = X_lf_tmp  # noqa: N806

        for x_hf in self.X_hf:
            if X_tmp.shape[0] > 0:
                id = closest_node(x_hf, X_tmp, self.ll)  # noqa: A001
                new_x_lf_tmp = np.vstack([new_x_lf_tmp, x_hf])
                X_tmp = np.delete(X_tmp, id, axis=0)  # noqa: N806
                self.n_LFHFoverlap += 1

        new_x_lf_tmp = np.vstack([new_x_lf_tmp, X_tmp])
        new_x_lf_tmp, new_y_lf_tmp, self.id_sim_lf = FEM_batch_lf(
            new_x_lf_tmp, self.id_sim_lf
        )

        self.X_lf, self.Y_lf = (
            np.vstack([model_lf.X_existing, new_x_lf_tmp]),
            np.vstack([model_lf.Y_existing, new_y_lf_tmp]),
        )

        if self.X_lf.shape[0] != 0:
            if self.X_hf.shape[1] != self.X_lf.shape[1]:
                msg = f'Error importing input data: dimension inconsistent: high fidelity model have {self.X_hf.shape[1]} RV(s) but low fidelity model have {self.X_lf.shape[1]}.'
                self.exit(msg)

            if self.Y_hf.shape[1] != self.Y_lf.shape[1]:
                msg = f'Error importing input data: dimension inconsistent: high fidelity model have {self.Y_hf.shape[1]} QoI(s) but low fidelity model have {self.Y_lf.shape[1]}.'
                self.exit(msg)

        stoch_idx = []  # noqa: F841
        for i in range(y_dim):
            print('Setting up QoI {} among {}'.format(i + 1, y_dim))  # noqa: T201, UP032
            self.m_list[i] = self.set_XY(
                self.m_list[i],
                i,
                self.X_hf,
                self.Y_hf[:, i][np.newaxis].transpose(),
                self.X_lf,
                self.Y_lf[:, i][np.newaxis].transpose(),
            )  # log-transform is inside set_XY

            # check stochastic ?
            # if self.stochastic[i] and not self.do_mf:
            #     # see if we can run it parallel
            #     X_new, X_idx, indices, counts = np.unique(  # noqa: N806, RUF100
            #         self.X_hf,
            #         axis=0,
            #         return_index=True,
            #         return_counts=True,
            #         return_inverse=True,
            #     )
            #     n_unique = X_new.shape[0]
            #     if n_unique == self.X_hf.shape[0]:  # no repl
            #         stoch_idx += [i]
            #
            # else:
            #     #
            #     #   original calibration
            #     #
            #     print("Setting up QoI {} among {}".format(i+1,y_dim))
            #     self.m_list[i] = self.set_XY(
            #         self.m_list[i],
            #         i,
            #         self.X_hf,
            #         self.Y_hf[:, i][np.newaxis].transpose(),
            #         self.X_lf,
            #         self.Y_lf[:, i][np.newaxis].transpose(),
            #     )  # log-transform is inside set_XY

            # # parllel run
            # if len(stoch_idx)>0:
            #     iterables = (
            #         (
            #             copy.deepcopy(self.m_list[i]),
            #             i,
            #             self.x_dim,
            #             self.X_hf,
            #             self.Y_hf[:, i][np.newaxis].transpose(),
            #             self.create_kernel,
            #             self.create_gpy_model,
            #             self.do_logtransform,
            #             self.predictStoMeans,
            #             self.set_normalizer
            #         )
            #         for i in stoch_idx
            #     )
            #     result_objs = list(self.pool.starmap(set_XY_indi, iterables))
            #     for ny, m_tmp_, Y_mean_, normMeans_, normVars_, m_var_list_, var_str_, indices_unique_, n_unique_hf_ in result_objs:  # noqa: N806, RUF100
            #         self.m_tmp[ny] = m_tmp_
            #         self.Y_mean[ny] = Y_mean_
            #         self.normMeans[ny] = normMeans_
            #         self.normVars[ny] = normVars_
            #         self.m_var_list[ny] = m_var_list_
            #         self.var_str[ny] = var_str_
            #         self.indices_unique = indices_unique_
            #         self.n_unique_hf[ny] = n_unique_hf_

        #
        # Verification measures
        #

        self.NRMSE_hist = np.zeros((1, y_dim), float)
        self.NRMSE_idx = np.zeros((1, 1), int)

        print('======== RUNNING GP DoE ===========', flush=True)  # noqa: T201

        #
        # Run Design of experiments
        #

        exit_flag = False
        nc1 = self.nc1
        nq = self.nq
        n_new = 0
        while exit_flag == False:  # noqa: E712
            # Initial calibration

            # Calibrate self.m_list
            self.Y_cvs, self.Y_cv_vars, self.Y_cv_var_w_measures, e2 = (
                self.calibrate()
            )

            if self.do_logtransform:
                # self.Y_cv = np.exp(2*self.Y_cvs+self.Y_cv_vars)*(np.exp(self.Y_cv_vars)-1) # in linear space
                # TODO: Let us use median instead of mean?  # noqa: TD002
                self.Y_cv = np.exp(self.Y_cvs)
                self.Y_cv_var = np.exp(2 * self.Y_cvs + self.Y_cv_vars) * (
                    np.exp(self.Y_cv_vars) - 1
                )  # in linear space

                self.Y_cv_var_w_measure = np.exp(
                    2 * self.Y_cvs + self.Y_cv_var_w_measures
                ) * (np.exp(self.Y_cv_var_w_measures) - 1)  # in linear space
            else:
                self.Y_cv = self.Y_cvs
                self.Y_cv_var = self.Y_cv_vars
                self.Y_cv_var_w_measure = self.Y_cv_var_w_measures

            if self.n_unique_hf < model_hf.thr_count:
                if self.doeIdx == 'HF':
                    tmp_doeIdx = self.doeIdx  # single fideility  # noqa: N806
                else:
                    tmp_doeIdx = 'HFHF'  # HF in multifideility  # noqa: N806

                [x_new_hf, y_idx_hf, score_hf] = self.run_design_of_experiments(
                    nc1, nq, e2, tmp_doeIdx
                )
            else:
                score_hf = 0

            if self.id_sim_lf < model_lf.thr_count:
                [x_new_lf, y_idx_lf, score_lf] = self.run_design_of_experiments(
                    nc1, nq, e2, 'LF'
                )
            else:
                score_lf = 0  # score : reduced amount of variance

            if self.doeIdx == 'HFLF':
                fideilityIdx = np.argmax(  # noqa: N806
                    [score_hf / self.time_hf_avg, score_lf / self.time_lf_avg]
                )
                if fideilityIdx == 0:
                    tmp_doeIdx = 'HF'  # noqa: N806
                else:
                    tmp_doeIdx = 'LF'  # noqa: N806
            else:
                tmp_doeIdx = self.doeIdx  # noqa: N806

            if self.do_logtransform:
                Y_hfs = np.log(self.Y_hf)  # noqa: N806
            else:
                Y_hfs = self.Y_hf  # noqa: N806

            NRMSE_val = self.normalized_mean_sq_error(self.Y_cvs, Y_hfs)  # noqa: N806
            self.NRMSE_hist = np.vstack((self.NRMSE_hist, np.array(NRMSE_val)))
            self.NRMSE_idx = np.vstack((self.NRMSE_idx, i))

            if (
                self.n_unique_hf
                >= model_hf.thr_count  # self.id_sim_hf >= model_hf.thr_count
                and self.id_sim_lf >= model_lf.thr_count
            ):
                n_iter = i
                self.exit_code = 'count'
                if self.id_sim_hf == 0 and self.id_sim_lf == 0:
                    self.exit_code = 'data'
                exit_flag = True
                break

            if (
                self.X_hf.shape[0] == model_hf.thr_count
                and np.sum(self.stochastic) == 0
            ):
                # This is when replicated unwantedly
                n_iter = i
                self.exit_code = 'count'
                exit_flag = True
                break

            if np.max(NRMSE_val) < model_hf.thr_NRMSE:
                n_iter = i
                self.exit_code = 'accuracy'
                exit_flag = True
                break

            if time.time() - t_init > model_hf.thr_t - self.calib_time:
                n_iter = i  # noqa: F841
                self.exit_code = 'time'
                doe_off = True  # noqa: F841
                break

            if tmp_doeIdx.startswith('HF'):
                n_new = x_new_hf.shape[0]
                if n_new + self.n_unique_hf > model_hf.thr_count:
                    n_new = model_hf.thr_count - self.n_unique_hf
                    x_new_hf = x_new_hf[0:n_new, :]
                x_hf_new, y_hf_new, self.id_sim_hf = FEM_batch_hf(
                    x_new_hf, self.id_sim_hf
                )
                self.X_hf = np.vstack([self.X_hf, x_hf_new])
                self.Y_hf = np.vstack([self.Y_hf, y_hf_new])
                i = self.n_unique_hf + n_new

            if tmp_doeIdx.startswith('LF'):
                n_new = x_new_lf.shape[0]
                if n_new + self.id_sim_lf > model_lf.thr_count:
                    n_new = model_lf.thr_count - self.id_sim_lf
                    x_new_lf = x_new_lf[0:n_new, :]
                x_lf_new, y_lf_new, self.id_sim_lf = FEM_batch_lf(
                    x_new_lf, self.id_sim_lf
                )
                self.X_lf = np.vstack([self.X_lf, x_lf_new])
                self.Y_lf = np.vstack([self.Y_lf, y_lf_new])
                i = self.id_sim_lf + n_new
                # TODO  # noqa: TD002, TD004

            # print(">> {:.2f} s".format(time.time() - t_init))

            for ny in range(self.y_dim):
                self.m_list[ny] = self.set_XY(
                    self.m_list[ny],
                    ny,
                    self.X_hf,
                    self.Y_hf[:, ny][np.newaxis].transpose(),
                    self.X_lf,
                    self.Y_lf[:, ny][np.newaxis].transpose(),
                )  # log-transform is inside set_XY

        self.sim_time = time.time() - t_init
        self.NRMSE_val = NRMSE_val

        self.verify()
        self.verify_nugget()
        print(f'my exit code = {self.exit_code}', flush=True)  # noqa: T201
        print(f'1. count = {self.id_sim_hf}', flush=True)  # noqa: T201
        print(f'1. count_unique = {self.n_unique_hf}', flush=True)  # noqa: T201
        print(f'2. max(NRMSE) = {np.max(self.NRMSE_val)}', flush=True)  # noqa: T201
        print(f'3. time = {self.sim_time:.2f} s', flush=True)  # noqa: T201

        if develop_mode:
            print('CV: inbound50')  # noqa: T201
            print(self.inbound50)  # noqa: T201
            print('CV: max coverage')  # noqa: T201
            print(self.max_coverage)  # noqa: T201
            print('CV: quantile values')  # noqa: T201
            print(self.quantile_reconst_list)  # noqa: T201

            ny = 0
            nx = 0
            sorted_y_std = np.sqrt(self.Y_cv_var_w_measure[:, ny])
            sorted_y_std0 = np.sqrt(self.Y_cv_var[:, ny])

            plt.errorbar(
                self.X_hf[:, nx],
                (self.Y_cv[:, ny]),
                yerr=sorted_y_std,
                fmt='x',
                alpha=50 / self.X_hf.shape[0],
            )
            plt.errorbar(
                self.X_hf[:, nx],
                (self.Y_cv[:, ny]),
                yerr=sorted_y_std0,
                fmt='x',
                alpha=50 / self.X_hf.shape[0],
            )
            plt.scatter(self.X_hf[:, nx], (self.Y_hf[:, ny]), c='r', alpha=0.1)
            plt.title('RV={}, QoI={}'.format(nx + 1, ny + 1))  # noqa: UP032
            plt.show()

            log_Y_cv_sample = np.random.normal(  # noqa: F841, N806
                self.Y_cvs[:, ny], np.sqrt(self.Y_cv_var_w_measures[:, ny])
            )

            ny = 0
            plt.scatter(self.Y_hf[:, ny], self.Y_cv[:, ny], alpha=0.1)
            plt.errorbar(
                self.Y_hf[:, ny],
                self.Y_cv[:, ny],
                yerr=sorted_y_std,
                fmt='x',
                alpha=0.1,
            )
            plt.scatter(self.Y_hf[:, ny], self.Y_hf[:, ny], alpha=0.1)
            plt.title('QoI = {}'.format(ny + 1))  # noqa: UP032
            plt.show()

            plt.scatter(
                np.log10(self.Y_hf[:, ny]),
                np.log10((self.Y_cv[:, ny])),  # noqa: UP034
                alpha=1,
                marker='x',
            )
            plt.plot(
                np.log10(self.Y_hf[:, ny]),
                np.log10(self.Y_hf[:, ny]),
                alpha=1,
                color='r',
            )
            mycor_CV = np.corrcoef(self.Y_hf[:, nx], self.Y_cv[:, ny])[1, 0]  # noqa: N806
            mycor_log_CV = np.corrcoef(  # noqa: N806
                np.log(self.Y_hf[:, nx]), np.log(self.Y_cv[:, ny])
            )[1, 0]
            plt.title(
                f'train CV rho={round(mycor_CV*100)/100} rho_log={round(mycor_log_CV*100)/100}'
            )
            plt.xlabel('QoI exact')
            plt.ylabel('QoI pred median')
            plt.grid()
            plt.show()

            [a, b] = self.m_list[0].predict(self.m_list[0].X)

            if sum(self.linear_list) > 0:
                a = (
                    a
                    + self.lin_list[ny].predict(self.X_hf[:, self.linear_list])[
                        :, 0:1
                    ]
                )
            # Don't use b

            plt.scatter(
                np.log10(self.Y_hf[:, ny]),
                np.log10(np.exp(a[:, ny])),
                alpha=1,
                marker='x',
            )
            plt.plot(
                np.log10(self.Y_hf[:, ny]),
                np.log10(self.Y_hf[:, ny]),
                alpha=1,
                color='r',
            )
            mycor = np.corrcoef(self.Y_hf[:, ny], np.exp(a[:, ny]))[1, 0]
            mycor_log = np.corrcoef(np.log(self.Y_hf[:, ny]), a[:, ny])[1, 0]
            plt.title(
                f'train rho={round(mycor*100)/100} rho_log={round(mycor_log*100)/100}'
            )
            plt.xlabel('QoI exact')
            plt.ylabel('QoI pred median')
            plt.grid()
            plt.show()

            plt.scatter((self.Y_hf[:, ny]), (np.exp(a[:, ny])), alpha=1, marker='x')
            plt.plot((self.Y_hf[:, ny]), (self.Y_hf[:, ny]), alpha=1, color='r')
            mycor = np.corrcoef(self.Y_hf[:, ny], np.exp(a[:, ny]))[1, 0]
            mycor_log = np.corrcoef(np.log(self.Y_hf[:, ny]), a[:, ny])[1, 0]
            plt.title(
                f'train rho={round(mycor*100)/100} rho_log={round(mycor_log*100)/100}'
            )
            plt.xlabel('QoI exact')
            plt.ylabel('QoI pred median')
            plt.grid()
            plt.show()

            plt.scatter((self.X_hf[:, nx]), (self.Y_hf[:, ny]), alpha=1, color='r')
            plt.scatter((self.X_hf[:, nx]), (np.exp(a[:, ny])), alpha=1, marker='x')
            plt.xlabel('X')
            plt.ylabel('QoI pred median')
            plt.grid()
            plt.show()

            # plt.scatter(np.log10(self.Y_hf[:, ny]),np.log10(np.exp(log_Y_cv_sample)), alpha=1,marker='x');
            # plt.plot(np.log10(self.Y_hf[:, ny]),np.log10(self.Y_hf[:, ny]),alpha=1,color='r');
            # mycor = np.corrcoef(self.Y_hf[:, ny], np.exp(log_Y_cv_sample))[1,0]
            # plt.title(f"train CV samples rho={round(mycor*100)/100}")
            # plt.xlabel("QoI exact");
            # plt.ylabel("QoI pred sample");
            # plt.grid()
            # plt.show()

            X_test = np.genfromtxt(  # noqa: N806
                r'C:\Users\SimCenter\Dropbox\SimCenterPC\Stochastic_GP_validation\input_test.csv',
                delimiter=',',
            )
            Y_test = np.genfromtxt(  # noqa: N806
                r'C:\Users\SimCenter\Dropbox\SimCenterPC\Stochastic_GP_validation\output_test.csv',
                delimiter=',',
            )
            [a, b] = self.m_list[0].predict(X_test)

            if sum(self.linear_list) > 0:
                a = (
                    a
                    + self.lin_list[ny].predict(X_test[:, self.linear_list])[:, 0:1]
                )
            # Don't use b
            plt.scatter(
                np.log10(Y_test), np.log10(np.exp(a[:, ny])), alpha=1, marker='x'
            )
            plt.plot(np.log10(Y_test), np.log10(Y_test), alpha=1, color='r')
            mycor_Test = np.corrcoef(Y_test, np.exp(a[:, ny]))[1, 0]  # noqa: N806
            mycor_log_Test = np.corrcoef(np.log(Y_test), a[:, ny])[1, 0]  # noqa: N806
            plt.title(
                f'test rho={round(mycor_Test*100)/100} rho_log={round(mycor_log_Test*100)/100}'
            )
            plt.xlabel('QoI exact')
            plt.ylabel('QoI pred median')
            plt.grid()
            plt.show()
            #
            # Predict variance
            #

            log_var_pred, dum = self.m_var_list[0].predict(X_test)
            log_Y_var_pred_w_measure = (  # noqa: N806
                b + np.exp(log_var_pred) * self.m_list[ny].Gaussian_noise.parameters
            )

            qualtile_vals = np.arange(0.1, 1, 0.1)
            qualtile_reconst = np.zeros([len(qualtile_vals)])
            for nqu in range(len(qualtile_vals)):
                Q_b = norm.ppf(  # noqa: N806
                    qualtile_vals[nqu],
                    loc=a,
                    scale=np.sqrt(log_Y_var_pred_w_measure),
                )
                qualtile_reconst[nqu] = (
                    np.sum((np.log(Y_test) < Q_b[:, 0])) / Y_test.shape[0]  # noqa: UP034
                )

            quant_err = abs(qualtile_reconst - qualtile_vals)
            print(f'Test: max coverage err: {np.max(quant_err)}')  # noqa: T201
            print(f'Test: mean coverage err: {np.mean(quant_err)}')  # noqa: T201
            print('Test: quantile range')  # noqa: T201
            print(qualtile_reconst)  # noqa: T201
            print(f'Corr(log) for CV: {round(mycor_log_CV*100)/100}')  # noqa: T201
            print(f'Corr(log) for Test: {round(mycor_log_Test*100)/100}')  # noqa: T201
            print('')  # noqa: T201, FURB105

    def verify(self):  # noqa: D102
        Y_cv = self.Y_cv  # noqa: N806
        Y = self.Y_hf  # noqa: N806
        model_hf = self.modelInfoHF

        if model_hf.is_model:
            n_err = 1000
            Xerr = model_hf.resampling(self.m_list[0].X, n_err)  # noqa: N806

            y_pred_var = np.zeros((n_err, self.y_dim))
            y_data_var = np.zeros((n_err, self.y_dim))
            y_pred_mean = np.zeros((n_err, self.y_dim))
            y_base_var = np.zeros((self.y_dim,))
            for ny in range(self.y_dim):
                m_tmp = self.m_list[ny]
                # y_data_var[:, ny] = np.var(Y[:, ny])
                y_data_var[:, ny] = np.var(self.m_list[ny].Y)
                # if self.do_logtransform:
                #     log_mean = np.mean(np.log(Y[:, ny]))
                #     log_var = np.var(np.log(Y[:, ny]))
                #     y_var_vals = np.exp(2*log_mean+log_var)*(np.exp(log_var)-1) # in linear space
                # else:
                #     y_var_vals = np.var(Y[:, ny])

                for ns in range(n_err):
                    y_preds, y_pred_vars = self.predict(
                        m_tmp, Xerr[ns, :][np.newaxis]
                    )
                    y_pred_var[ns, ny] = y_pred_vars
                    y_pred_mean[ns, ny] = y_preds

                # dummy, y_base_var[ny] = self.predict(m_tmp, Xerr[ns, :][np.newaxis]*10000)
                dummy, y_base_var[ny] = self.predict(
                    m_tmp, Xerr[ns, :][np.newaxis] * 10000
                )

                # if self.do_logtransform:
                #    y_pred_var[ns, ny] = np.exp(2 * y_preds + y_pred_vars) * (
                #            np.exp(y_pred_vars) - 1
                #    )
                # else:
                #    y_pred_var[ns, ny] = y_pred_vars

            error_ratio2_Pr = y_pred_var / y_data_var  # noqa: N806
            # print(np.max(error_ratio2_Pr, axis=0), flush=True)  # noqa: RUF100, T201

            perc_thr_tmp = np.hstack(
                [np.array([1]), np.arange(10, 1000, 50), np.array([999])]
            )
            error_sorted = np.sort(np.max(error_ratio2_Pr, axis=1), axis=0)
            self.perc_val = error_sorted[perc_thr_tmp]  # criteria
            self.perc_thr = 1 - (perc_thr_tmp) * 0.001  # ratio=simulation/sampling

            self.perc_thr = self.perc_thr.tolist()
            self.perc_val = self.perc_val.tolist()

        else:
            self.perc_thr = 0
            self.perc_val = 0

        corr_val = np.zeros((self.y_dim,))
        R2_val = np.zeros((self.y_dim,))  # noqa: N806
        for ny in range(self.y_dim):
            corr_val[ny] = np.corrcoef(Y[:, ny], Y_cv[:, ny])[0, 1]
            R2_val[ny] = 1 - np.sum(pow(Y_cv[:, ny] - Y[:, ny], 2)) / np.sum(
                pow(Y_cv[:, ny] - np.mean(Y_cv[:, ny]), 2)
            )
            if np.var(Y[:, ny]) == 0:
                corr_val[ny] = 1
                R2_val[ny] = 0

        self.corr_val = corr_val
        self.R2_val = R2_val

    def verify_nugget(self):  # noqa: D102
        Y_cv = self.Y_cv  # noqa: N806
        Y_cv_var_w_measure = self.Y_cv_var_w_measure  # noqa: N806
        Y = self.Y_hf  # noqa: N806
        model_hf = self.modelInfoHF  # noqa: F841

        self.quantile_reconst_list = np.zeros((self.y_dim, 9))
        self.max_coverage = np.zeros((self.y_dim,))
        self.mean_coverage = np.zeros((self.y_dim,))
        self.inbound50 = np.zeros((self.y_dim,))
        self.Gausspvalue = np.zeros((self.y_dim,))

        if not self.do_mf:
            for ny in range(self.y_dim):
                if not self.do_logtransform:
                    #
                    # Interquarltile range
                    #

                    PI_lb = norm.ppf(  # noqa: N806
                        0.25,
                        loc=Y_cv[:, ny],
                        scale=np.sqrt(Y_cv_var_w_measure[:, ny]),
                    )
                    PI_ub = norm.ppf(  # noqa: N806
                        0.75,
                        loc=Y_cv[:, ny],
                        scale=np.sqrt(Y_cv_var_w_measure[:, ny]),
                    )
                    num_in_bound = np.sum((Y[:, ny] > PI_lb) * (Y[:, ny] < PI_ub))

                    #
                    # coverage range
                    #

                    qualtile_vals = np.arange(0.1, 1, 0.1)
                    qualtile_reconst = np.zeros([len(qualtile_vals)])
                    for nqu in range(len(qualtile_vals)):
                        Q_b = np.squeeze(  # noqa: N806
                            norm.ppf(
                                qualtile_vals[nqu],
                                loc=Y_cv[:, ny],
                                scale=np.sqrt(Y_cv_var_w_measure[:, ny]),
                            )
                        )

                        # print(Y[:, ny])
                        # print(Q_b)
                        qualtile_reconst[nqu] = np.sum(Y[:, ny] < Q_b) / Y.shape[0]

                    quant_err = abs(qualtile_reconst - qualtile_vals)

                    norm_residual = (Y[:, ny] - Y_cv[:, ny]) / np.sqrt(
                        Y_cv_var_w_measure[:, ny]
                    )
                    stats = cramervonmises(norm_residual, 'norm')

                else:
                    # mu = np.log(Y_cv[:, ny])
                    # sigm = np.sqrt(
                    #     np.log(Y_cv_var_w_measure[:, ny] / pow(Y_cv[:, ny], 2) + 1)
                    # )
                    log_Y_cv = self.Y_cvs[:, ny]  # noqa: N806
                    log_Y_cv_var_w_measure = self.Y_cv_var_w_measures[:, ny]  # noqa: N806

                    #
                    # Interquarltile range
                    #

                    PI_lb = norm.ppf(  # noqa: N806
                        0.25, loc=log_Y_cv, scale=np.sqrt(log_Y_cv_var_w_measure)
                    ).tolist()
                    PI_ub = norm.ppf(  # noqa: N806
                        0.75, loc=log_Y_cv, scale=np.sqrt(log_Y_cv_var_w_measure)
                    ).tolist()
                    num_in_bound = np.sum(
                        (np.log(Y[:, ny]) > PI_lb) * (np.log(Y[:, ny]) < PI_ub)
                    )

                    #
                    # coverage range
                    #

                    qualtile_vals = np.arange(0.1, 1, 0.1)
                    qualtile_reconst = np.zeros([len(qualtile_vals)])
                    for nqu in range(len(qualtile_vals)):
                        Q_b = norm.ppf(  # noqa: N806
                            qualtile_vals[nqu],
                            loc=log_Y_cv,
                            scale=np.sqrt(log_Y_cv_var_w_measure),
                        ).tolist()
                        qualtile_reconst[nqu] = (
                            np.sum((np.log(Y[:, ny]) < Q_b)) / Y.shape[0]  # noqa: UP034
                        )

                    quant_err = abs(qualtile_reconst - qualtile_vals)

                    #
                    # cramervonmises
                    #

                    norm_residual = (np.log(Y[:, ny]) - log_Y_cv) / np.sqrt(
                        log_Y_cv_var_w_measure
                    )
                    stats = cramervonmises(norm_residual, 'norm')

                self.quantile_reconst_list[ny, :] = qualtile_reconst
                self.max_coverage[ny] = np.max(quant_err)
                self.mean_coverage[ny] = np.mean(quant_err)
                self.inbound50[ny] = num_in_bound / Y.shape[0]
                self.Gausspvalue[ny] = stats.pvalue

        else:
            pass

    def save_model(self, filename):  # noqa: C901, D102, PLR0915
        if self.isEEUQ:
            self.rv_name_new = []
            for nx in range(self.x_dim):
                if self.modelInfoHF.xDistTypeArr[nx] == 'U':
                    self.rv_name_new += [self.rv_name[nx]]

            if len(self.IM_names) > 0:
                self.rv_name_new += self.IM_names

            self.rv_name = self.rv_name_new
            self.x_dim = len(self.rv_name_new)

        if self.do_mf:
            with open(self.work_dir + '/' + filename + '.pkl', 'wb') as file:  # noqa: PTH123
                pickle.dump(self.m_list, file)

        header_string_x = ' ' + ' '.join([str(elem) for elem in self.rv_name]) + ' '
        header_string_y = ' ' + ' '.join([str(elem) for elem in self.g_name])
        header_string = header_string_x + header_string_y

        xy_data = np.concatenate(
            (
                np.asmatrix(np.arange(1, self.X_hf.shape[0] + 1)).T,
                self.X_hf,
                self.Y_hf,
            ),
            axis=1,
        )
        xy_data = xy_data.astype(float)
        self.X_hf = self.X_hf.astype(float)
        self.Y_hf = self.Y_hf.astype(float)

        np.savetxt(
            self.work_dir + '/dakotaTab.out',
            xy_data,
            header=header_string,
            fmt='%1.4e',
            comments='%',
        )
        np.savetxt(
            self.work_dir + '/inputTab.out',
            self.X_hf,
            header=header_string_x,
            fmt='%1.4e',
            comments='%',
        )
        np.savetxt(
            self.work_dir + '/outputTab.out',
            self.Y_hf,
            header=header_string_y,
            fmt='%1.4e',
            comments='%',
        )

        y_ub = np.zeros(self.Y_cv.shape)
        y_lb = np.zeros(self.Y_cv.shape)
        y_ubm = np.zeros(self.Y_cv.shape)  # with measruement
        y_lbm = np.zeros(self.Y_cv.shape)

        if not self.do_logtransform:
            for ny in range(self.y_dim):
                y_lb[:, ny] = norm.ppf(
                    0.05, loc=self.Y_cv[:, ny], scale=np.sqrt(self.Y_cv_var[:, ny])
                ).tolist()
                y_ub[:, ny] = norm.ppf(
                    0.95, loc=self.Y_cv[:, ny], scale=np.sqrt(self.Y_cv_var[:, ny])
                ).tolist()
                y_lbm[:, ny] = norm.ppf(
                    0.05,
                    loc=self.Y_cv[:, ny],
                    scale=np.sqrt(self.Y_cv_var_w_measure[:, ny]),
                ).tolist()
                y_ubm[:, ny] = norm.ppf(
                    0.95,
                    loc=self.Y_cv[:, ny],
                    scale=np.sqrt(self.Y_cv_var_w_measure[:, ny]),
                ).tolist()
        else:
            for ny in range(self.y_dim):
                mu = np.log(self.Y_cv[:, ny])
                sig = np.sqrt(
                    np.log(self.Y_cv_var[:, ny] / pow(self.Y_cv[:, ny], 2) + 1)
                )
                y_lb[:, ny] = lognorm.ppf(0.05, s=sig, scale=np.exp(mu)).tolist()
                y_ub[:, ny] = lognorm.ppf(0.95, s=sig, scale=np.exp(mu)).tolist()

                sig_m = np.sqrt(
                    np.log(
                        self.Y_cv_var_w_measure[:, ny] / pow(self.Y_cv[:, ny], 2) + 1
                    )
                )
                y_lbm[:, ny] = lognorm.ppf(0.05, s=sig_m, scale=np.exp(mu)).tolist()
                y_ubm[:, ny] = lognorm.ppf(0.95, s=sig_m, scale=np.exp(mu)).tolist()

        xy_sur_data = np.hstack(
            (
                xy_data,
                self.Y_cv,
                y_lb,
                y_ub,
                self.Y_cv_var,
                y_lbm,
                y_ubm,
                self.Y_cv_var_w_measure,
            )
        )
        g_name_sur = self.g_name
        header_string_sur = (
            header_string
            + ' '
            + '.median '.join(g_name_sur)
            + '.median '
            + '.q5 '.join(g_name_sur)
            + '.q5 '
            + '.q95 '.join(g_name_sur)
            + '.q95 '
            + '.var '.join(g_name_sur)
            + '.var '
            + '.q5_w_mnoise '.join(g_name_sur)
            + '.q5_w_mnoise '
            + '.q95_w_mnoise '.join(g_name_sur)
            + '.q95_w_mnoise '
            + '.var_w_mnoise '.join(g_name_sur)
            + '.var_w_mnoise '
        )

        np.savetxt(
            self.work_dir + '/surrogateTab.out',
            xy_sur_data,
            header=header_string_sur,
            fmt='%1.4e',
            comments='%',
        )

        #
        # Save surrogateinfo
        #
        results = {}

        hfJson = {}  # noqa: N806
        hfJson['doSampling'] = self.modelInfoHF.is_model
        hfJson['doSimulation'] = self.modelInfoHF.is_model
        hfJson['DoEmethod'] = self.modelInfoHF.doe_method
        hfJson['thrNRMSE'] = self.modelInfoHF.thr_NRMSE
        hfJson['valSamp'] = self.modelInfoHF.n_existing + self.id_sim_hf
        hfJson['valSampUnique'] = self.n_unique_hf
        hfJson['valSim'] = self.id_sim_hf

        constIdx = []  # noqa: N806
        constVal = []  # noqa: N806
        for ny in range(self.y_dim):
            if np.var(self.Y_hf[:, ny]) == 0:
                constIdx += [ny]  # noqa: N806
                constVal += [np.mean(self.Y_hf[:, ny])]  # noqa: N806

        hfJson['constIdx'] = constIdx
        hfJson['constVal'] = constVal

        results['inpData'] = self.modelInfoHF.inpData
        results['outData'] = self.modelInfoHF.outData
        results['valSamp'] = self.X_hf.shape[0]
        results['doStochastic'] = self.stochastic
        results['doNormalization'] = self.set_normalizer
        results['isEEUQ'] = self.isEEUQ
        results['isWEUQ'] = self.isWEUQ

        if self.isEEUQ:
            if len(self.IM_names) > 0:
                IM_sub_Json = {}  # noqa: N806
                IM_sub_Json['IntensityMeasure'] = self.intensityMeasure
                IM_sub_Json['GeneralInformation'] = {'units': self.unitInfo}
                IM_sub_Json['Events'] = {}

                results['intensityMeasureInfo'] = IM_sub_Json

        results['highFidelityInfo'] = hfJson

        lfJson = {}  # noqa: N806
        if self.do_mf:
            lfJson['doSampling'] = self.modelInfoLF.is_data
            lfJson['doSimulation'] = self.modelInfoLF.is_model
            lfJson['DoEmethod'] = self.modelInfoLF.doe_method
            lfJson['thrNRMSE'] = self.modelInfoLF.thr_NRMSE
            lfJson['valSamp'] = self.modelInfoLF.n_existing + self.id_sim_lf
            lfJson['valSim'] = self.id_sim_lf
            results['inpData'] = self.modelInfoLF.inpData
            results['outData'] = self.modelInfoLF.outData
            results['valSamp'] = self.X_lf.shape[0]

            results['lowFidelityInfo'] = lfJson

        else:
            results['lowFidelityInfo'] = 'None'

        results['doLogtransform'] = self.do_logtransform
        results['doLinear'] = self.do_linear
        results['doMultiFidelity'] = self.do_mf
        results['kernName'] = self.kernel
        results['terminationCode'] = self.exit_code
        results['valTime'] = self.sim_time
        results['xdim'] = self.x_dim
        results['ydim'] = self.y_dim
        results['xlabels'] = self.rv_name
        results['ylabels'] = self.g_name
        results['yExact'] = {}
        results['yPredict'] = {}
        results['valNRMSE'] = {}
        results['valR2'] = {}
        results['valCorrCoeff'] = {}
        results['valIQratio'] = {}
        results['valPval'] = {}
        results['valCoverageVals'] = {}
        results['yPredict_PI_lb'] = {}
        results['yPredict_PI_ub'] = {}
        results['xExact'] = {}
        results['valNugget'] = {}
        results['valNugget1'] = {}
        results['valNugget2'] = {}

        for nx in range(self.x_dim):
            results['xExact'][self.rv_name[nx]] = self.X_hf[:, nx].tolist()

        for ny in range(self.y_dim):
            results['yExact'][self.g_name[ny]] = self.Y_hf[:, ny].tolist()
            results['yPredict'][self.g_name[ny]] = self.Y_cv[:, ny].tolist()

            if not self.do_logtransform:
                results['yPredict_PI_lb'][self.g_name[ny]] = norm.ppf(
                    0.25,
                    loc=self.Y_cv[:, ny],
                    scale=np.sqrt(self.Y_cv_var_w_measure[:, ny]),
                ).tolist()
                results['yPredict_PI_ub'][self.g_name[ny]] = norm.ppf(
                    0.75,
                    loc=self.Y_cv[:, ny],
                    scale=np.sqrt(self.Y_cv_var_w_measure[:, ny]),
                ).tolist()
            else:
                mu = np.log(self.Y_cv[:, ny])
                sigm = np.sqrt(
                    np.log(
                        self.Y_cv_var_w_measure[:, ny] / pow(self.Y_cv[:, ny], 2) + 1
                    )
                )
                results['yPredict_PI_lb'][self.g_name[ny]] = lognorm.ppf(
                    0.25, s=sigm, scale=np.exp(mu)
                ).tolist()
                results['yPredict_PI_ub'][self.g_name[ny]] = lognorm.ppf(
                    0.75, s=sigm, scale=np.exp(mu)
                ).tolist()

            # if self.do_logtransform:
            #         log_mean = 0
            #         log_var = float(self.m_list[ny]['Gaussian_noise.variance']) # nugget in log-space
            #         nuggetVal_linear = np.exp(2*log_mean+log_var)*(np.exp(log_var)-1) # in linear space

            if self.do_mf:
                results['valNugget1'][self.g_name[ny]] = float(
                    self.m_list[ny].gpy_model['mixed_noise.Gaussian_noise.variance']
                    * self.normVars[ny]
                )
                results['valNugget2'][self.g_name[ny]] = float(
                    self.m_list[ny].gpy_model[
                        'mixed_noise.Gaussian_noise_1.variance'
                    ]
                    * self.normVars[ny]
                )
            elif not self.heteroscedastic:
                results['valNugget'][self.g_name[ny]] = float(
                    self.m_list[ny]['Gaussian_noise.variance'] * self.normVars[ny]
                )

            results['valNRMSE'][self.g_name[ny]] = self.NRMSE_val[ny]
            results['valR2'][self.g_name[ny]] = self.R2_val[ny]
            results['valCorrCoeff'][self.g_name[ny]] = self.corr_val[ny]
            results['valIQratio'][self.g_name[ny]] = self.inbound50[ny]
            results['valPval'][self.g_name[ny]] = self.Gausspvalue[ny]
            results['valCoverageVals'][self.g_name[ny]] = self.quantile_reconst_list[
                ny
            ].tolist()

            if np.isnan(self.NRMSE_val[ny]) or np.isinf(self.NRMSE_val[ny]):
                results['valNRMSE'][self.g_name[ny]] = 'null'
            if np.isnan(self.R2_val[ny]) or np.isinf(self.R2_val[ny]):
                results['valR2'][self.g_name[ny]] = 'null'
            if np.isnan(self.corr_val[ny]) or np.isinf(self.corr_val[ny]):
                results['valCorrCoeff'][self.g_name[ny]] = 'null'

        results['predError'] = {}
        results['predError']['percent'] = self.perc_thr
        results['predError']['value'] = self.perc_val
        # results["fem"] = self.femInfo

        rv_list = []
        for nx in range(len(self.rvName)):
            rvs = {}
            rvs['name'] = self.rvName[nx]
            rvs['distribution'] = self.rvDist[nx]
            rvs['value'] = self.rvVal[nx]
            rv_list = rv_list + [rvs]  # noqa: RUF005
        results['randomVariables'] = rv_list

        # Used for surrogate
        results['modelInfo'] = {}

        for ny in range(self.y_dim):
            #
            # Save the variance model
            #
            if self.stochastic[ny]:
                results['modelInfo'][self.g_name[ny] + '_Var'] = {}
                for parname in self.m_var_list[ny].parameter_names():
                    results['modelInfo'][self.g_name[ny] + '_Var'][parname] = list(
                        eval('self.m_var_list[ny].' + parname)  # noqa: S307
                    )
                results['modelInfo'][self.g_name[ny] + '_Var'][
                    'TrainingSamplesY'
                ] = self.m_var_list[ny].Y.flatten().tolist()
            else:
                results['modelInfo'][self.g_name[ny] + '_Var'] = 0
            #
            # Save the main model
            #
            if not self.do_mf:
                results['modelInfo'][self.g_name[ny]] = {}
                for parname in self.m_list[ny].parameter_names():
                    results['modelInfo'][self.g_name[ny]][parname] = list(
                        eval('self.m_list[ny].' + parname)  # noqa: S307
                    )

            #
            # Save the linear
            #
            if self.do_linear > 0:
                results['modelInfo'][self.g_name[ny] + '_Lin'] = {}
                results['modelInfo'][self.g_name[ny] + '_Lin'][
                    'predictorList'
                ] = []  # TBA
                results['modelInfo'][self.g_name[ny] + '_Lin']['coef'] = np.squeeze(
                    self.lin_list[ny].coef_
                ).tolist()
                results['modelInfo'][self.g_name[ny] + '_Lin']['intercept'] = float(
                    self.lin_list[ny].intercept_
                )

        if self.isEEUQ or self.isWEUQ:
            # read SAM.json
            SAMpath = self.work_dir + '/templatedir/SAM.json'  # noqa: N806
            try:
                with open(SAMpath, encoding='utf-8') as f:  # noqa: PTH123
                    SAMjson = json.load(f)  # noqa: N806
            except Exception:  # noqa: BLE001
                with open(SAMpath + '.sc', encoding='utf-8') as f:  # noqa: PTH123
                    SAMjson = json.load(f)  # noqa: N806

            EDPpath = self.work_dir + '/templatedir/EDP.json'  # noqa: N806
            with open(EDPpath, encoding='utf-8') as f:  # noqa: PTH123
                EDPjson = json.load(f)  # noqa: N806
            results['SAM'] = SAMjson
            results['EDP'] = EDPjson

        with open(self.work_dir + '/dakota.out', 'w', encoding='utf-8') as fp:  # noqa: PTH123
            json.dump(results, fp, indent=1)

        with open(self.work_dir + '/GPresults.out', 'w') as file:  # noqa: PTH123
            file.write('* Problem setting\n')
            file.write(f'  - dimension of x : {self.x_dim}\n')
            file.write(f'  - dimension of y : {self.y_dim}\n')
            if self.doe_method:
                file.write(f'  - design of experiments : {self.doe_method} \n')

            # if not self.do_doe:
            #     if self.do_simulation and self.do_sampling:
            #         file.write(
            #             "  - design of experiments (DoE) turned off - DoE evaluation time exceeds the model simulation time \n")
            file.write('\n')

            file.write('* High-fidelity model\n')
            # file.write("  - sampling : {}\n".format(self.modelInfoHF.is_model))
            file.write(f'  - simulation : {self.modelInfoHF.is_model}\n')
            file.write('\n')

            if self.do_mf:
                file.write('* Low-fidelity model\n')
                # file.write("  - sampling : {}\n".format(self.modelInfoLF.is_model))
                file.write(f'  - simulation : {self.modelInfoLF.is_model}\n')
                file.write('\n')

            file.write('* Convergence\n')
            file.write(f'  - exit code : "{self.exit_code}"\n')
            file.write('    analysis terminated ')
            if self.exit_code == 'count':
                file.write(
                    f'as number of counts reached the maximum (HFmax={self.modelInfoHF.thr_count})\n'
                )
                if self.do_mf:
                    file.write(
                        f'as number of counts reached the maximum (HFmax={self.modelInfoHF.thr_count}, LFmax={self.modelInfoLF.thr_count})\n'
                    )

            elif self.exit_code == 'accuracy':
                file.write(
                    f'as minimum accuracy level (NRMSE={self.thr_NRMSE:.2f}) is achieved"\n'
                )
            elif self.exit_code == 'time':
                file.write(
                    f'as maximum running time (t={self.thr_t:.1f}s) reached"\n'
                )
            elif self.exit_code == 'data':
                file.write('without simulation\n')
            else:
                file.write('- cannot identify the exit code\n')

            file.write(f'  - number of HF simulations : {self.id_sim_hf}\n')
            if self.do_mf:
                file.write(f'  - number of LF simulations : {self.id_sim_lf}\n')

            file.write(
                f'  - maximum normalized root-mean-squared error (NRMSE): {np.max(self.NRMSE_val):.5f}\n'
            )

            for ny in range(self.y_dim):
                file.write(f'     {self.g_name[ny]} : {self.NRMSE_val[ny]:.2f}\n')

            file.write(f'  - analysis time : {self.sim_time:.1f} sec\n')
            file.write(f'  - calibration interval : {self.cal_interval}\n')
            file.write('\n')

            file.write('* GP parameters\n'.format())
            file.write(f'  - Kernel : {self.kernel}\n')
            file.write(f'  - Linear : {self.do_linear}\n\n')

            if not self.do_mf:
                for ny in range(self.y_dim):
                    file.write(f'  [{self.g_name[ny]}]\n')
                    m_tmp = self.m_list[ny]
                    for parname in m_tmp.parameter_names():
                        file.write(f'    - {parname} ')
                        parvals = eval('m_tmp.' + parname)  # noqa: S307
                        if len(parvals) == self.x_dim:
                            file.write('\n')
                            for nx in range(self.x_dim):
                                file.write(
                                    f'       {self.rv_name[nx]} : {parvals[nx]:.2e}\n'
                                )
                        else:
                            file.write(f' : {parvals[0]:.2e}\n')
                    file.write('\n'.format())

        print('Results Saved', flush=True)  # noqa: T201
        return 0

    def run_design_of_experiments(self, nc1, nq, e2, doeIdx='HF'):  # noqa: C901, N803, D102, PLR0912, PLR0915
        if doeIdx == 'LF':
            lfset = set([tuple(x) for x in self.X_lf.tolist()])  # noqa: C403
            hfset = set([tuple(x) for x in self.X_hf.tolist()])  # noqa: C403
            hfsamples = hfset - lfset
            if len(hfsamples) == 0:
                lf_additional_candi = np.zeros((0, self.x_dim))
            else:
                lf_additional_candi = np.array([np.array(x) for x in hfsamples])

            def sampling(N):  # noqa: N803
                return model_lf.sampling(N)

        else:

            def sampling(N):  # noqa: N803
                return model_hf.sampling(N)

        # doeIdx = 0
        # doeIdx = 1 #HF
        # doeIdx = 2 #LF
        # doeIdx = 3 #HF and LF

        model_hf = self.modelInfoHF
        model_lf = self.modelInfoLF

        X_hf = self.X_hf  # noqa: N806
        Y_hf = self.Y_hf  # noqa: N806
        X_lf = self.X_lf  # noqa: N806
        Y_lf = self.Y_lf  # noqa: N806
        ll = self.ll  # TODO which ll?  # noqa: TD002, TD004

        y_var = np.var(Y_hf, axis=0)  # normalization
        y_idx = np.argmax(np.sum(e2 / y_var, axis=0))
        if np.max(y_var) == 0:
            # if this Y is constant
            self.doe_method = 'none'
            self.doe_stop = True

            # dimension of interest
        m_tmp_list = self.m_list
        m_stack = copy.deepcopy(m_tmp_list[y_idx])

        r = 1

        if self.doe_method == 'none':
            update_point = sampling(self.cal_interval)
            score = 0

        elif self.doe_method == 'pareto':
            #
            # Initial candidates
            #

            xc1 = sampling(nc1)  # same for hf/lf
            xq = sampling(nq)  # same for hf/lf

            if doeIdx.startswith('LF'):
                xc1 = np.vstack([xc1, lf_additional_candi])
                nc1 = xc1.shape[0]
            #
            # MMSE prediction
            #

            yc1_pred, yc1_var = self.predict(m_stack, xc1)  # use only variance
            cri1 = np.zeros(yc1_pred.shape)
            cri2 = np.zeros(yc1_pred.shape)

            for i in range(nc1):
                wei = weights_node2(xc1[i, :], X_hf, ll)
                # cri2[i] = sum(e2[:, y_idx] / Y_pred_var[:, y_idx] * wei.T)
                cri2[i] = sum(e2[:, y_idx] * wei.T)

            VOI = np.zeros(yc1_pred.shape)  # noqa: N806
            for i in range(nc1):
                pdfvals = (
                    m_stack.kern.K(np.array([xq[i]]), xq) ** 2
                    / m_stack.kern.K(np.array([xq[0]])) ** 2
                )
                VOI[i] = np.mean(pdfvals) * np.prod(
                    np.diff(model_hf.xrange, axis=1)
                )  # * np.prod(np.diff(self.xrange))
                cri1[i] = yc1_var[i] * VOI[i]

            cri1 = (cri1 - np.min(cri1)) / (np.max(cri1) - np.min(cri1))
            cri2 = (cri2 - np.min(cri2)) / (np.max(cri2) - np.min(cri2))
            logcrimi1 = np.log(cri1[:, 0])
            logcrimi2 = np.log(cri2[:, 0])

            rankid = np.zeros(nc1)
            varRank = np.zeros(nc1)  # noqa: N806
            biasRank = np.zeros(nc1)  # noqa: N806
            for id in range(nc1):  # noqa: A001
                idx_tmp = np.argwhere(
                    (logcrimi1 >= logcrimi1[id]) * (logcrimi2 >= logcrimi2[id])
                )
                varRank[id] = np.sum(logcrimi1 >= logcrimi1[id])
                biasRank[id] = np.sum(logcrimi2 >= logcrimi2[id])
                rankid[id] = idx_tmp.size

            num_1rank = np.sum(rankid == 1)
            idx_1rank = list((np.argwhere(rankid == 1)).flatten())

            if doeIdx.startswith('HF'):
                X_stack = X_hf  # noqa: N806
                Y_stack = Y_hf[:, y_idx][np.newaxis].T  # noqa: N806
            elif doeIdx.startswith('LF'):
                X_stack = X_lf  # noqa: N806
                Y_stack = Y_lf[:, y_idx][np.newaxis].T  # noqa: N806

            if num_1rank < self.cal_interval:
                # When number of pareto is smaller than cal_interval
                prob = np.ones((nc1,))
                prob[list(rankid == 1)] = 0
                prob = prob / sum(prob)
                idx_pareto = idx_1rank + list(
                    np.random.choice(nc1, self.cal_interval - num_1rank, p=prob)
                )
            else:
                idx_pareto_candi = idx_1rank.copy()
                m_tmp = copy.deepcopy(m_stack)

                # get MMSEw
                score = np.squeeze(cri1 * cri2)
                score_candi = score[idx_pareto_candi]
                best_local = np.argsort(-score_candi)[0]
                best_global = idx_1rank[best_local]

                idx_pareto_new = [best_global]
                del idx_pareto_candi[best_local]

                for i in range(self.cal_interval - 1):  # noqa: B007
                    X_stack = np.vstack([X_stack, xc1[best_global, :][np.newaxis]])  # noqa: N806
                    # any variables
                    Y_stack = np.vstack([Y_stack, np.zeros((1, 1))])  # noqa: N806

                    if doeIdx.startswith('HF'):
                        m_stack = self.set_XY(m_stack, y_idx, X_stack, Y_stack)
                    elif doeIdx.startswith('LF'):  # any variables
                        m_tmp = self.set_XY(
                            m_tmp, y_idx, self.X_hf, self.Y_hf, X_stack, Y_stack
                        )

                    dummy, Yq_var = self.predict(m_stack, xc1[idx_pareto_candi, :])  # noqa: N806
                    cri1 = Yq_var * VOI[idx_pareto_candi]
                    cri1 = (cri1 - np.min(cri1)) / (np.max(cri1) - np.min(cri1))
                    score_tmp = (
                        cri1 * cri2[idx_pareto_candi]
                    )  # only update the variance

                    best_local = np.argsort(-np.squeeze(score_tmp))[0]
                    best_global = idx_pareto_candi[best_local]
                    idx_pareto_new = idx_pareto_new + [best_global]  # noqa: RUF005
                    del idx_pareto_candi[best_local]
                idx_pareto = idx_pareto_new

            update_point = xc1[idx_pareto, :]
            score = 0

        elif self.doe_method == 'imse':
            update_point = np.zeros((self.cal_interval, self.x_dim))
            update_score = np.zeros((self.cal_interval, 1))

            if doeIdx.startswith('HF'):
                X_stack = X_hf  # noqa: N806
                Y_stack = Y_hf[:, y_idx][np.newaxis].T  # noqa: N806
            elif doeIdx.startswith('LF'):
                X_stack = X_lf  # noqa: N806
                Y_stack = Y_lf[:, y_idx][np.newaxis].T  # noqa: N806

            for ni in range(self.cal_interval):
                #
                # Initial candidates
                #
                xc1 = sampling(nc1)  # same for hf/lf
                if doeIdx.startswith('LF'):
                    xc1 = np.vstack([xc1, lf_additional_candi])
                    nc1 = xc1.shape[0]

                xq = sampling(nq)  # same for hf/lf

                dummy, Yq_var = self.predict(m_stack, xq)  # noqa: N806
                if ni == 0:
                    IMSEbase = 1 / xq.shape[0] * sum(Yq_var.flatten())  # noqa: N806

                tmp = time.time()
                if self.do_parallel:
                    iterables = (
                        (
                            copy.deepcopy(m_stack),
                            xc1[i, :][np.newaxis],
                            xq,
                            np.ones((nq, self.y_dim)),
                            i,
                            y_idx,
                            doeIdx,
                        )
                        for i in range(nc1)
                    )
                    result_objs = list(self.pool.starmap(imse, iterables))
                    IMSEc1 = np.zeros(nc1)  # noqa: N806
                    for IMSE_val, idx in result_objs:  # noqa: N806
                        IMSEc1[idx] = IMSE_val
                    print(  # noqa: T201
                        f'IMSE: finding the next DOE {ni} - parallel .. time = {time.time() - tmp:.2f}'
                    )  # 7s # 3-4s
                    # TODO: terminate it gracefully....  # noqa: TD002
                    # see https://stackoverflow.com/questions/21104997/keyboard-interrupt-with-pythons-multiprocessing
                    try:
                        while True:
                            time.sleep(0.5)
                            if all([r.ready() for r in result]):  # type: ignore # noqa: C419, F821
                                break
                    except KeyboardInterrupt:
                        pool.terminate()  # type: ignore # noqa: F821
                        pool.join()  # type: ignore # noqa: F821

                else:
                    IMSEc1 = np.zeros(nc1)  # noqa: N806
                    for i in range(nc1):
                        IMSEc1[i], dummy = imse(
                            copy.deepcopy(m_stack),
                            xc1[i, :][np.newaxis],
                            xq,
                            np.ones((nq, self.y_dim)),
                            i,
                            y_idx,
                            doeIdx,
                        )
                    print(  # noqa: T201
                        f'IMSE: finding the next DOE {ni} - serial .. time = {time.time() - tmp}'
                    )  # 4s

                new_idx = np.argmin(IMSEc1, axis=0)
                x_point = xc1[new_idx, :][np.newaxis]

                X_stack = np.vstack([X_stack, x_point])  # noqa: N806
                # any variables
                Y_stack = np.vstack([Y_stack, np.zeros((1, 1))])  # noqa: N806
                update_point[ni, :] = x_point

                if doeIdx == 'HFHF':
                    m_stack = self.set_XY(
                        m_stack,
                        y_idx,
                        X_stack,
                        Y_stack,
                        self.X_lf,
                        self.Y_lf[:, y_idx][np.newaxis].T,
                    )
                elif doeIdx == 'HF':
                    m_stack = self.set_XY(m_stack, y_idx, X_stack, Y_stack)
                elif doeIdx == 'LF':  # any variables
                    m_stack = self.set_XY(
                        m_stack,
                        y_idx,
                        self.X_hf,
                        self.Y_hf[:, y_idx][np.newaxis].T,
                        X_stack,
                        Y_stack,
                    )

            score = IMSEbase - np.min(IMSEc1, axis=0)

        elif self.doe_method == 'imsew':
            update_point = np.zeros((self.cal_interval, self.x_dim))
            update_score = np.zeros((self.cal_interval, 1))  # noqa: F841

            if doeIdx.startswith('HF'):
                X_stack = X_hf  # noqa: N806
                Y_stack = Y_hf[:, y_idx][np.newaxis].T  # noqa: N806
            elif doeIdx.startswith('LF'):
                X_stack = X_lf  # noqa: N806
                Y_stack = Y_lf[:, y_idx][np.newaxis].T  # noqa: N806

            for ni in range(self.cal_interval):
                #
                # Initial candidates
                #
                xc1 = sampling(nc1)  # same for hf/lf
                if doeIdx.startswith('LF'):
                    xc1 = np.vstack([xc1, lf_additional_candi])
                    nc1 = xc1.shape[0]

                xq = sampling(nq)  # same for hf/lf

                phiq = np.zeros((nq, self.y_dim))
                for i in range(nq):
                    phiq[i, :] = e2[closest_node(xq[i, :], X_hf, ll)]
                phiqr = pow(phiq[:, y_idx], r)

                dummy, Yq_var = self.predict(m_stack, xq)  # noqa: N806
                if ni == 0:
                    IMSEbase = (  # noqa: N806
                        1 / xq.shape[0] * sum(phiqr.flatten() * Yq_var.flatten())
                    )

                tmp = time.time()
                if self.do_parallel:
                    iterables = (
                        (
                            copy.deepcopy(m_stack),
                            xc1[i, :][np.newaxis],
                            xq,
                            phiqr,
                            i,
                            y_idx,
                            doeIdx,
                        )
                        for i in range(nc1)
                    )
                    result_objs = list(self.pool.starmap(imse, iterables))
                    IMSEc1 = np.zeros(nc1)  # noqa: N806
                    for IMSE_val, idx in result_objs:  # noqa: N806
                        IMSEc1[idx] = IMSE_val
                    print(  # noqa: T201
                        f'IMSE: finding the next DOE {ni} - parallel .. time = {time.time() - tmp:.2f}'
                    )  # 7s # 3-4s
                else:
                    IMSEc1 = np.zeros(nc1)  # noqa: N806
                    for i in range(nc1):
                        IMSEc1[i], dummy = imse(
                            copy.deepcopy(m_stack),
                            xc1[i, :][np.newaxis],
                            xq,
                            phiqr,
                            i,
                            y_idx,
                            doeIdx,
                        )
                        if np.mod(i, 200) == 0:
                            # 4s
                            print(f'IMSE iter {ni}, candi {i}/{nc1}')  # noqa: T201
                    print(  # noqa: T201
                        f'IMSE: finding the next DOE {ni} - serial .. time = {time.time() - tmp}'
                    )  # 4s

                new_idx = np.argmin(IMSEc1, axis=0)
                x_point = xc1[new_idx, :][np.newaxis]

                X_stack = np.vstack([X_stack, x_point])  # noqa: N806
                # any variables
                Y_stack = np.vstack([Y_stack, np.zeros((1, 1))])  # noqa: N806
                update_point[ni, :] = x_point

                if doeIdx == 'HFHF':
                    m_stack = self.set_XY(
                        m_stack,
                        y_idx,
                        X_stack,
                        Y_stack,
                        self.X_lf,
                        self.Y_lf[:, y_idx][np.newaxis].T,
                    )
                elif doeIdx == 'HF':
                    m_stack = self.set_XY(m_stack, y_idx, X_stack, Y_stack)
                elif doeIdx == 'LF':  # any variables
                    m_stack = self.set_XY(
                        m_stack,
                        y_idx,
                        self.X_hf,
                        self.Y_hf[:, y_idx][np.newaxis].T,
                        X_stack,
                        Y_stack,
                    )

            score = IMSEbase - np.min(IMSEc1, axis=0)

        elif self.doe_method == 'mmsew':
            if doeIdx.startswith('HF'):
                X_stack = X_hf  # noqa: N806
                Y_stack = Y_hf[:, y_idx][np.newaxis].T  # noqa: N806
            elif doeIdx.startswith('LF'):
                X_stack = X_lf  # noqa: N806
                Y_stack = Y_lf[:, y_idx][np.newaxis].T  # noqa: N806

            update_point = np.zeros((self.cal_interval, self.x_dim))

            for ni in range(self.cal_interval):
                xc1 = sampling(nc1)  # same for hf/lf
                if doeIdx.startswith('LF'):
                    xc1 = np.vstack([xc1, lf_additional_candi])
                    nc1 = xc1.shape[0]

                phic = np.zeros((nc1, self.y_dim))
                for i in range(nc1):
                    phic[i, :] = e2[closest_node(xc1[i, :], X_hf, ll)]
                phicr = pow(phic[:, y_idx], r)

                yc1_pred, yc1_var = self.predict(m_stack, xc1)  # use only variance
                MMSEc1 = yc1_var.flatten() * phicr.flatten()  # noqa: N806
                new_idx = np.argmax(MMSEc1, axis=0)
                x_point = xc1[new_idx, :][np.newaxis]

                X_stack = np.vstack([X_stack, x_point])  # noqa: N806
                # any variables
                Y_stack = np.vstack([Y_stack, np.zeros((1, 1))])  # noqa: N806
                # m_stack.set_XY(X=X_stack, Y=Y_stack)
                if doeIdx.startswith('HF'):
                    m_stack = self.set_XY(m_stack, y_idx, X_stack, Y_stack)
                elif doeIdx.startswith('LF'):  # any variables
                    m_tmp = self.set_XY(
                        m_tmp, y_idx, self.X_hf, self.Y_hf, X_stack, Y_stack
                    )
                update_point[ni, :] = x_point

            score = np.max(MMSEc1, axis=0)

        elif self.doe_method == 'mmse':
            if doeIdx.startswith('HF'):
                X_stack = X_hf  # noqa: N806
                Y_stack = Y_hf[:, y_idx][np.newaxis].T  # noqa: N806
            elif doeIdx.startswith('LF'):
                X_stack = X_lf  # noqa: N806
                Y_stack = Y_lf[:, y_idx][np.newaxis].T  # noqa: N806

            update_point = np.zeros((self.cal_interval, self.x_dim))

            for ni in range(self.cal_interval):
                xc1 = sampling(nc1)  # same for hf/lf
                if doeIdx.startswith('LF'):
                    xc1 = np.vstack([xc1, lf_additional_candi])
                    nc1 = xc1.shape[0]

                yc1_pred, yc1_var = self.predict(m_stack, xc1)  # use only variance
                MMSEc1 = yc1_var.flatten()  # noqa: N806
                new_idx = np.argmax(MMSEc1, axis=0)
                x_point = xc1[new_idx, :][np.newaxis]

                X_stack = np.vstack([X_stack, x_point])  # noqa: N806
                # any variables
                Y_stack = np.vstack([Y_stack, np.zeros((1, 1))])  # noqa: N806
                # m_stack.set_XY(X=X_stack, Y=Y_stack)

                # if doeIdx.startswith("HF"):
                #     self.set_XY(m_stack, X_stack, Y_stack)
                # elif doeIdx.startswith("LF"):  # any variables
                #     self.set_XY(m_stack, self.X_hf, self.Y_hf, X_stack, Y_stack)

                if doeIdx == 'HFHF':
                    m_stack = self.set_XY(
                        m_stack,
                        y_idx,
                        X_stack,
                        Y_stack,
                        self.X_lf,
                        self.Y_lf[:, y_idx][np.newaxis].T,
                    )
                elif doeIdx == 'HF':
                    m_stack = self.set_XY(m_stack, y_idx, X_stack, Y_stack)
                elif doeIdx == 'LF':  # any variables
                    m_stack = self.set_XY(
                        m_stack,
                        y_idx,
                        self.X_hf,
                        self.Y_hf[:, y_idx][np.newaxis].T,
                        X_stack,
                        Y_stack,
                    )

                update_point[ni, :] = x_point

            score = np.max(MMSEc1, axis=0)
        else:
            msg = (
                'Error running SimCenterUQ: cannot identify the doe method <'
                + self.doe_method
                + '>'
            )
            self.exit(msg)

        return update_point, y_idx, score

    def normalized_mean_sq_error(self, yp, ye):  # noqa: D102
        n = yp.shape[0]
        data_bound = np.max(ye, axis=0) - np.min(ye, axis=0)
        RMSE = np.sqrt(1 / n * np.sum(pow(yp - ye, 2), axis=0))  # noqa: N806
        NRMSE = RMSE / data_bound  # noqa: N806
        NRMSE[np.argwhere(data_bound == 0)] = 0
        return NRMSE

    def get_cross_validation_err(self):  # noqa: D102
        print('Calculating cross validation errors', flush=True)  # noqa: T201
        time_tmp = time.time()
        X_hf = self.X_hf  # contains separate samples  # noqa: N806
        nsamp = self.X_hf.shape[0]
        ydim = self.Y_hf.shape[1]

        e2 = np.zeros((nsamp, ydim))  # only for unique...
        Y_pred = np.zeros((nsamp, ydim))  # noqa: N806
        Y_pred_var = np.zeros((nsamp, ydim))  # noqa: N806
        Y_pred_var_w_measure = np.zeros((nsamp, ydim))  # noqa: N806
        #
        # Efficient cross validation TODO: check if it works for heteroskedacstic
        #

        if (not self.do_mf) and (
            not self.heteroscedastic
        ):  # note: heteroscedastic is not our stochastic kriging
            # X_unique, dum, indices, dum = np.unique(X_hf, axis=0, return_index=True, return_counts=True,
            #                                   return_inverse=True)
            # self.n_unique_hf = indices.shape[0]

            indices = self.indices_unique

            for ny in range(ydim):
                Xm = self.m_list[ny].X  # contains unique samples  # noqa: N806
                Ym = self.m_list[ny].Y  # noqa: N806

                # works both for stochastic/stochastic
                nugget_mat = (
                    np.diag(np.squeeze(self.var_str[ny]))
                    * self.m_list[
                        ny
                    ].Gaussian_noise.parameters  # TODO  # noqa: TD002, TD004
                )

                Rmat = self.m_list[ny].kern.K(Xm)  # noqa: N806
                Rinv = np.linalg.inv(Rmat + nugget_mat)  # noqa: N806
                e = np.squeeze(
                    np.matmul(Rinv, (Ym - self.normMeans[ny]))
                ) / np.squeeze(np.diag(Rinv))
                # e = np.squeeze(np.matmul(Rinv, (Ym))) / np.squeeze(np.diag(Rinv))

                # works both for stochastic/stochastic
                for nx in range(X_hf.shape[0]):
                    e2[nx, ny] = e[indices[nx]] ** 2
                    # Y_pred_var[nx, ny] = 1 / np.diag(Rinv)[indices[nx]]  * self.normVars[ny]
                    Y_pred[nx, ny] = self.Y_mean[ny][indices[nx]] - e[indices[nx]]
                    # Y_pred_var_w_measure[nx, ny] = Y_pred_var[nx, ny]  + self.m_list[ny].Gaussian_noise.parameters[0]*self.var_str[ny][indices[nx]] * self.normVars[ny]
                    Y_pred_var_w_measure[nx, ny] = (
                        1 / np.diag(Rinv)[indices[nx]] * self.normVars[ny]
                    )
                    Y_pred_var[nx, ny] = max(
                        0,
                        Y_pred_var_w_measure[nx, ny]
                        - self.m_list[ny].Gaussian_noise.parameters[0]
                        * self.var_str[ny][indices[nx]]
                        * self.normVars[ny],
                    )

                if sum(self.linear_list) > 0:
                    Y_pred[:, ny] = (
                        Y_pred[:, ny]
                        + self.lin_list[ny].predict(X_hf[:, self.linear_list])[:, 0]
                    )

        else:
            Y_hf = self.Y_hf  # noqa: N806
            Y_pred2 = np.zeros(Y_hf.shape)  # noqa: N806
            Y_pred_var2 = np.zeros(Y_hf.shape)  # noqa: N806
            e22 = np.zeros(Y_hf.shape)

            for ny in range(Y_hf.shape[1]):
                m_tmp = copy.deepcopy(self.m_list[ny])
                for ns in range(X_hf.shape[0]):
                    X_tmp = np.delete(X_hf, ns, axis=0)  # noqa: N806
                    Y_tmp = np.delete(Y_hf, ns, axis=0)  # noqa: N806

                    if self.stochastic:
                        Y_meta_tmp = m_tmp.Y_metadata  # noqa: N806
                        Y_meta_tmp['variance_structure'] = np.delete(
                            m_tmp.Y_metadata['variance_structure'], ns, axis=0
                        )
                        m_tmp.set_XY2(
                            X_tmp,
                            Y_tmp[:, ny][np.newaxis].transpose(),
                            Y_metadata=Y_meta_tmp,
                        )

                    else:
                        m_tmp.set_XY(X_tmp, Y_tmp[:, ny][np.newaxis].transpose())
                    print(ns)  # noqa: T201
                    # m_tmp = self.set_XY(
                    #     m_tmp,
                    #     ny,
                    #     X_tmp,
                    #     Y_tmp[:, ny][np.newaxis].transpose(),
                    #     self.X_lf,
                    #     self.Y_lf[:, ny][np.newaxis].transpose(),
                    # )

                    x_loo = X_hf[ns, :][np.newaxis]
                    Y_pred_tmp, Y_err_tmp = self.predict(m_tmp, x_loo)  # noqa: N806

                    Y_pred2[ns, ny] = Y_pred_tmp
                    Y_pred_var2[ns, ny] = Y_err_tmp

                    if self.do_logtransform:
                        Y_exact = np.log(Y_hf[ns, ny])  # noqa: N806
                    else:
                        Y_exact = Y_hf[ns, ny]  # noqa: N806

                    e22[ns, ny] = pow((Y_pred_tmp - Y_exact), 2)  # for nD outputs

                Y_pred = Y_pred2  # noqa: N806
                Y_pred_var = Y_pred_var2  # noqa: N806
                if not self.do_mf:
                    Y_pred_var_w_measure[:, ny] = (
                        Y_pred_var2[:, ny]
                        + self.m_list[ny].Gaussian_noise.parameters
                        * self.normVars[ny]
                    )
                else:
                    # TODO account for Gaussian_noise.parameters as well  # noqa: TD002, TD004
                    Y_pred_var_w_measure[:, ny] = (
                        Y_pred_var2[:, ny]
                        + self.m_list[
                            ny
                        ].gpy_model.mixed_noise.Gaussian_noise_1.parameters
                        * self.normVars[ny]
                    )
                e2 = e22
                # np.hstack([Y_pred_var,Y_pred_var2])
                # np.hstack([e2,e22])
                r"""
                
                import matplotlib.pyplot as plt
                plt.plot(Y_pred_var/self.normVars[ny]); plt.plot(Y_pred_var2/self.normVars[ny]); 
                plt.title("With nugget (Linear)"); plt.xlabel("Training sample id"); plt.ylabel("LOOCV variance (before multiplying $\sigma_z^2$)"); plt.legend(["Closedform","iteration"]);
                
                plt.show(); 
                """  # noqa: W291, W293
        print(  # noqa: T201
            f'     Cross validation calculation time: {time.time() - time_tmp:.2f} s',
            flush=True,
        )
        return Y_pred, Y_pred_var, Y_pred_var_w_measure, e2


def imse(m_tmp, xcandi, xq, phiqr, i, y_idx, doeIdx='HF'):  # noqa: ARG001, N803, D103
    if doeIdx == 'HF':
        X = m_tmp.X  # noqa: N806
        Y = m_tmp.Y  # noqa: N806
        X_tmp = np.vstack([X, xcandi])  # noqa: N806
        # any variables
        Y_tmp = np.vstack([Y, np.zeros((1, Y.shape[1]))])  # noqa: N806
        # self.set_XY(m_tmp, X_tmp, Y_tmp)
        m_tmp.set_XY(X_tmp, Y_tmp)
        dummy, Yq_var = m_tmp.predict(xq)  # noqa: N806

    elif doeIdx == 'HFHF':
        idxHF = np.argwhere(m_tmp.gpy_model.X[:, -1] == 0).T[0]  # noqa: N806
        idxLF = np.argwhere(m_tmp.gpy_model.X[:, -1] == 1).T[0]  # noqa: N806
        X_hf = m_tmp.gpy_model.X[idxHF, :-1]  # noqa: N806
        Y_hf = m_tmp.gpy_model.Y[idxHF, :]  # noqa: N806
        X_lf = m_tmp.gpy_model.X[idxLF, :-1]  # noqa: N806
        Y_lf = m_tmp.gpy_model.Y[idxLF, :]  # noqa: N806
        X_tmp = np.vstack([X_hf, xcandi])  # noqa: N806
        # any variables
        Y_tmp = np.vstack([Y_hf, np.zeros((1, Y_hf.shape[1]))])  # noqa: N806
        # self.set_XY(m_tmp, X_tmp, Y_tmp, X_lf, Y_lf)
        X_list_tmp, Y_list_tmp = (  # noqa: N806
            emf.convert_lists_to_array.convert_xy_lists_to_arrays(  # noqa: F821 # type: ignore
                [X_tmp, X_lf], [Y_tmp, Y_lf]
            )
        )
        m_tmp.set_data(X=X_list_tmp, Y=Y_list_tmp)
        xq_list = convert_x_list_to_array([xq, np.zeros((0, xq.shape[1]))])  # type: ignore # noqa: F821
        dummy, Yq_var = m_tmp.predict(xq_list)  # noqa: N806

    elif doeIdx.startswith('LF'):
        idxHF = np.argwhere(m_tmp.gpy_model.X[:, -1] == 0).T[0]  # noqa: N806
        idxLF = np.argwhere(m_tmp.gpy_model.X[:, -1] == 1).T[0]  # noqa: N806
        X_hf = m_tmp.gpy_model.X[idxHF, :-1]  # noqa: N806
        Y_hf = m_tmp.gpy_model.Y[idxHF, :]  # noqa: N806
        X_lf = m_tmp.gpy_model.X[idxLF, :-1]  # noqa: N806
        Y_lf = m_tmp.gpy_model.Y[idxLF, :]  # noqa: N806
        X_tmp = np.vstack([X_lf, xcandi])  # noqa: N806
        # any variables
        Y_tmp = np.vstack([Y_lf, np.zeros((1, Y_lf.shape[1]))])  # noqa: N806
        # self.set_XY(m_tmp, X_hf, Y_hf, X_tmp, Y_tmp)
        X_list_tmp, Y_list_tmp = (  # noqa: N806
            emf.convert_lists_to_array.convert_xy_lists_to_arrays(  # noqa: F821 # type: ignore
                [X_hf, X_tmp], [Y_hf, Y_tmp]
            )
        )
        m_tmp.set_data(X=X_list_tmp, Y=Y_list_tmp)
        xq_list = convert_x_list_to_array([xq, np.zeros((0, xq.shape[1]))])  # type: ignore # noqa: F821
        dummy, Yq_var = m_tmp.predict(xq_list)  # noqa: N806
    else:
        print(f'doe method <{doeIdx}> is not supported', flush=True)  # noqa: T201

    # dummy, Yq_var = self.predict(m_tmp,xq)
    IMSEc1 = 1 / xq.shape[0] * sum(phiqr.flatten() * Yq_var.flatten())  # noqa: N806

    return IMSEc1, i


class model_info:  # noqa: D101
    def __init__(  # noqa: C901
        self,
        surrogateJson,  # noqa: N803
        rvJson,  # noqa: N803
        work_dir,
        x_dim,
        y_dim,
        n_processor,
        global_seed,
        idx=0,
    ):
        def exit_tmp(msg):
            print(msg)  # noqa: T201
            print(msg, file=sys.stderr)  # noqa: T201
            exit(-1)  # noqa: PLR1722

        # idx = -1 : no info (dummy) paired with 0
        # idx = 0 : single fidelity
        # idx = 1 : high fidelity FEM run with tag 1
        # idx = 2 : low fidelity
        self.idx = idx
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.global_seed = global_seed
        #
        # Get [X_existing, Y_existing, n_existing, n_total]
        #

        self.model_without_sampling = False  # default
        if idx == 0:
            # not MF
            if surrogateJson['method'] == 'Sampling and Simulation':
                self.is_model = True
                self.is_data = surrogateJson['existingDoE']
            elif surrogateJson['method'] == 'Import Data File':
                self.is_model = False
                self.is_data = True
                if not surrogateJson['outputData']:
                    self.model_without_sampling = True  # checkbox not checked...
            else:
                msg = 'Error reading json: either select "Import Data File" or "Sampling and Simulation"'
                exit_tmp(msg)

        elif idx == 1 or idx == 2:  # noqa: PLR1714, PLR2004
            # MF
            self.is_data = True  # default
            self.is_model = surrogateJson['fromModel']
            if self.is_model:
                self.is_data = surrogateJson['existingDoE']
        elif idx == -1:
            self.is_data = False
            self.is_model = False

        if idx == 0:
            # single model
            input_file = 'templatedir/inpFile.in'
            output_file = 'templatedir/outFile.in'
        elif idx == 1:
            # high-fidelity
            input_file = 'templatedir/inpFile_HF.in'
            output_file = 'templatedir/outFile_HF.in'
        elif idx == 2:  # noqa: PLR2004
            # low-fidelity
            input_file = 'templatedir/inpFile_LF.in'
            output_file = 'templatedir/outFile_LF.in'

        if self.is_data:
            self.inpData = os.path.join(work_dir, input_file)  # noqa: PTH118
            self.outData = os.path.join(work_dir, output_file)  # noqa: PTH118

            self.X_existing = read_txt(self.inpData, exit_tmp)
            self.n_existing = self.X_existing.shape[0]

            if self.X_existing.shape[1] != self.x_dim:
                msg = f'Error importing input data - dimension inconsistent: have {self.x_dim} RV(s) but have {self.X_existing.shape[1]} column(s).'
                exit_tmp(msg)

            if not self.model_without_sampling:  # i.e. check box clicked
                self.Y_existing = read_txt(self.outData, exit_tmp)

                if self.Y_existing.shape[1] != self.y_dim:
                    msg = f'Error importing input data - dimension inconsistent: have {self.y_dim} QoI(s) but have {self.Y_existing.shape[1]} column(s).'
                    exit_tmp(msg)

                if self.Y_existing.shape[0] != self.X_existing.shape[0]:
                    msg = f'Error importing input data: numbers of samples of inputs ({self.X_existing.shape[0]}) and outputs ({self.Y_existing.shape[0]}) are inconsistent'
                    exit_tmp(msg)
            else:
                self.Y_existing = np.zeros((0, y_dim))

        else:
            self.inpData = ''
            self.outData = ''
            self.X_existing = np.zeros((0, x_dim))
            self.Y_existing = np.zeros((0, y_dim))
            self.n_existing = 0

        if self.is_model:
            self.doe_method = surrogateJson['DoEmethod']
            self.doe_method = surrogateJson['DoEmethod']

            self.thr_count = surrogateJson['samples']  # number of samples
            if self.thr_count == 1:
                msg = 'The number of samples should be greater.'
                exit_tmp(msg)

            if self.doe_method == 'None':
                self.user_init = self.thr_count
            else:
                try:
                    self.user_init = surrogateJson['initialDoE']
                except:  # noqa: E722
                    self.user_init = -1  # automate

            self.nugget_opt = surrogateJson['nuggetOpt']
            if self.nugget_opt == 'Heteroscedastic':
                self.numSampToBeRepl = surrogateJson['numSampToBeRepl']
                self.numRepl = surrogateJson['numRepl']
                self.numSampRepldone = False

                if self.numRepl == -1:  # use default
                    self.numRepl = 10
                # elif self.numRepl < 2 :
                #    msg = "Error reading json: number of replications should be greater than 1 and a value greater than 5 is recommended"
                #    exit_tmp(msg)

                if self.numSampToBeRepl == -1:  # use default
                    self.numSampToBeRepl = 8 * x_dim
                # elif self.numSampToBeRepl < 2 or self.numSampToBeRepl > self.thr_count:
                #     msg = "Error reading json: number of samples to be replicated should be greater than 1 and smaller than the number of the original samples. A value greater than 4*#RV is recommended"
                #     exit_tmp(msg)
                #     #self.numSampRepldone = True
            else:
                self.numSampToBeRepl = 0
                self.numRepl = 0
                self.numSampRepldone = True

            # convergence criteria
            self.thr_NRMSE = surrogateJson['accuracyLimit']
            self.thr_t = surrogateJson['timeLimit'] * 60

            self.xrange = np.empty((0, 2), float)
            self.xDistTypeArr = []
            for rv in rvJson:
                if rv['distribution'] == 'Uniform':
                    self.xrange = np.vstack(
                        (self.xrange, [rv['lowerbound'], rv['upperbound']])
                    )
                    self.xDistTypeArr += ['U']
                elif rv['distribution'] == 'discrete_design_set_string':
                    self.xrange = np.vstack((self.xrange, [1, len(rv['elements'])]))
                    self.xDistTypeArr += ['DS']
                else:
                    msg = 'Error in input RV: all RV should be set to Uniform distribution'
                    exit_tmp(msg)

        else:
            self.doe_method = 'None'
            self.user_init = 0
            self.thr_count = 0
            self.thr_NRMSE = 0.02
            self.thr_t = float('inf')
            if self.is_data:
                self.xrange = np.vstack(
                    [
                        np.min(self.X_existing, axis=0),
                        np.max(self.X_existing, axis=0),
                    ]
                ).T
            else:
                self.xrange = np.zeros((self.x_dim, 2))
        # TODO should I use "effective" number of dims?  # noqa: TD002, TD004
        self.ll = self.xrange[:, 1] - self.xrange[:, 0]
        if self.user_init <= 0:  # automated choice 8*D
            n_init_tmp = int(np.ceil(8 * self.x_dim / n_processor) * n_processor)
        else:
            n_init_tmp = int(
                np.ceil(self.user_init / n_processor) * n_processor
            )  # Make every workers busy
        self.n_init = min(self.thr_count, n_init_tmp)
        # self.n_init = 4
        self.doe_method = self.doe_method.lower()

    def sampling(self, n):  # noqa: D102
        # n is "total" samples

        if n > 0:
            X_samples = np.zeros((n, self.x_dim))  # noqa: N806
            # LHS
            sampler = qmc.LatinHypercube(d=self.x_dim, seed=self.global_seed)
            U = sampler.random(n=n)  # noqa: N806
            for nx in range(self.x_dim):
                if self.xDistTypeArr[nx] == 'U':
                    X_samples[:, nx] = (
                        U[:, nx] * (self.xrange[nx, 1] - self.xrange[nx, 0])
                        + self.xrange[nx, 0]
                    )
                else:
                    X_samples[:, nx] = np.ceil(U[:, nx] * self.xrange[nx, 1])

            if (
                self.numRepl
            ) * self.numSampToBeRepl > 0 and not self.numSampRepldone:
                X_samples = np.vstack(  # noqa: N806
                    [
                        X_samples,
                        np.tile(
                            X_samples[0 : self.numSampToBeRepl, :],
                            (self.numRepl - 1, 1),
                        ),
                    ]
                )
                self.numSampRepldone = True
        else:
            X_samples = np.zeros((0, self.x_dim))  # noqa: N806

        return X_samples

    def resampling(self, X, n):  # noqa: N803, D102
        # n is "total" samples
        # cube bounds obtained from data
        dim = X.shape[1]
        minvals = np.min(X, axis=0)
        maxvals = np.max(X, axis=0)
        print(dim)  # noqa: T201
        X_samples = np.zeros((n, dim))  # noqa: N806

        sampler = qmc.LatinHypercube(d=dim)
        U = sampler.random(n=n)  # noqa: N806

        for nx in range(dim):
            X_samples[:, nx] = U[:, nx] * (maxvals[nx] - minvals[nx]) + minvals[nx]

        return X_samples

    # def set_FEM(self, rv_name, do_parallel, y_dim, t_init):
    #     self.rv_name = rv_name
    #     self.do_parallel = do_parallel
    #     self.y_dim = y_dim
    #     self.t_init = t_init
    #     self.total_sim_time = 0
    #
    # def run_FEM(self,X, id_sim):
    #     tmp = time.time()
    #     if self.is_model:
    #         X, Y, id_sim = self.run_FEM_batch(X, id_sim, self.rv_name, self.do_parallel, self.y_dim, self.t_init, self.thr_t, runIdx=self.runIdx)
    #
    #     else:
    #         X, Y, id_sim =  np.zeros((0, self.x_dim)), np.zeros((0, self.y_dim)), id_sim
    #
    #     self.total_sim_time += tmp
    #     self.avg_sim_time = self.total_sim_time / id_sim
    #
    #     return X, Y, id_sim


# Additional functions


def weights_node2(node, nodes, ls):  # noqa: D103
    nodes = np.asarray(nodes)
    deltas = nodes - node
    deltas_norm = np.zeros(deltas.shape)
    for nx in range(ls.shape[0]):
        deltas_norm[:, nx] = (
            (deltas[:, nx]) / ls[nx] * nodes.shape[0]
        )  # additional weights?
    dist_ls = np.sqrt(np.sum(pow(deltas_norm, 2), axis=1))
    weig = np.exp(-pow(dist_ls, 2))
    if sum(weig) == 0:
        weig = np.ones(nodes.shape[0])
    return weig / sum(weig)


def calibrating(  # noqa: C901, D103
    m_tmp,
    nugget_opt_tmp,
    nuggetVal,  # noqa: N803
    normVar,  # noqa: N803
    do_mf,
    do_heteroscedastic,
    nopt,
    ny,
    n_processor,
    is_paralle_opt_safe,
    init_noise_var,
    init_process_var,
):  # nuggetVal = self.nuggetVal[ny]
    np.random.seed(int(ny))
    random.seed(int(ny))

    msg = ''

    if do_heteroscedastic:
        variance_keyword = 'het_Gauss.variance'
    else:
        variance_keyword = 'Gaussian_noise.variance'

    if not do_mf:
        if nugget_opt_tmp == 'Optimize':
            # m_tmp[variance_keyword].unfix()
            X = m_tmp.X  # noqa: N806
            # for parname in m_tmp.parameter_names():
            #     if parname.endswith('lengthscale'):
            #         for nx in range(X.shape[1]):  # noqa: B007, RUF100
            #             myrange = np.max(X, axis=0) - np.min(X, axis=0)
            #             exec('m_tmp.' + parname + '[[nx]] = myrange[nx]')  # noqa: RUF100, S102

            m_tmp[variance_keyword].constrain_bounded(0.05, 2, warning=False)
            for parname in m_tmp.parameter_names():
                if parname.endswith('lengthscale'):
                    for nx in range(X.shape[1]):  # noqa: B007
                        myrange = np.max(X, axis=0) - np.min(X, axis=0)  # noqa: F841, RUF100
                        exec(  # noqa: S102
                            'm_tmp.'
                            + parname
                            + '[[nx]].constrain_bounded(myrange[nx] / X.shape[0]*10, myrange[nx],warning=False)'
                        )
                        # m_tmp[parname][nx].constrain_bounded(myrange[nx] / X.shape[0], myrange[nx]*100)
        elif nugget_opt_tmp == 'Fixed Values':
            m_tmp[variance_keyword].constrain_fixed(
                nuggetVal[ny] / normVar, warning=False
            )
        elif nugget_opt_tmp == 'Fixed Bounds':
            m_tmp[variance_keyword].constrain_bounded(
                nuggetVal[ny][0] / normVar, nuggetVal[ny][1] / normVar, warning=False
            )
        elif nugget_opt_tmp == 'Zero':
            m_tmp[variance_keyword].constrain_fixed(0, warning=False)
            X = m_tmp.X  # noqa: N806
            for parname in m_tmp.parameter_names():
                if parname.endswith('lengthscale'):
                    for nx in range(X.shape[1]):  # noqa: B007
                        myrange = np.max(X, axis=0) - np.min(X, axis=0)
                        exec('m_tmp.' + parname + '[[nx]] = myrange[nx]')  # noqa: S102
        elif nugget_opt_tmp == 'Heteroscedastic':
            if init_noise_var == None:  # noqa: E711
                init_noise_var = 1

            if init_process_var == None:  # noqa: E711
                init_process_var = 1
            X = m_tmp.X  # noqa: N806

            for parname in m_tmp.parameter_names():
                if parname.endswith('lengthscale'):
                    for nx in range(X.shape[1]):  # noqa: B007
                        myrange = np.max(X, axis=0) - np.min(X, axis=0)  # noqa: F841
                        exec(  # noqa: S102
                            'm_tmp.'
                            + parname
                            + '[[nx]].constrain_bounded(myrange[nx] / X.shape[0]*10, myrange[nx],warning=False)'
                        )
                        exec(  # noqa: S102
                            'm_tmp.' + parname + '[[nx]] = myrange[nx]*1'
                        )

                        # m_tmp[parname][nx] = myrange[nx]*0.1
                        # m_tmp[parname][nx].constrain_bounded(myrange[nx] / X.shape[0], myrange[nx]*100)
                        # TODO change the kernel  # noqa: TD002, TD004
            # m_tmp[variance_keyword].constrain_bounded(0.05/np.mean(m_tmp.Y_metadata['variance_structure']),2/np.mean(m_tmp.Y_metadata['variance_structure']),warning=False)
            m_tmp.Gaussian_noise.constrain_bounded(
                0.5 * init_noise_var, 2 * init_noise_var, warning=False
            )
        else:
            msg = 'Nugget keyword not identified: ' + nugget_opt_tmp

    if do_mf:
        # TODO: is this right?  # noqa: TD002
        if nugget_opt_tmp == 'Optimize':
            m_tmp.gpy_model.mixed_noise.Gaussian_noise.unfix()
            m_tmp.gpy_model.mixed_noise.Gaussian_noise_1.unfix()

        elif nugget_opt_tmp == 'Fixed Values':
            # m_tmp.gpy_model.mixed_noise.Gaussian_noise.constrain_fixed(self.nuggetVal[ny])
            # m_tmp.gpy_model.mixed_noise.Gaussian_noise_1.constrain_fixed(self.nuggetVal[ny])
            msg = 'Currently Nugget Fixed Values option is not supported'
            # self.exit(msg)

        elif nugget_opt_tmp == 'Fixed Bounds':
            # m_tmp.gpy_model.mixed_noise.Gaussian_noise.constrain_bounded(self.nuggetVal[ny][0],
            #                                                                       self.nuggetVal[ny][1])
            # m_tmp.gpy_model.mixed_noise.Gaussian_noise_1.constrain_bounded(self.nuggetVal[ny][0],
            #                                                                         self.nuggetVal[ny][1])
            msg = 'Currently Nugget Fixed Bounds option is not supported'
            # self.exit(msg)

        elif nugget_opt_tmp == 'Zero':
            m_tmp.gpy_model.mixed_noise.Gaussian_noise.constrain_fixed(
                0, warning=False
            )
            m_tmp.gpy_model.mixed_noise.Gaussian_noise_1.constrain_fixed(
                0, warning=False
            )

    if msg == '':
        # m_tmp.optimize()
        # n=0;
        if not do_mf:
            # Here

            print('Calibrating final surrogate')  # noqa: T201
            m_tmp = my_optimize_restart(m_tmp, nopt)
  # noqa: RUF100, W293
        # if develop_mode:
        #     print(m_tmp)
        #     #print(m_tmp.rbf.lengthscale)
        #     tmp = m_tmp.predict(m_tmp.X)
        #     plt.title("Original Mean QoI")
        #     plt.scatter(m_tmp.Y, tmp[0],alpha=0.1)
        #     plt.scatter(m_tmp.Y, m_tmp.Y,alpha=0.1)
        #     plt.xlabel("exact")
        #     plt.ylabel("pred")
        #     plt.show()

        # m_tmp.optimize_restarts(
        #     num_restarts=nopt,
        #     parallel=is_paralle_opt_safe,
        #     num_processes=n_processor,
        #     verbose=True,
        # )
        else:
            m_tmp.gpy_model.optimize_restarts(
                num_restarts=nopt,
                parallel=is_paralle_opt_safe,
                num_processes=n_processor,
                verbose=False,
            )
        # print(m_tmp)  # noqa: RUF100, T201
        # while n+20 <= nopt:
        #     m_tmp.optimize_restarts(num_restarts=20)
        #     n = n+20
        # if not nopt==n:
        #     m_tmp.optimize_restarts(num_restarts=nopt-n)

        print(flush=True)  # noqa: T201

    return m_tmp, msg, ny


def my_optimize_restart(m, n_opt):  # noqa: D103
    init = time.time()
    n_sample = len(m.Y)
    idx = list(range(n_sample))
    n_batch = 700
    n_cluster = int(np.ceil(len(m.Y) / n_batch))
    n_batch = np.ceil(n_sample / n_cluster)
    random.shuffle(idx)
    X_full = m.X  # noqa: N806
    Y_full = m.Y  # noqa: N806

    log_likelihoods = np.zeros((n_cluster,))
    errors = np.zeros((n_cluster,))
    m_list = []
    for nc in range(n_cluster):
        inside_cluster = idx[
            int(n_batch * nc) : int(np.min([n_batch * (nc + 1), n_sample]))
        ]

        #
        # Testing if this works better for parallel run
        #
        @monkeypatch_method(GPy.core.GP)
        def subsample_XY(self, idx):  # noqa: N802, N803, RUF100
            if self.Y_metadata is not None:
                new_meta = self.Y_metadata
                new_meta['variance_structure'] = self.Y_metadata[
                    'variance_structure'
                ][idx]
                self.Y_metadata.update(new_meta)
                print('metadata_updated')  # noqa: T201
            self.set_XY(self.X[idx, :], self.Y[idx, :])

        @monkeypatch_method(GPy.core.GP)
        def set_XY2(self, X=None, Y=None, Y_metadata=None):  # noqa: N802, N803
            if Y_metadata is not None:
                if self.Y_metadata is None:
                    self.Y_metadata = Y_metadata
                else:
                    self.Y_metadata.update(Y_metadata)
                    print('metadata_updated')  # noqa: T201
            self.set_XY(X, Y)

        m_subset = m.copy()
        # m_subset.set_XY2(m_subset.X[inside_cluster,:],m_subset.Y[inside_cluster,:],{""})
        m_subset.subsample_XY(inside_cluster)
        # m_subset.optimize(max_f_eval=1000)
        m_subset.optimize_restarts(n_opt)
        variance1 = m_subset.normalizer.std**2
        # Option 1
        tmp_all = m_subset.predict(X_full)
        errors[nc] = np.linalg.norm(tmp_all[0][:, 0] - Y_full[:, 0])

        # Option 2
        m_subset.set_XY2(X_full, Y_full, m.Y_metadata)
        variance2 = m_subset.normalizer.std**2
        m_subset.Gaussian_noise.variance = (
            m_subset.Gaussian_noise.variance * variance1 / variance2
        )
        log_likelihoods[nc] = m_subset.log_likelihood()
        m_list += [copy.deepcopy(m_subset)]
        print(  # noqa: T201
            '  cluster {} among {} : logL {}'.format(  # noqa: UP032
                nc + 1, n_cluster, log_likelihoods[nc]
            )
        )

        # import matplotlib.pyplot as plt
        # tmp_all = m_subset.predict(X_full); plt.scatter(tmp_all[0][:, 0], Y_full[:, 0],alpha=0.1); plt.scatter(Y_full[:, 0], Y_full[:, 0],alpha=0.1);  plt.show()
        # tmp_subset = m_subset.predict(X_full[inside_cluster]); plt.scatter(tmp_subset[0][:, 0], Y_full[inside_cluster, 0],alpha=0.1); plt.scatter(Y_full[inside_cluster, 0], Y_full[inside_cluster, 0],alpha=0.1);  plt.show()

    # best_cluster = np.argmin(errors) # not capturing skedasticity
    best_cluster = np.argmax(log_likelihoods)
    m = m_list[best_cluster]
    # m.kern.parameters = kernels[best_cluster][0]
    # m.Gaussian_noise.parameters = kernels[best_cluster][1]
    # m.parameters_changed()
    # tmp = m.predict(X_full[inside_cluster]); plt.scatter(tmp[0][:, 0], Y_full[inside_cluster, 0]); plt.scatter(Y_full[inside_cluster, 0], Y_full[inside_cluster, 0]);  plt.show()
    print('Elapsed time: {:.2f} s'.format(time.time() - init))  # noqa: T201, UP032

    return m


def set_XY_indi(  # noqa: N802, D103
    m_tmp,
    ny,
    x_dim,
    X_hf,  # noqa: N803
    Y_hf,  # noqa: N803
    create_kernel,
    create_gpy_model,
    do_logtransform,
    predictStoMeans,  # noqa: N803
    set_normalizer,  # noqa: ARG001
):
    #
    # check if X dimension has changed...
    #
    x_current_dim = x_dim
    for parname in m_tmp.parameter_names():
        if parname.endswith('lengthscale'):
            exec('x_current_dim = len(m_tmp.' + parname + ')')  # noqa: S102

    if x_current_dim != X_hf.shape[1]:
        kr = create_kernel(X_hf.shape[1])
        X_dummy = np.zeros((1, X_hf.shape[1]))  # noqa: N806
        Y_dummy = np.zeros((1, 1))  # noqa: N806
        m_new = create_gpy_model(X_dummy, Y_dummy, kr)
        m_tmp = m_new.copy()
        # m_tmp.optimize()

    if do_logtransform:
        if np.min(Y_hf) < 0:
            raise 'Error running SimCenterUQ - Response contains negative values. Please uncheck the log-transform option in the UQ tab'  # noqa: B016
        Y_hfs = np.log(Y_hf)  # noqa: N806
    else:
        Y_hfs = Y_hf  # noqa: N806

    # Y_mean=Y_hfs[X_idx]
    # Y_mean1, nugget_mean1 = self.predictStoMeans(X_new, Y_mean)
    Y_mean1, nugget_mean1, initial_noise_variance, initial_process_variance = (  # noqa: N806
        predictStoMeans(X_hf, Y_hfs)
    )

    Y_metadata, m_var, norm_var_str = self.predictStoVars(  # noqa: F821, N806 # type: ignore
        X_hf, (Y_hfs - Y_mean1) ** 2, X_hf, Y_hfs, X_hf.shape[0]
    )
    m_tmp.set_XY2(X_hf, Y_hfs, Y_metadata=Y_metadata)

    m_var_list = m_var
    var_str = norm_var_str
    indices_unique = range(Y_hfs.shape[0])
    n_unique_hf = X_hf.shape[0]
    Y_mean = Y_hfs  # noqa: N806

    normMeans = 0  # noqa: N806
    normVars = 1  # noqa: N806

    return (
        ny,
        m_tmp,
        Y_mean,
        normMeans,
        normVars,
        m_var_list,
        var_str,
        indices_unique,
        n_unique_hf,
    )


def closest_node(x, X, ll):  # noqa: N803, D103
    X = np.asarray(X)  # noqa: N806
    deltas = X - x
    deltas_norm = np.zeros(deltas.shape)
    for nx in range(X.shape[1]):
        deltas_norm[:, nx] = deltas[:, nx] / ll[nx]
    dist_2 = np.einsum('ij,ij->i', deltas_norm, deltas_norm)  # square sum

    return np.argmin(dist_2)


def read_txt(text_dir, exit_fun):  # noqa: D103
    if not os.path.exists(text_dir):  # noqa: PTH110
        msg = 'Error: file does not exist: ' + text_dir
        exit_fun(msg)
    with open(text_dir) as f:  # noqa: PTH123
        # Iterate through the file until the table starts
        header_count = 0
        for line in f:
            if line.replace(' ', '').startswith('%'):
                header_count = header_count + 1
            else:
                break
                # print(line)
        try:
            with open(text_dir) as f:  # noqa: PTH123, PLW2901
                X = np.loadtxt(f, skiprows=header_count)  # noqa: N806
        except ValueError:
            with open(text_dir) as f:  # noqa: PTH123, PLW2901
                try:
                    X = np.genfromtxt(f, skip_header=header_count, delimiter=',')  # noqa: N806
                    X = np.atleast_2d(X)  # noqa: N806
                    # if there are extra delimiter, remove nan
                    if np.isnan(X[-1, -1]):
                        X = np.delete(X, -1, 1)  # noqa: N806
                    # X = np.loadtxt(f, skiprows=header_count, delimiter=',')
                except ValueError:
                    msg = 'Error: unsupported file format ' + text_dir
                    exit_fun(msg)
        if np.isnan(X).any():
            msg = (
                'Error: unsupported file format '
                + text_dir
                + '.\nThe header should have % character in front.'
            )
            exit_fun(msg)

    if X.ndim == 1:
        X = np.array([X]).transpose()  # noqa: N806

    return X


if __name__ == '__main__':
    main(sys.argv)

    sys.stderr.close()

    # try:
    #     main(sys.argv)
    #     open(os.path.join(os.getcwd(), errFileName ), 'w').close()
    # except Exception:
    #     f = open(os.path.join(os.getcwd(), errFileName ), 'w')
    #     traceback.print_exc(file=f)
    #     f.close()
    # exit(1)
