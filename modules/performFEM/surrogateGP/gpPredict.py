import json as json  # noqa: INP001, D100, PLC0414
import os
import pickle as pickle  # noqa: PLC0414
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.stats import lognorm, norm
from sklearn.linear_model import LinearRegression

try:
    moduleName = 'GPy'  # noqa: N816
    import GPy as GPy  # noqa: PLC0414
except:  # noqa: E722
    print(  # noqa: T201
        'Error running surrogate prediction - Failed to import module: Surrogate modeling module uses GPy python package which is facing a version compatibility issue at this moment (01.05.2024). To use the surrogate module, one needs to update manually the GPy version to 1.13. The instruction can be found in the the documentation: https://nheri-simcenter.github.io/quoFEM-Documentation/common/user_manual/usage/desktop/SimCenterUQSurrogate.html#lblsimsurrogate',
        file=sys.stderr,
    )
    exit(-1)  # noqa: PLR1722

try:
    moduleName = 'GPy'  # noqa: N816
    import GPy as GPy  # noqa: PLC0414

    moduleName = 'emukit'  # noqa: N816
    from emukit.multi_fidelity.convert_lists_to_array import (
        convert_x_list_to_array,
    )

    moduleName = 'Pandas'  # noqa: N816
    import pandas as pd

    error_tag = False  # global variable
except:  # noqa: E722
    error_tag = True
    print(  # noqa: T201
        'Error running surrogate prediction - Failed to import module:' + moduleName,
        file=sys.stderr,
    )
    exit(-1)  # noqa: PLR1722


# from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_xy_lists_to_arrays


def main(params_dir, surrogate_dir, json_dir, result_file, input_json):  # noqa: C901, D103, PLR0912, PLR0915
    global error_file  # noqa: PLW0602

    os_type = sys.platform.lower()
    run_type = 'runninglocal'

    #
    # create a log file
    #

    msg0 = os.path.basename(os.getcwd()) + ' : '  # noqa: PTH109, PTH119
    file_object = open('surrogateLog.log', 'a')  # noqa: SIM115, PTH123

    folderName = os.path.basename(os.getcwd())  # noqa: PTH109, PTH119, N806
    sampNum = folderName.split('.')[-1]  # noqa: N806

    #
    # read json -- current input file
    #

    def error_exit(msg):
        error_file.write(msg)  # local
        error_file.close()
        file_object.write(msg0 + msg)  # global file
        file_object.close()
        print(msg, file=sys.stderr)  # noqa: T201
        exit(-1)  # noqa: PLR1722

    def error_warning(msg):
        # error_file.write(msg)
        file_object.write(msg)
        # print(msg)

    if not os.path.exists(json_dir):  # noqa: PTH110
        msg = 'Error in surrogate prediction: File not found -' + json_dir
        error_exit(msg)

    with open(json_dir) as f:  # noqa: PTH123
        try:
            sur = json.load(f)
        except ValueError:
            msg = 'invalid json format: ' + json_dir
            error_exit(msg)

    isEEUQ = sur.get('isEEUQ', False)  # noqa: N806
    isWEUQ = sur.get('isWEUQ', False)  # noqa: N806
    isHydroUQ = sur.get('isHydroUQ', False)  # noqa: N806

    if isEEUQ or isWEUQ or isHydroUQ:
        dakota_path = 'sc_scInput.json'
    else:
        dakota_path = input_json

    try:
        with open(dakota_path) as f:  # current input file  # noqa: PTH123
            inp_tmp = json.load(f)
    except:  # noqa: E722
        try:
            # current input file
            with open('sc_inputRWHALE.json') as f:  # noqa: PTH123
                inp_tmp = json.load(f)
        except:  # noqa: S110, E722
            pass

    try:
        if isEEUQ or isWEUQ or isHydroUQ:
            inp_fem = inp_tmp['Applications']['Modeling']
        else:
            inp_fem = inp_tmp['FEM']
    except:  # noqa: E722
        inp_fem = {}
        print('invalid json format - dakota.json')  # noqa: T201

    norm_var_thr = inp_fem.get('varThres', 0.02)
    when_inaccurate = inp_fem.get('femOption', 'continue')
    do_mf = False  # noqa: F841
    myseed = inp_fem.get('gpSeed', None)
    prediction_option = inp_fem.get('predictionOption', 'random')
    if myseed == None:  # noqa: E711
        try:
            myseed = inp_tmp['UQ']['samplingMethodData']['seed']
        except:  # noqa: E722
            myseed = 42
        # folderName = os.path.basename(os.path.dirname(os.getcwd()))  # noqa: N806, PTH109, PTH119, PTH120, RUF100
        # myseed = int(folderName) * int(1.0e7)

    np.random.seed(int(myseed) + int(sampNum))

    # if no g and rv,

    #
    # read json -- original input for training surrogate
    #

    f.close()

    did_stochastic = sur['doStochastic']
    did_logtransform = sur['doLogtransform']
    did_normalization = sur['doNormalization']
    kernel = sur['kernName']

    if kernel == 'Radial Basis':
        kern_name = 'rbf'
    elif kernel == 'Exponential':
        kern_name = 'Exponential'
    elif kernel == 'Matern 3/2':
        kern_name = 'Mat32'
    elif kernel == 'Matern 5/2':
        kern_name = 'Mat52'  # noqa: F841
    did_mf = sur['doMultiFidelity']

    # from json
    g_name_sur = list()  # noqa: C408
    ng_sur = 0
    Y = np.zeros((sur['highFidelityInfo']['valSamp'], sur['ydim']))  # noqa: N806
    for g in sur['ylabels']:
        g_name_sur += [g]
        Y[:, ng_sur] = np.array(sur['yExact'][g])
        ng_sur += 1

    rv_name_sur = list()  # noqa: C408
    nrv_sur = 0
    X = np.zeros((sur['highFidelityInfo']['valSamp'], sur['xdim']))  # noqa: N806
    for rv in sur['xlabels']:
        rv_name_sur += [rv]
        X[:, nrv_sur] = np.array(sur['xExact'][rv])
        nrv_sur += 1

    try:
        constIdx = sur['highFidelityInfo']['constIdx']  # noqa: N806
        constVal = sur['highFidelityInfo']['constVal']  # noqa: N806
    except:  # noqa: E722
        constIdx = []  # noqa: N806
        constVal = []  # noqa: N806

        # Read pickles

    if did_stochastic:
        #
        # Modify GPy package
        #
        def monkeypatch_method(cls):
            def decorator(func):
                setattr(cls, func.__name__, func)
                return func

            return decorator

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
                    # print("metadata_updated")

            self.set_XY(X, Y)

        def get_stochastic_variance(X, Y, x, ny):  # noqa: C901, N803
            # X_unique, X_idx, indices, counts = np.unique(X, axis=0, return_index=True, return_counts=True, return_inverse=True)
            X_unique, dummy, indices, counts = np.unique(  # noqa: N806
                X, axis=0, return_index=True, return_counts=True, return_inverse=True
            )

            idx_repl = [i for i in np.where(counts > 1)[0]]  # noqa: C416

            if len(idx_repl) > 0:
                n_unique = X_unique.shape[0]
                Y_mean, Y_var = np.zeros((n_unique, 1)), np.zeros((n_unique, 1))  # noqa: N806

                for idx in range(n_unique):
                    Y_subset = Y[[i for i in np.where(indices == idx)[0]], :]  # noqa: C416, N806
                    Y_mean[idx, :] = np.mean(Y_subset, axis=0)
                    Y_var[idx, :] = np.var(Y_subset, axis=0)

                if (np.max(Y_var) / np.var(Y_mean) < 1.0e-10) and len(idx_repl) > 0:  # noqa: PLR2004
                    return np.ones((X.shape[0], 1))

                # kernel_var = GPy.kern.Matern52(
                #    input_dim=nrv_sur, ARD=True
                # ) + GPy.kern.Linear(input_dim=nrv_sur, ARD=True)
                kernel_var = GPy.kern.Matern52(input_dim=nrv_sur, ARD=True)
                log_vars = np.log(Y_var[idx_repl])
                m_var = GPy.models.GPRegression(
                    X_unique[idx_repl, :],
                    log_vars,
                    kernel_var,
                    normalizer=True,
                    Y_metadata=None,
                )
                # print("Collecting variance field of ny={}".format(ny))
                for key, val in sur['modelInfo'][g_name_sur[ny] + '_Var'].items():  # noqa: B007, PERF102
                    exec('m_var.' + key + '= np.array(val)')  # noqa: S102

                log_var_pred, dum = m_var.predict(X_unique)
                var_pred = np.exp(log_var_pred)

                if did_normalization:
                    # Y_normFact = np.var(Y_mean)  # noqa: N806, RUF100
                    Y_normFact = np.mean(var_pred.T[0])  # noqa: N806
                else:
                    Y_normFact = 1  # noqa: N806

                norm_var_str = (
                    (var_pred.T[0]) / Y_normFact
                )  # if normalization was used..

                log_var_pred_x, dum = m_var.predict(x)
                nugget_var_pred_x = np.exp(log_var_pred_x.T[0]) / Y_normFact

            else:
                X_unique = X  # noqa: N806
                Y_mean = Y  # noqa: N806
                indices = range(Y.shape[0])

                #
                # check if we have an old example file - to be deleted in the future
                #
                old_version = False
                for key, val in sur['modelInfo'][g_name_sur[ny] + '_Var'].items():  # noqa: B007, PERF102
                    if 'sum' in key:
                        old_version = True
                        break

                if old_version:
                    print(  # noqa: T201
                        'The surrogate model was trained using an older version of the tool. Please retrain the model using this version or use older version.',
                        file=sys.stderr,
                    )
                    exit(-1)  # noqa: PLR1722

                log_vars = np.atleast_2d(
                    sur['modelInfo'][g_name_sur[ny] + '_Var']['TrainingSamplesY']
                ).T

                kernel_var = GPy.kern.Matern52(input_dim=nrv_sur, ARD=True)

                m_var = GPy.models.GPRegression(
                    X, log_vars, kernel_var, normalizer=True, Y_metadata=None
                )

                # print("Variance field obtained for ny={}".format(ny))
                for key, val in sur['modelInfo'][g_name_sur[ny] + '_Var'].items():  # noqa: B007, PERF102
                    exec('m_var.' + key + '= np.array(val)')  # noqa: S102

                log_var_pred, dum = m_var.predict(X)
                var_pred = np.exp(log_var_pred)

                if did_normalization:
                    # Y_normFact = np.var(Y)  # noqa: N806, RUF100
                    Y_normFact = np.mean(var_pred.T[0])  # noqa: N806

                else:
                    Y_normFact = 1  # noqa: N806

                norm_var_str = (
                    (var_pred.T[0]) / Y_normFact
                )  # if normalization was used..

                log_var_pred_x, dum = m_var.predict(x)
                nugget_var_pred_x = np.exp(log_var_pred_x.T[0]) / Y_normFact

            return (
                X_unique,
                Y_mean,
                norm_var_str,
                counts,
                nugget_var_pred_x,
                np.var(Y_mean),
            )

    # REQUIRED: rv_name, y_var

    # Collect also dummy rvs
    id_vec = []
    rv_name_dummy = []

    t_total = time.process_time()  # noqa: F841
    first_rv_found = False
    first_dummy_found = False

    #
    # Check how many RVs overlap
    #

    with open(params_dir) as x_file:  # noqa: PTH123
        data = x_file.readlines()
        nrv = int(data[0])
        for i in range(nrv):
            name_values = data[i + 1].split()
            name = name_values[0]
            # print(name)

            # = pass if is string. GP cannot handle that
            if ((name == 'MultipleEvent') or (name == 'eventID')) and isEEUQ:  # noqa: PLR1714
                continue

            if (
                not name_values[1]
                .replace('.', '', 1)
                .replace('e', '', 1)
                .replace('-', '', 2)
                .replace('+', '', 1)
                .isdigit()
            ):
                # surrogate model does not accept discrete
                continue

            # = atleast_2d because there may be multiple samples
            samples = np.atleast_2d([float(vals) for vals in name_values[1:]]).T
            ns = len(samples)
            if name not in rv_name_sur:
                rv_name_dummy += [name]
                if not first_dummy_found:
                    rv_val_dummy = samples
                    first_dummy_found = True
                else:
                    rv_val_dummy = np.hstack([rv_val_dummy, samples])
                continue

            id_map = rv_name_sur.index(name)
            # print(name)
            # print(rv_name_sur)
            # try:
            #    id_map = rv_name_sur.index(name)
            # except ValueError:
            #    msg = 'Error importing input data: variable "{}" not identified.'.format(name)
            #    error_exit(msg)

            if not first_rv_found:
                nsamp = ns
                rv_tmp = samples
                id_vec = [id_map]
                first_rv_found = True
            else:
                rv_tmp = np.hstack([rv_tmp, samples])
                id_vec += [id_map]

            if ns != nsamp:
                msg = 'Error importing input data to surrogate: sample size in params.in is not consistent.'
                error_exit(msg)

        g_idx = []
        for edp in inp_tmp['EDP']:
            edp_names = []
            if edp['length'] == 1:
                edp_names += [edp['name']]
            else:
                for i in range(edp['length']):
                    edp_names += [edp['name'] + '_' + str(i + 1)]
            try:
                for i in range(edp['length']):
                    id_map = g_name_sur.index(edp_names[i])
                    g_idx += [id_map]
            except ValueError:
                msg = 'Error importing input data to surrogate: qoi "{}" not identified.'.format(
                    edp['name']
                )
                error_exit(msg)

    # if eeuq
    first_eeuq_found = False
    if sur.get('intensityMeasureInfo') != None:  # noqa: E711
        with open('IMinput.json', 'w') as f:  # noqa: PTH123
            mySurrogateJson = sur['intensityMeasureInfo']  # noqa: N806
            json.dump(mySurrogateJson, f)

        computeIM = os.path.join(  # noqa: PTH118, N806
            os.path.dirname(  # noqa: PTH120
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # noqa: PTH100, PTH120
            ),
            'createEVENT',
            'groundMotionIM',
            'IntensityMeasureComputer.py',
        )

        pythonEXE = sys.executable  # noqa: N806
        # compute IMs

        if os.path.exists('EVENT.json') and os.path.exists('IMinput.json'):  # noqa: PTH110
            os.system(  # noqa: S605
                f'{pythonEXE} {computeIM} --filenameAIM IMinput.json --filenameEVENT EVENT.json --filenameIM IM.json --geoMeanVar'
            )
        else:
            msg = f'IMinput.json and EVENT.json not found in workdir.{sampNum}. Cannot calculate IMs.'
            error_exit(msg)

        first_eeuq_found = False
        if os.path.exists('IM.csv'):  # noqa: PTH110
            # print("IM.csv found")
            tmp1 = pd.read_csv(('IM.csv'), index_col=None)
            if tmp1.empty:
                # print("IM.csv in wordir.{} is empty.".format(cur_id))
                return

            IMnames = list(map(str, tmp1))  # noqa: N806
            IMvals = tmp1.to_numpy()  # noqa: N806
            nrv2 = len(IMnames)
            for i in range(nrv2):
                name = IMnames[i]
                samples = np.atleast_2d(IMvals[:, i])
                ns = len(samples)
                try:
                    id_map = rv_name_sur.index(name)
                except ValueError:
                    msg = f'Error importing input data to surrogate: variable "{name}" not identified.'
                    error_exit(msg)

                if not first_eeuq_found:
                    nsamp = ns
                    rv_tmp2 = samples
                    id_vec2 = [id_map]
                    first_eeuq_found = True
                else:
                    rv_tmp2 = np.hstack([rv_tmp2, samples])
                    id_vec2 += [id_map]

                if ns != nsamp:
                    msg = 'Error importing input data to surrogate: sample size in params.in is not consistent.'
                    error_exit(msg)
        # TODO: fix for different nys m  # noqa: TD002

        if len(id_vec + id_vec2) != nrv_sur:
            missing_ids = set([i for i in range(len(rv_name_sur))]) - set(  # noqa: C403, C416
                id_vec + id_vec2
            )
            s = [str(rv_name_sur[id]) for id in missing_ids]  # noqa: A001

            if first_eeuq_found and all(
                [missingEDP.endswith('-2') for missingEDP in s]  # noqa: C419
            ):
                msg = 'ground motion dimension does not match with that of the training'
                # for i in range(len(s)):
                #     name = s[i]
                #     samples = np.zeros((1,nsamp))
                #     try:
                #         id_map = rv_name_sur.index(name)
                #     except ValueError:
                #         msg = 'Error importing input data: variable "{}" not identified.'.format(name)
                #         error_exit(msg)
                #     rv_tmp2 = np.hstack([rv_tmp2, samples])
                #     id_vec2 += [id_map]
                error_exit(msg)

    if first_eeuq_found:
        if first_rv_found:
            rv_tmp = np.hstack([rv_tmp, rv_tmp2])
            id_vec = id_vec + id_vec2
        else:
            rv_tmp = np.hstack([rv_tmp2])
            id_vec = id_vec2

    nrv = len(id_vec)
    if nrv != nrv_sur:
        # missing_ids = set([i for i in range(len(rv_name_sur))]) - set(id_vec)
        missing_ids = set([i for i in range(len(rv_name_sur))]).difference(  # noqa: C403, C416
            set(id_vec)
        )
        # print(missing_ids)
        s = [str(rv_name_sur[id]) for id in missing_ids]  # noqa: A001
        msg = 'Error in Surrogate prediction: Number of dimension inconsistent: Please define '
        msg += ', '.join(s)
        msg += ' at RV tab'
        error_exit(msg)

    # if os.path.getsize('../surrogateLog.log') == 0:
    #    file_object.write("numRV "+ str(nrv+len(rv_name_dummy)) +"\n")

    rv_val = np.zeros((nsamp, nrv))
    for i in range(nrv):
        rv_val[:, id_vec[i]] = rv_tmp[:, i]

    if kernel == 'Radial Basis':
        kr = GPy.kern.RBF(input_dim=nrv_sur, ARD=True)
    elif kernel == 'Exponential':
        kr = GPy.kern.Exponential(input_dim=nrv_sur, ARD=True)
    elif kernel == 'Matern 3/2':
        kr = GPy.kern.Matern32(input_dim=nrv_sur, ARD=True)
    elif kernel == 'Matern 5/2':
        kr = GPy.kern.Matern52(input_dim=nrv_sur, ARD=True)

    if sur['doLinear']:
        # kr = kr + GPy.kern.Linear(input_dim=nrv_sur, ARD=True)
        did_linear = True
        lin_index = [True] * nrv
        lin_list = []
        for ny in range(ng_sur):
            tmp_lin = LinearRegression()
            tmp_lin.coef_ = np.array(
                sur['modelInfo'][g_name_sur[ny] + '_Lin']['coef']
            )
            tmp_lin.intercept_ = np.array(
                sur['modelInfo'][g_name_sur[ny] + '_Lin']['intercept']
            )
            lin_list += [tmp_lin]
    else:
        did_linear = False

    # preprocessing..

    if did_logtransform:
        Y = np.log(Y)  # noqa: N806

    if did_linear:
        for ny in range(ng_sur):
            y_lin_pred = lin_list[ny].predict(X[:, lin_index])
            Y[:, ny] = Y[:, ny] - y_lin_pred

    kg = kr
    m_list = list()  # noqa: C408
    nugget_var_list = [0] * ng_sur

    if not did_mf:
        for ny in range(ng_sur):
            if did_stochastic[ny]:
                m_list = m_list + [  # noqa: RUF005
                    GPy.models.GPRegression(
                        X,
                        Y[:, ny][np.newaxis].transpose(),
                        kernel=kg.copy(),
                        normalizer=did_normalization,
                    )
                ]
                (
                    X_unique,  # noqa: N806
                    Y_mean,  # noqa: N806
                    norm_var_str,
                    counts,
                    nugget_var_pred,
                    Y_normFact,  # noqa: N806
                ) = get_stochastic_variance(X, Y[:, ny][np.newaxis].T, rv_val, ny)
                Y_metadata = {'variance_structure': norm_var_str / counts}  # noqa: N806
                m_list[ny].set_XY2(X_unique, Y_mean, Y_metadata=Y_metadata)
                for key, val in sur['modelInfo'][g_name_sur[ny]].items():  # noqa: B007, PERF102
                    exec('m_list[ny].' + key + '= np.array(val)')  # noqa: S102

                nugget_var_list[ny] = (
                    m_list[ny].Gaussian_noise.parameters
                    * nugget_var_pred
                    * Y_normFact
                )

            else:
                m_list = m_list + [  # noqa: RUF005
                    GPy.models.GPRegression(
                        X,
                        Y[:, ny][np.newaxis].transpose(),
                        kernel=kg.copy(),
                        normalizer=True,
                    )
                ]
                for key, val in sur['modelInfo'][g_name_sur[ny]].items():  # noqa: B007, PERF102
                    exec('m_list[ny].' + key + '= np.array(val)')  # noqa: S102

                Y_normFact = np.var(Y[:, ny])  # noqa: N806
                nugget_var_list[ny] = np.squeeze(
                    np.array(m_list[ny].Gaussian_noise.parameters)
                    * np.array(Y_normFact)
                )

    else:
        with open(surrogate_dir, 'rb') as file:  # noqa: PTH123
            m_list = pickle.load(file)  # noqa: S301

        for ny in range(ng_sur):
            Y_normFact = np.var(Y[:, ny])  # noqa: N806
            nugget_var_list[ny] = (
                m_list[ny].gpy_model['mixed_noise.Gaussian_noise.variance']
                * Y_normFact
            )

    # read param in file and sort input
    y_dim = len(m_list)

    y_pred_median = np.zeros([nsamp, y_dim])
    y_pred_var_tmp = np.zeros([nsamp, y_dim])  # might be log space
    y_pred_var_m_tmp = np.zeros([nsamp, y_dim])  # might be log space

    y_pred_var = np.zeros([nsamp, y_dim])
    y_pred_var_m = np.zeros([nsamp, y_dim])

    y_data_var = np.zeros([nsamp, y_dim])
    y_samp = np.zeros([nsamp, y_dim])
    y_q1 = np.zeros([nsamp, y_dim])
    y_q3 = np.zeros([nsamp, y_dim])
    y_q1m = np.zeros([nsamp, y_dim])
    y_q3m = np.zeros([nsamp, y_dim])

    for ny in range(y_dim):
        y_data_var[:, ny] = np.var(m_list[ny].Y)
        if ny in constIdx:
            y_pred_median_tmp, y_pred_var_tmp[ny], y_pred_var_m_tmp[ny] = (
                np.ones([nsamp]) * constVal[constIdx.index(ny)],
                np.zeros([nsamp]),
                np.zeros([nsamp]),
            )
        else:
            y_pred_median_tmp, y_pred_var_tmp_tmp = predict(
                m_list[ny], rv_val, did_mf
            )  # noiseless
            y_pred_median_tmp = np.squeeze(y_pred_median_tmp)
            y_pred_var_tmp_tmp = np.squeeze(y_pred_var_tmp_tmp)

            if did_linear:
                y_lin_pred = lin_list[ny].predict(rv_val[:, lin_index])
                y_pred_median_tmp = y_pred_median_tmp + y_lin_pred

        y_pred_var_tmp[:, ny] = y_pred_var_tmp_tmp
        y_pred_var_m_tmp[:, ny] = y_pred_var_tmp_tmp + np.squeeze(
            nugget_var_list[ny]
        )
        y_samp_tmp = np.random.normal(
            y_pred_median_tmp, np.sqrt(y_pred_var_m_tmp[:, ny])
        )

        if did_logtransform:
            y_pred_median[:, ny] = np.exp(y_pred_median_tmp)
            y_pred_var[:, ny] = np.exp(
                2 * y_pred_median_tmp + y_pred_var_tmp[:, ny]
            ) * (np.exp(y_pred_var_tmp[:, ny]) - 1)
            y_pred_var_m[:, ny] = np.exp(
                2 * y_pred_median_tmp + y_pred_var_m_tmp[:, ny]
            ) * (np.exp(y_pred_var_m_tmp[:, ny]) - 1)

            y_samp[:, ny] = np.exp(y_samp_tmp)

            y_q1[:, ny] = lognorm.ppf(
                0.05,
                s=np.sqrt(y_pred_var_tmp[:, ny]),
                scale=np.exp(y_pred_median_tmp),
            )
            y_q3[:, ny] = lognorm.ppf(
                0.95,
                s=np.sqrt(y_pred_var_tmp[:, ny]),
                scale=np.exp(y_pred_median_tmp),
            )
            y_q1m[:, ny] = lognorm.ppf(
                0.05,
                s=np.sqrt(y_pred_var_m_tmp[:, ny]),
                scale=np.exp(y_pred_median_tmp),
            )
            y_q3m[:, ny] = lognorm.ppf(
                0.95,
                s=np.sqrt(y_pred_var_m_tmp[:, ny]),
                scale=np.exp(y_pred_median_tmp),
            )

        else:
            y_pred_median[:, ny] = y_pred_median_tmp
            y_pred_var[:, ny] = y_pred_var_tmp[:, ny]
            y_pred_var_m[:, ny] = y_pred_var_m_tmp[:, ny]
            y_samp[:, ny] = y_samp_tmp
            y_q1[:, ny] = norm.ppf(
                0.05, loc=y_pred_median_tmp, scale=np.sqrt(y_pred_var_tmp[:, ny])
            )
            y_q3[:, ny] = norm.ppf(
                0.95, loc=y_pred_median_tmp, scale=np.sqrt(y_pred_var_tmp[:, ny])
            )
            y_q1m[:, ny] = norm.ppf(
                0.05, loc=y_pred_median_tmp, scale=np.sqrt(y_pred_var_m_tmp[:, ny])
            )
            y_q3m[:, ny] = norm.ppf(
                0.95, loc=y_pred_median_tmp, scale=np.sqrt(y_pred_var_m_tmp[:, ny])
            )

        if np.isnan(y_samp[:, ny]).any():
            y_samp[:, ny] = np.nan_to_num(y_samp[:, ny])
        if np.isnan(y_pred_var[:, ny]).any():
            y_pred_var[:, ny] = np.nan_to_num(y_pred_var[:, ny])
        if np.isnan(y_pred_var_m[:, ny]).any():
            y_pred_m_var[:, ny] = np.nan_to_num(y_pred_m_var[:, ny])  # noqa: F821

        # for parname in m_list[ny].parameter_names():
        #    if (kern_name in parname) and parname.endswith('variance'):
        #        exec('y_pred_prior_var[ny]=m_list[ny].' + parname)

    # error_ratio1 = y_pred_var.T / y_pred_prior_var
    error_ratio2 = y_pred_var_m_tmp / y_data_var
    idx = np.argmax(error_ratio2, axis=1) + 1

    """
    if np.max(error_ratio1) > norm_var_thr:

        is_accurate = False
        idx = np.argmax(error_ratio1) + 1

        msg = 'Prediction error of output {} is {:.2f}%, which is greater than threshold={:.2f}%  '.format(idx, np.max(
            error_ratio1)*100, norm_var_thr*100)
    """

    is_accurate_array = np.max(error_ratio2, axis=1) < norm_var_thr

    y_pred_subset = np.zeros([nsamp, len(g_idx)])
    msg1 = []
    for ns in range(nsamp):
        msg0 = folderName.split('.')[0] + '.' + str(int(sampNum) + ns) + ' : '

        if not is_accurate_array[ns]:
            msg1 += [
                f'Prediction error level of output {idx[ns]} is {np.max(error_ratio2[ns]) * 100:.2f}%, which is greater than threshold={norm_var_thr * 100:.2f}%  '
            ]
        else:
            msg1 += ['']

        if not is_accurate_array[ns]:
            if when_inaccurate == 'doSimulation':
                #
                # (1) create "workdir.idx " folder :need C++17 to use the files system namespace
                #
                templatedirFolder = os.path.join(os.getcwd(), 'templatedir_SIM')  # noqa: PTH109, PTH118, N806

                if (
                    (isEEUQ or isWEUQ or isHydroUQ) and nsamp == 1
                ):  # because stochastic ground motion generation uses folder number when generating random seed.............
                    current_dir_i = os.path.join(  # noqa: PTH118
                        os.getcwd(),  # noqa: PTH109
                        f'subworkdir.{sampNum}',
                    )
                else:
                    current_dir_i = os.path.join(os.getcwd(), f'subworkdir.{1 + ns}')  # noqa: PTH109, PTH118

                try:
                    shutil.copytree(templatedirFolder, current_dir_i)
                except Exception:  # noqa: BLE001
                    try:
                        shutil.copytree(templatedirFolder, current_dir_i)
                    except Exception as ex:  # noqa: BLE001
                        msg = 'Error running FEM: ' + str(ex)

                # change directory, create params.in
                if isEEUQ or isWEUQ or isHydroUQ:
                    shutil.copyfile(
                        os.path.join(os.getcwd(), 'params.in'),  # noqa: PTH109, PTH118
                        os.path.join(current_dir_i, 'params.in'),  # noqa: PTH118
                    )
                    shutil.copyfile(
                        os.path.join(os.getcwd(), 'EVENT.json.sc'),  # noqa: PTH109, PTH118
                        os.path.join(current_dir_i, 'EVENT.json.sc'),  # noqa: PTH118
                    )

                    #
                    # Replace parts of AIM
                    #
                    with open(os.path.join(current_dir_i, 'AIM.json.sc')) as f:  # noqa: PTH118, PTH123
                        try:
                            AIMsc = json.load(f)  # noqa: N806
                        except ValueError:
                            msg = 'invalid AIM in template. Simulation of original model cannot be perfomred'
                            error_exit(msg)
                    AIMsc['Events'] = inp_tmp['Events']
                    AIMsc['Applications']['Events'] = inp_tmp['Applications'][
                        'Events'
                    ]
                    with open(os.path.join(current_dir_i, 'AIM.json.sc'), 'w') as f:  # noqa: PTH118, PTH123
                        json.dump(AIMsc, f, indent=2)

                    #
                    # Copy PEER RECORDS
                    #
                    for fname in os.listdir(current_dir_i):
                        if fname.startswith('PEER-Record-'):
                            os.remove(os.path.join(current_dir_i, fname))  # noqa: PTH107, PTH118
                        if fname.startswith('RSN') and fname.endswith('AT2'):
                            os.remove(os.path.join(current_dir_i, fname))  # noqa: PTH107, PTH118

                    for fname in os.listdir(os.getcwd()):  # noqa: PTH109
                        if fname.startswith('PEER-Record-'):
                            shutil.copyfile(
                                os.path.join(os.getcwd(), fname),  # noqa: PTH109, PTH118
                                os.path.join(current_dir_i, fname),  # noqa: PTH118
                            )

                    #
                    # Replace parts of drive
                    #

                    if os_type.startswith('win'):
                        driver_name = 'driver.bat'
                    else:
                        driver_name = 'driver'

                    with open(os.path.join(os.getcwd(), driver_name)) as f:  # noqa: PTH109, PTH118, PTH123
                        event_driver = f.readline()

                    with open(os.path.join(current_dir_i, driver_name), 'r+') as f:  # noqa: PTH118, PTH123
                        # Read the original contents of the file
                        contents = f.readlines()
                        # Modify the first line
                        contents[0] = event_driver
                        # Truncate the file
                        f.seek(0)
                        f.truncate()
                        # Write the modified contents to the file
                        f.writelines(contents)

                else:
                    outF = open(current_dir_i + '/params.in', 'w')  # noqa: SIM115, PTH123, N806
                    outF.write(f'{nrv}\n')
                    for i in range(nrv):
                        outF.write(f'{rv_name_sur[i]} {rv_val[ns, i]}\n')
                    outF.close()

                os.chdir(current_dir_i)

                # run workflowDriver

                if isEEUQ or isWEUQ or isHydroUQ:
                    if (
                        os_type.lower().startswith('win')
                        and run_type.lower() == 'runninglocal'
                    ):
                        workflowDriver = 'sc_driver.bat'  # noqa: N806
                    else:
                        workflowDriver = 'sc_driver'  # noqa: N806
                elif (
                    os_type.lower().startswith('win')
                    and run_type.lower() == 'runninglocal'
                ):
                    workflowDriver = 'driver.bat'  # noqa: N806
                else:
                    workflowDriver = 'driver'  # noqa: N806

                workflow_run_command = f'{current_dir_i}/{workflowDriver}'
                subprocess.Popen(workflow_run_command, shell=True).wait()  # noqa: S602

                # back to directory, copy result.out
                # shutil.copyfile(os.path.join(sim_dir, 'results.out'), os.path.join(os.getcwd(), 'results.out'))

                with open('results.out') as f:  # noqa: PTH123
                    y_pred = np.array([np.loadtxt(f)]).flatten()
                    y_pred_subset[ns, :] = y_pred[g_idx]

                os.chdir('../')

                msg2 = msg0 + msg1[ns] + '- RUN original model\n'
                error_warning(msg2)
                # exit(-1)

            elif when_inaccurate == 'giveError':
                msg2 = msg0 + msg1[ns] + '- EXIT\n'
                error_exit(msg2)

            elif when_inaccurate == 'continue':
                msg2 = (
                    msg0
                    + msg1[ns]
                    + '- CONTINUE [Warning: results may not be accurate]\n'
                )
                error_warning(msg2)

                if prediction_option.lower().startswith('median'):
                    y_pred_subset[ns, :] = y_pred_median[ns, g_idx]
                elif prediction_option.lower().startswith('rand'):
                    y_pred_subset[ns, :] = y_samp[ns, g_idx]

        else:
            msg3 = (
                msg0
                + f'Prediction error level of output {idx[ns]} is {np.max(error_ratio2[ns]) * 100:.2f}%\n'
            )
            error_warning(msg3)

            if prediction_option.lower().startswith('median'):
                y_pred_subset[ns, :] = y_pred_median[ns, g_idx]
            elif prediction_option.lower().startswith('rand'):
                y_pred_subset[ns, :] = y_samp[ns, g_idx]

    with open(result_file, 'w') as f:  # noqa: PTH123
        np.savetxt(f, y_pred_subset, fmt='%.5e')

    y_pred_median_subset = y_pred_median[:, g_idx]
    y_q1_subset = y_q1[:, g_idx]
    y_q3_subset = y_q3[:, g_idx]
    y_q1m_subset = y_q1m[:, g_idx]
    y_q3m_subset = y_q3m[:, g_idx]
    y_pred_var_subset = y_pred_var[:, g_idx]
    y_pred_var_m_subset = y_pred_var_m[:, g_idx]

    #
    # tab file
    #

    #
    # Add dummy RVs
    #
    if first_dummy_found:
        rv_name_sur = rv_name_sur + rv_name_dummy
        rv_val = np.hstack([rv_val, rv_val_dummy])

    g_name_subset = [g_name_sur[i] for i in g_idx]

    if int(sampNum) == 1:
        with open('../surrogateTabHeader.out', 'w') as header_file:  # noqa: PTH123
            # write header
            # if os.path.getsize('../surrogateTab.out') == 0:
            header_file.write(
                '%eval_id interface '
                + ' '.join(rv_name_sur)
                + ' '
                + ' '.join(g_name_subset)
                + ' '
                + '.median '.join(g_name_subset)
                + '.median '
                + '.q5 '.join(g_name_subset)
                + '.q5 '
                + '.q95 '.join(g_name_subset)
                + '.q95 '
                + '.var '.join(g_name_subset)
                + '.var '
                + '.q5_w_mnoise '.join(g_name_subset)
                + '.q5_w_mnoise '
                + '.q95_w_mnoise '.join(g_name_subset)
                + '.q95_w_mnoise '
                + '.var_w_mnoise '.join(g_name_subset)
                + '.var_w_mnoise \n'
            )
            # write values

    with open('../surrogateTab.out', 'a') as tab_file:  # noqa: PTH123
        # write header
        # if os.path.getsize('../surrogateTab.out') == 0:
        #    tab_file.write("%eval_id interface "+ " ".join(rv_name_sur) + " "+ " ".join(g_name_subset) + " " + ".median ".join(g_name_subset) + ".median "+ ".q5 ".join(g_name_subset) + ".q5 "+ ".q95 ".join(g_name_subset) + ".q95 " +".var ".join(g_name_subset) + ".var " + ".q5_w_mnoise ".join(g_name_subset) + ".q5_w_mnoise "+ ".q95_w_mnoise ".join(g_name_subset) + ".q95_w_mnoise " +".var_w_mnoise ".join(g_name_subset) + ".var_w_mnoise \n")
        # write values

        for ns in range(nsamp):
            rv_list = ' '.join(f'{rv:e}' for rv in rv_val[ns, :])
            ypred_list = ' '.join(f'{yp:e}' for yp in y_pred_subset[ns, :])
            ymedian_list = ' '.join(f'{ym:e}' for ym in y_pred_median_subset[ns, :])
            yQ1_list = ' '.join(f'{yq1:e}' for yq1 in y_q1_subset[ns, :])  # noqa: N806
            yQ3_list = ' '.join(f'{yq3:e}' for yq3 in y_q3_subset[ns, :])  # noqa: N806
            ypredvar_list = ' '.join(f'{ypv:e}' for ypv in y_pred_var_subset[ns, :])
            yQ1m_list = ' '.join(f'{yq1:e}' for yq1 in y_q1m_subset[ns, :])  # noqa: N806
            yQ3m_list = ' '.join(f'{yq3:e}' for yq3 in y_q3m_subset[ns, :])  # noqa: N806
            ypredvarm_list = ' '.join(
                f'{ypv:e}' for ypv in y_pred_var_m_subset[ns, :]
            )

            tab_file.write(
                str(int(sampNum) + ns)
                + ' NO_ID '
                + rv_list
                + ' '
                + ypred_list
                + ' '
                + ymedian_list
                + ' '
                + yQ1_list
                + ' '
                + yQ3_list
                + ' '
                + ypredvar_list
                + ' '
                + yQ1m_list
                + ' '
                + yQ3m_list
                + ' '
                + ypredvarm_list
                + ' \n'
            )

    error_file.close()
    file_object.close()


def predict(m, X, did_mf):  # noqa: N803, D103
    if not did_mf:
        return m.predict_noiseless(X)
    else:  # noqa: RET505
        # TODO change below to noiseless  # noqa: TD002, TD004
        X_list = convert_x_list_to_array([X, X])  # noqa: N806
        X_list_l = X_list[: X.shape[0]]  # noqa: N806, F841
        X_list_h = X_list[X.shape[0] :]  # noqa: N806
        return m.predict(X_list_h)


if __name__ == '__main__':
    error_file = open('../surrogate.err', 'w')  # noqa: SIM115, PTH123
    inputArgs = sys.argv  # noqa: N816

    if not inputArgs[2].endswith('.json'):
        msg = 'ERROR: surrogate information file (.json) not set'
        error_file.write(msg)
        exit(-1)  # noqa: PLR1722

    # elif not inputArgs[3].endswith('.pkl'):
    #     msg = 'ERROR: surrogate model file (.pkl) not set'
    #     print(msg); error_file.write(msg); exit(-1)

    # elif len(inputArgs) < 4 or len(inputArgs) > 4:
    #     msg = 'ERROR: put right number of argv'
    #     print(msg); error_file.write(msg); exit(-1)

    """
    params_dir = 'params.in'
    surrogate_dir = 'C:/Users/yisan/Desktop/quoFEMexamples/surrogates/SimGpModel_2_better.pkl'
    result_file = 'results_GP.out'
    """
    """
    try:
        opts, args = getopt.getopt(argv)
    except getopt.GetoptError:
        print
        'surrogate_pred.py -i <dir_params.in> -o <dir_SimGpModel.pkl>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print
            'surrogate_pred.py -i <dir_params.in> -o <dir_SimGpModel.pkl>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
   """

    params_dir = inputArgs[1]
    surrogate_meta_dir = inputArgs[2]
    input_json = inputArgs[3]  # scInput.json

    if len(inputArgs) > 4:  # noqa: PLR2004
        surrogate_dir = inputArgs[4]
    else:
        surrogate_dir = 'dummy'  # not used

    result_file = 'results.out'

    sys.exit(
        main(params_dir, surrogate_dir, surrogate_meta_dir, result_file, input_json)
    )
