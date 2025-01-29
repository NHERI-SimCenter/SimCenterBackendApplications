import glob  # noqa: INP001, D100
import json
import os
import shutil
import stat
import subprocess
import sys
import time

import numpy as np
import pandas as pd


class UQengine:  # noqa: D101
    def __init__(self, inputArgs):  # noqa: N803
        self.work_dir = inputArgs[1].replace(os.sep, '/')
        self.inputFile = inputArgs[2]
        self.workflowDriver = inputArgs[3]
        self.os_type = inputArgs[4]
        self.run_type = inputArgs[5]

        self.IM_names = []  # used in EEUQ

        jsonPath = self.inputFile  # noqa: N806
        if not os.path.isabs(jsonPath):  # noqa: PTH117
            # for quoFEM
            jsonPath = self.work_dir + '/templatedir/' + self.inputFile  # noqa: N806

        # temporary for EEUQ....
        jsonDir, jsonName = os.path.split(jsonPath)  # noqa: N806
        eeJsonPath = os.path.join(jsonDir, 'sc_' + jsonName)  # noqa: PTH118, N806

        if os.path.exists(eeJsonPath):  # noqa: PTH110
            self.inputFile = eeJsonPath
            jsonPath = eeJsonPath  # noqa: N806

        with open(jsonPath) as f:  # noqa: PTH123
            dakotaJson = json.load(f)  # noqa: N806, F841

        # self.workflowDriver = "workflow_driver"
        # if self.os_type.lower().startswith('win'):
        #    self.workflowDriver = "workflow_driver.bat"

    def cleanup_workdir(self):  # noqa: C901, D102, RUF100
        # if template dir already contains results.out, give an error

        # Cleanup working directory if needed

        del_paths = glob.glob(os.path.join(self.work_dir, 'workdir*'))  # noqa: PTH118, PTH207
        for del_path in del_paths:
            # change permission for  workflow_driver.bat
            self.workflowDriver_path = os.path.join(del_path, self.workflowDriver)  # noqa: PTH118
            # if os.path.exists(self.workflowDriver_path):
            #     os.chmod(self.workflowDriver_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

            # Change permission
            for root, dirs, files in os.walk(del_path):
                for d in dirs:
                    os.chmod(  # noqa: PTH101
                        os.path.join(root, d),  # noqa: PTH118
                        stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO,  # noqa: S103
                    )
                for f in files:
                    os.chmod(  # noqa: PTH101
                        os.path.join(root, f),  # noqa: PTH118
                        stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO,  # noqa: S103
                    )

            try:
                shutil.rmtree(del_path)
            except Exception as msg:  # noqa: BLE001
                self.exit(str(msg))

        del_outputs = glob.glob(os.path.join(self.work_dir, '*out'))  # noqa: PTH118, PTH207
        for del_out in del_outputs:
            os.remove(del_out)  # noqa: PTH107

        del_pkls = glob.glob(os.path.join(self.work_dir, '*pkl'))  # noqa: PTH118, PTH207
        for del_pkl in del_pkls:
            os.remove(del_pkl)  # noqa: PTH107

        # try:
        #    del_errs = glob.glob(os.path.join(self.work_dir, '*err'))  # noqa: PTH118, PTH207, RUF100
        #    for del_err in del_errs:
        #        os.remove(del_err)  # noqa: PTH107, RUF100
        # except:  # noqa: E722, RUF100, S110
        #    pass

        if glob.glob(os.path.join(self.work_dir, 'templatedir', 'results.out')):  # noqa: PTH118, PTH207
            try:
                os.remove(os.path.join(self.work_dir, 'templatedir', 'results.out'))  # noqa: PTH107, PTH118
            except:  # noqa: E722
                msg = 'Your main folder (where the main FEM script is located) already contains results.out. To prevent any confusion, please delete this file first'
                self.exit(msg)

        print('working directory cleared')  # noqa: T201

    def set_FEM(self, rv_name, do_parallel, y_dim, t_init, t_thr):  # noqa: N802, D102
        self.rv_name = rv_name
        self.do_parallel = do_parallel
        self.y_dim = y_dim
        self.t_init = t_init
        self.t_thr = t_thr
        self.total_sim_time = 0

    def run_FEM_batch(self, X, id_sim, runIdx=0, alterInput=[]):  # noqa: B006, C901, N802, N803, D102
        if runIdx == -1:
            # dummy run
            return X, np.zeros((0, self.y_dim)), id_sim
        workflowDriver = self.workflowDriver  # noqa: N806
        #
        # serial run
        #

        X = np.atleast_2d(X)  # noqa: N806
        nsamp = X.shape[0]
        if not self.do_parallel:
            Y = np.zeros((nsamp, self.y_dim))  # noqa: N806
            for ns in range(nsamp):
                Y_tmp, id_sim_current = run_FEM(  # noqa: N806
                    X[ns, :],
                    id_sim + ns,
                    self.rv_name,
                    self.work_dir,
                    workflowDriver,
                    runIdx,
                )
                if Y_tmp.shape[0] != self.y_dim:
                    msg = f'model output <results.out> in sample {ns} contains {Y_tmp.shape[0]} value(s) while the number of QoIs specified is {y_dim}'  # type: ignore # noqa: F821

                    self.exit(msg)
                Y[ns, :] = Y_tmp
                if time.time() - self.t_init > self.t_thr:
                    X = X[:ns, :]  # noqa: N806
                    Y = Y[:ns, :]  # noqa: N806
                    break
            Nsim = id_sim_current - id_sim + 1  # noqa: N806

        #
        # parallel run
        #

        if self.do_parallel:
            print(f'Running {nsamp} simulations in parallel')  # noqa: T201
            tmp = time.time()
            iterables = (
                (
                    X[i, :][np.newaxis],
                    id_sim + i,
                    self.rv_name,
                    self.work_dir,
                    self.workflowDriver,
                    runIdx,
                )
                for i in range(nsamp)
            )
            try:
                result_objs = list(self.pool.starmap(run_FEM, iterables))
                print(f'Simulation time = {time.time() - tmp} s')  # noqa: T201
            except KeyboardInterrupt:
                print('Ctrl+c received, terminating and joining pool.')  # noqa: T201
                try:
                    self.pool.shutdown()
                except Exception:  # noqa: BLE001
                    sys.exit()

            Nsim = len(list(result_objs))  # noqa: N806
            Y = np.zeros((Nsim, self.y_dim))  # noqa: N806
            for val, id in result_objs:  # noqa: A001
                if isinstance(val, str):
                    self.exit(val)
                elif val.shape[0]:
                    if val.shape[0] != self.y_dim:
                        msg = f'model output <results.out> in sample {id + 1} contains {val.shape[0]} value(s) while the number of QoIs specified is {self.y_dim}'
                        self.exit(msg)

                if np.isnan(np.sum(val)):
                    Nsim = id - id_sim  # noqa: N806
                    X = X[:Nsim, :]  # noqa: N806
                    Y = Y[:Nsim, :]  # noqa: N806
                else:
                    Y[id - id_sim, :] = val

        if len(alterInput) > 0:
            idx = alterInput[0]
            X = np.hstack([X[:, :idx], X[:, idx + 1 :]])  # noqa: N806

            # IM_vals = self.compute_IM(id_sim+1, id_sim + Nsim)
            # IM_list = list(map(str, IM_vals))[1:]
            # self.IM_names = IM_list
            # idx = alterInput[0]
            # X_new = np.hstack([X[:,:idx],IM_vals.to_numpy()[:,1:]])
            # X_new = np.hstack([X_new, X[:,idx+1:]])
            # X = X_new.astype(np.double)

        #
        # In case EEUQ
        #

        IM_vals = self.compute_IM(id_sim + 1, id_sim + Nsim)  # noqa: N806
        if IM_vals is None:
            X = X.astype(np.double)  # noqa: N806
        else:
            self.IM_names = list(map(str, IM_vals))[1:]
            X_new = np.hstack([X, IM_vals.to_numpy()[:, 1:]])  # noqa: N806
            X = X_new.astype(np.double)  # noqa: N806

        return X, Y, id_sim + Nsim

    def compute_IM(self, i_begin, i_end):  # noqa: N802, D102
        workdir_list = [
            os.path.join(self.work_dir, f'workdir.{int(i)}')  # noqa: PTH118
            for i in range(i_begin, i_end + 1)
        ]

        # intensity measure app
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
        for cur_workdir in workdir_list:
            os.chdir(cur_workdir)
            if os.path.exists('EVENT.json') and os.path.exists('AIM.json'):  # noqa: PTH110
                os.system(  # noqa: S605
                    f'{pythonEXE} {computeIM} --filenameAIM AIM.json --filenameEVENT EVENT.json --filenameIM IM.json  --geoMeanVar'
                )
            os.chdir(self.work_dir)

        # collect IMs from different workdirs
        for i, cur_workdir in enumerate(workdir_list):
            cur_id = int(cur_workdir.split('.')[-1])
            if os.path.exists(os.path.join(cur_workdir, 'IM.csv')):  # noqa: PTH110, PTH118
                print(f'IM.csv found in wordir.{cur_id}')  # noqa: T201
                tmp1 = pd.read_csv(
                    os.path.join(cur_workdir, 'IM.csv'),  # noqa: PTH118
                    index_col=None,
                )
                if tmp1.empty:
                    print(f'IM.csv in wordir.{cur_id} is empty.')  # noqa: T201
                    return None
                tmp2 = pd.DataFrame(
                    {'%eval_id': [cur_id for x in range(len(tmp1.index))]}
                )
                if i == 0:
                    im_collector = pd.concat([tmp2, tmp1], axis=1)
                else:
                    tmp3 = pd.concat([tmp2, tmp1], axis=1)
                    im_collector = pd.concat([im_collector, tmp3])
            else:
                print(f'IM.csv NOT found in wordir.{cur_id}')  # noqa: T201
                return None
        im_collector = im_collector.sort_values(by=['%eval_id'])

        return im_collector  # noqa: RET504
        # im_collector.to_csv('IM.csv', index=False)

    def readJson(self):  # noqa: N802, D102
        pass

    def make_pool(  # noqa: D102
        self, seed_val=42
    ):
        if self.run_type.lower() == 'runninglocal':
            from multiprocessing import Pool

            n_processor = os.cpu_count()

            if n_processor > 32:  # noqa: PLR2004
                n_processor = 8
            pool = Pool(n_processor, initializer=initfn, initargs=(seed_val,))

        else:
            from mpi4py import MPI # type: ignore  # noqa: I001
            from mpi4py.futures import MPIPoolExecutor # type: ignore

            self.world = MPI.COMM_WORLD
            n_processor = self.world.Get_size()
            pool = MPIPoolExecutor()
        return n_processor, pool

    #
    # Someplace to write down error messages
    #

    def create_errLog(self):  # noqa: N802, D102
        # self.errfile = open(os.path.join(self.work_dir, "dakota.err"), "a")
        pass

    def exit(self, msg):  # noqa: D102
        print(msg, file=sys.stderr)  # noqa: T201
        print(msg)  # noqa: T201
        # sys.stderr.write(msg)
        # self.errfile.write(msg)
        # self.errfile.close()
        exit(-1)  # noqa: PLR1722

    def terminate_errLog(self):  # noqa: N802, D102
        # self.errfile.close()
        pass

    #
    # To read text
    #


def run_FEM(X, id_sim, rv_name, work_dir, workflowDriver, runIdx=0):  # noqa: C901, N802, N803, D103
    if runIdx == 0:
        templatedirFolder = '/templatedir'  # noqa: N806
        workdirFolder = '/workdir.' + str(id_sim + 1)  # noqa: N806
    else:
        templatedirFolder = '/templatedir.' + str(runIdx)  # noqa: N806
        workdirFolder = '/workdir.' + str(runIdx) + '.' + str(id_sim + 1)  # noqa: N806

    X = np.atleast_2d(X)  # noqa: N806
    x_dim = X.shape[1]

    if X.shape[0] > 1:
        msg = 'do one simulation at a time'
        return msg, id_sim
    #
    # (1) create "workdir.idx " folder :need C++17 to use the files system namespace
    #

    current_dir_i = work_dir + workdirFolder
    try:
        shutil.copytree(work_dir + templatedirFolder, current_dir_i)
    except Exception:  # noqa: BLE001
        try:
            shutil.copytree(work_dir + templatedirFolder, current_dir_i)

        except Exception as ex:  # noqa: BLE001
            msg = 'Error running FEM: ' + str(ex)
            return msg, id_sim

    #
    # (2) write param.in file
    #

    outF = open(current_dir_i + '/params.in', 'w')  # noqa: SIM115, PTH123, N806
    outF.write(f'{x_dim}\n')
    for i in range(x_dim):
        outF.write(f'{rv_name[i]} {X[0, i]}\n')
    outF.close()

    if runIdx == 0:
        print(f'RUNNING FEM: working directory {id_sim + 1} created')  # noqa: T201
    else:
        print(f'RUNNING FEM: working directory {runIdx}-{id_sim + 1} created')  # noqa: T201

    #
    # (3) run workflow_driver.bat
    #

    os.chdir(current_dir_i)
    workflow_run_command = f'{current_dir_i}/{workflowDriver}  1> workflow.log 2>&1'
    # subprocess.check_call(
    #    workflow_run_command,
    #    shell=True,
    #    stdout=subprocess.DEVNULL,
    #    stderr=subprocess.STDOUT,
    # )    # subprocess.check_call(workflow_run_command, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
    # => to end grasefully
    returnCode = subprocess.call(  # noqa: S602, N806, F841
        workflow_run_command,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )  # subprocess.check_call(workflow_run_command, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)

    #
    # (4) reading results
    #

    if glob.glob('results.out'):  # noqa: PTH207
        g = np.loadtxt('results.out').flatten()
    else:
        msg = 'Error running FEM: results.out missing at ' + current_dir_i
        if glob.glob('ops.out'):  # noqa: PTH207
            with open('ops.out') as text_file:  # noqa: PTH123
                error_FEM = text_file.read()  # noqa: N806

            startingCharId = error_FEM.lower().find('error')  # noqa: N806

            if startingCharId > 0:
                startingCharId = max(0, startingCharId - 20)  # noqa: N806
                endingID = max(len(error_FEM), startingCharId + 200)  # noqa: N806
                errmsg = error_FEM[startingCharId:endingID]
                errmsg = errmsg.split(' ', 1)[1]
                errmsg = errmsg[0 : errmsg.rfind(' ')]
                msg += '\n'
                msg += 'your FEM model says...\n'
                msg += '........\n' + errmsg + '\n........ \n'
                msg += 'to read more, see ' + os.path.join(os.getcwd(), 'ops.out')  # noqa: PTH109, PTH118

        return msg, id_sim

    if g.shape[0] == 0:
        msg = 'Error running FEM: results.out is empty'
        if glob.glob('ops.out'):  # noqa: PTH207
            with open('ops.out') as text_file:  # noqa: PTH123
                error_FEM = text_file.read()  # noqa: N806

            startingCharId = error_FEM.lower().find('error')  # noqa: N806

            if startingCharId > 0:
                startingCharId = max(0, startingCharId - 20)  # noqa: N806
                endingID = max(len(error_FEM), startingCharId + 200)  # noqa: N806
                errmsg = error_FEM[startingCharId:endingID]
                errmsg = errmsg.split(' ', 1)[1]
                errmsg = errmsg[0 : errmsg.rfind(' ')]
                msg += '\n'
                msg += 'your FEM model says...\n'
                msg += '........\n' + errmsg + '\n........ \n'
                msg += 'to read more, see ' + os.path.join(os.getcwd(), 'ops.out')  # noqa: PTH109, PTH118

        return msg, id_sim

    os.chdir('../')

    if np.isnan(np.sum(g)):
        msg = f'Error running FEM: Response value at workdir.{id_sim + 1} is NaN'
        return msg, id_sim

    return g, id_sim

    # def readCSV(self):
    #     pass
    #     return
    # def MCS(self):
    #     pass
    # def makePool(self):
    #     pass


# for creating pool
def initfn(seed_val):  # noqa: D103
    np.random.seed(seed_val)  # enforcing seeds


#
# When sampled X is different from surrogate input X. e.g. we sample ground motion parameters or indices, but we use IM as input of GP
#

# def run_FEM_alterX(X, id_sim, rv_name, work_dir, workflowDriver, runIdx=0, alterIdx, alterFiles):
#     g, id_sim = run_FEM(X, id_sim, rv_name, work_dir, workflowDriver, runIdx=0)


#
# class simcenterUQ(UQengine):
#     def __init__(self):
#         pass
#         #
#
#
# class surrogate(simcenterUQ):
#     def __init__(self):
#         pass
#     def readUQ:
#         pass
#     def designExp(self):
#         # (1) generate samples
#
#             Y = self.runFEM()
#         # (2) calibrat
#         # loop
#
#
