# JGA  # noqa: CPY001, D100, N999
import importlib
import os
import sys

# import matplotlib.pyplot as plt
# export DISPLAY=localhost:0.0
from ctypes import *  # noqa: F403
from pathlib import Path

import numpy as np
import pandas as pd
import PLoM_library as plom
from general import *  # noqa: F403


class PLoM:  # noqa: D101
    def __init__(
        self,
        model_name='plom',
        data='',
        separator=',',
        col_header=False,  # noqa: FBT002
        constraints=None,
        run_tag=False,  # noqa: FBT002
        plot_tag=False,  # noqa: FBT002
        num_rlz=5,
        tol_pca=1e-6,
        epsilon_kde=25,
        tol_PCA2=1e-5,  # noqa: N803
        tol=1e-6,
        max_iter=50,
        runDiffMaps=True,  # noqa: FBT002, N803
        db_path=None,
    ):
        # basic setups
        self._basic_config(model_name=model_name, db_path=db_path)
        self.plot_tag = plot_tag
        # initialize constraints
        self.constraints = {}
        self.num_constraints = 0
        self.runDiffMaps = runDiffMaps
        # initialize input data
        if self.initialize_data(data, separator, col_header):
            self.logfile.write_msg(
                msg='PLoM: data loading failed.', msg_type='ERROR', msg_level=0
            )
        else:
            """
            # plot data matrix
            if self.plot_tag:
                smp = pd.plotting.scatter_matrix(self.X0, alpha=0.5, diagonal ='kde', figsize=(10,10))
                for ax in smp.ravel():
                    ax.set_xlabel(ax.get_xlabel(), fontsize = 6, rotation = 45)
                    ax.set_ylabel(ax.get_ylabel(), fontsize = 6, rotation = 45)
                plt.savefig(os.path.join(self.vl_path,'ScatterMatrix_X0.png'),dpi=480)
                self.logfile.write_msg(msg='PLoM: {} saved in {}.'.format('ScatterMatrix_X0.png',self.vl_path),msg_type='RUNNING',msg_level=0)
            """
        if not self.constraints:
            if self.add_constraints(constraints_file=constraints):
                self.logfile.write_msg(
                    msg='PLoM: constraints input failed.',
                    msg_type='ERROR',
                    msg_level=0,
                )
        # run
        if run_tag:
            self.logfile.write_msg(
                msg='PLoM: Running all steps to generate new samples.',
                msg_type='RUNNING',
                msg_level=0,
            )
            self.ConfigTasks()
            self.RunAlgorithm(
                n_mc=num_rlz,
                epsilon_pca=tol_pca,
                epsilon_kde=epsilon_kde,
                tol_PCA2=tol_PCA2,
                tol=tol,
                max_iter=max_iter,
                plot_tag=plot_tag,
                runDiffMaps=self.runDiffMaps,
            )
        else:
            self.logfile.write_msg(
                msg='PLoM: using ConfigTasks(task_list = FULL_TASK_LIST) to schedule a run.',
                msg_type='RUNNING',
                msg_level=0,
            )
            self.logfile.write_msg(
                msg='PLoM: using RunAlgorithm(n_mc=n_mc,epsilon_pca=epsilon_pca,epsilon_kde) to run simulations.',
                msg_type='RUNNING',
                msg_level=0,
            )

    def _basic_config(self, model_name=None, db_path=None):
        """Basic setups
        - model_name: job name (used for database name)
        """  # noqa: D205, D400, D401
        if not db_path:
            self.dir_log = os.path.join(  # noqa: PTH118
                os.path.dirname(os.path.abspath(__file__)),  # noqa: PTH100, PTH120
                'RunDir',
            )
            self.dir_run = os.path.join(  # noqa: PTH118
                os.path.dirname(os.path.abspath(__file__)),  # noqa: PTH100, PTH120
                'RunDir',
                model_name,
            )
        else:
            self.dir_log = db_path
            self.dir_run = os.path.join(db_path, model_name)  # noqa: PTH118
        # initialize logfile
        try:
            os.makedirs(self.dir_run, exist_ok=True)  # noqa: PTH103
            self.logfile = Logfile(logfile_dir=self.dir_log)  # noqa: F405
            self.logfile.write_msg(
                msg=f'PLoM: Running directory {self.dir_run} initialized.',
                msg_type='RUNNING',
                msg_level=0,
            )
        except:  # noqa: E722
            self.logfile.write_msg(
                msg=f'PLoM: Running directory {self.dir_run} cannot be initialized.',
                msg_type='ERROR',
                msg_level=0,
            )
        # initialize database server
        self.dbserver = None
        self.dbserver = DBServer(db_dir=self.dir_run, db_name=model_name + '.h5')  # noqa: F405
        try:
            self.dbserver = DBServer(db_dir=self.dir_run, db_name=model_name + '.h5')  # noqa: F405
        except:  # noqa: E722
            self.logfile.write_msg(
                msg='PLoM: database server initialization failed.',
                msg_type='ERROR',
                msg_level=0,
            )
        if self.dbserver:
            self.logfile.write_msg(
                msg='PLoM: database server initialized.',
                msg_type='RUNNING',
                msg_level=0,
            )
        # initialize visualization output path
        self.vl_path = os.path.join(self.dir_run, 'FigOut')  # noqa: PTH118
        try:
            os.makedirs(self.vl_path, exist_ok=True)  # noqa: PTH103
            self.logfile.write_msg(
                msg=f'PLoM: visualization folder {self.vl_path} initialized.',
                msg_type='RUNNING',
                msg_level=0,
            )
        except:  # noqa: E722
            self.logfile.write_msg(
                msg=f'PLoM: visualization folder {self.vl_path} not initialized.',
                msg_type='WARNING',
                msg_level=0,
            )

    def add_constraints(self, constraints_file=None):  # noqa: D102
        if not constraints_file:
            self.g_c = None
            self.D_x_g_c = None
            self.beta_c = []
            self.beta_c_aux = None
            self.lambda_i = 0
            self.psi = 0
            self.logfile.write_msg(
                msg='PLoM.add_constraints: no user-defined constraint - please use add_constraints(constraints_file=X) to add new constraints if any.',
                msg_type='WARNING',
                msg_level=0,
            )
            return 0

        try:
            # path
            path_constraints = Path(constraints_file).resolve()
            sys.path.insert(0, str(path_constraints.parent) + '/')
            # load the function
            new_constraints = importlib.__import__(
                path_constraints.name[:-3], globals(), locals(), [], 0
            )
        except:  # noqa: E722
            self.logfile.write_msg(
                msg=f'PLoM.add_constraints: could not add constraints {constraints_file}',
                msg_type='ERROR',
                msg_level=0,
            )
            return 1
        self.num_constraints = self.num_constraints + 1  # noqa: PLR6104
        try:
            self.constraints.update(
                {
                    'Constraint' + str(self.num_constraints): {
                        'filename': constraints_file,
                        'g_c': new_constraints.g_c,
                        'D_x_g_c': new_constraints.D_x_g_c,
                        'beta_c': new_constraints.beta_c(),
                        'beta_c_aux': new_constraints.beta_c_aux,
                    }
                }
            )
            self.g_c = new_constraints.g_c
            self.D_x_g_c = new_constraints.D_x_g_c
            self.beta_c = new_constraints.beta_c()
            self.beta_c_aux = new_constraints.beta_c_aux
            self.logfile.write_msg(
                msg='PLoM.add_constraints: constraints added.',
                msg_type='RUNNING',
                msg_level=0,
            )
            self.dbserver.add_item(
                item=[constraints_file], data_type='ConstraintsFile'
            )
        except:  # noqa: E722
            self.logfile.write_msg(
                msg=f'PLoM.add_constraints: at least one attribute (i.e., g_c, D_x_gc, beta_c, or beta_c_aux) missing in {constraints_file}',
                msg_type='ERROR',
                msg_level=0,
            )
            return 1
        return 0

    def switch_constraints(self, constraint_tag=1):
        """Selecting different constraints
        - constraint_tag: the tag of selected constraint
        """  # noqa: D205, D400, D401
        if constraint_tag > self.num_constraints:
            self.logfile.write_msg(
                msg=f'PLoM.switch_constraints: sorry the maximum constraint tag is {self.num_constraints}',
                msg_type='ERROR',
                msg_level=0,
            )
        try:
            self.g_c = self.constraints.get('Constraint' + str(constraint_tag)).get(
                'g_c'
            )
            self.D_x_g_c = self.constraints.get(
                'Constraint' + str(constraint_tag)
            ).get('D_x_g_c')
            self.beta_c = self.constraints.get(
                'Constraint' + str(constraint_tag)
            ).get('beta_c')
            self.beta_c_aux = self.constraints.get(
                'Constraint' + str(constraint_tag)
            ).get('beta_c_aux')
            self.dbserver.add_item(
                item=[
                    self.constraints.get('Constraint' + str(constraint_tag)).get(
                        'filename'
                    )
                ],
                data_type='ConstraintsFile',
            )
        except:  # noqa: E722
            self.logfile.write_msg(
                msg='PLoM.get_constraints: cannot get constraints',
                msg_type='ERROR',
                msg_level=0,
            )

    def delete_constraints(self):
        """Removing all current constraints"""  # noqa: D400, D401
        self.g_c = None
        self.D_x_g_c = None
        self.beta_c = []
        self.dbserver.add_item(item=[''], data_type='ConstraintsFile')

    def load_data(self, filename, separator=',', col_header=False):  # noqa: FBT002, C901, D102
        # initialize the matrix and data size
        X = []  # noqa: N806
        N = 0  # noqa: N806
        n = 0

        # check if the file exist
        import os  # noqa: PLC0415

        if not os.path.exists(filename):  # noqa: PTH110
            self.logfile.write_msg(
                msg=f'load_data: the input file {filename} is not found',
                msg_type='ERROR',
                msg_level=0,
            )
            return X, N, n

        # read data
        if os.path.splitext(filename)[-1] in ['.csv', '.dat', '.txt']:  # noqa: PLR6201, PTH122
            # txt data
            col = None
            if col_header:
                col = 0
            self.X0 = pd.read_table(filename, delimiter=separator, header=col)
            # remove all-nan column if any
            for cur_col in self.X0.columns:
                if all(np.isnan(self.X0.loc[:, cur_col])):
                    self.X0.drop(columns=cur_col)
            X = self.X0.to_numpy()  # noqa: N806

        elif os.path.splitext(filename)[-1] in ['.mat', '.json']:  # noqa: PLR6201, PTH122
            # json or mat
            if os.path.splitext(filename)[-1] == '.mat':  # noqa: PTH122
                import scipy.io as scio  # noqa: PLC0415

                matdata = scio.loadmat(filename)
                var_names = [
                    x for x in list(matdata.keys()) if not x.startswith('__')
                ]
                if len(var_names) == 1:
                    # single matrix
                    X = matdata[var_names[0]]  # noqa: N806
                    self.X0 = pd.DataFrame(
                        X, columns=['Var' + str(x) for x in X.shape[1]]
                    )
                else:
                    n = len(var_names)
                    # multiple columns
                    for cur_var in var_names:
                        X.append(matdata[cur_var].tolist())
                    X = np.array(X).T  # noqa: N806
                    X = X[0, :, :]  # noqa: N806
                    self.X0 = pd.DataFrame(X, columns=var_names)
            else:
                import json  # noqa: PLC0415

                with open(filename, encoding='utf-8') as f:  # noqa: PTH123
                    jsondata = json.load(f)
                var_names = list(jsondata.keys())
                # multiple columns
                for cur_var in var_names:
                    X.append(jsondata[cur_var])
                X = np.array(X).T  # noqa: N806
                self.X0 = pd.DataFrame(X, columns=var_names)

        elif os.path.splitext(filename)[-1] == '.h5':  # noqa: PTH122
            # this h5 can be either formatted by PLoM or not
            # a separate method to deal with this file
            X = self.load_h5(filename)  # noqa: N806

        else:
            self.logfile.write_msg(
                msg='PLoM.load_data: the file format is not supported yet.',
                msg_type='ERROR',
                msg_level=0,
            )
            self.logfile.write_msg(
                msg='PLoM.load_data: accepted data formats - csv, dat, txt, mat, json.',
                msg_type='WARNING',
                msg_level=0,
            )

        # Update data sizes
        N, n = X.shape  # noqa: N806
        self.logfile.write_msg(
            msg=f'PLoM.load_data: loaded data size = ({N}, {n}).',
            msg_type='RUNNING',
            msg_level=0,
        )

        # Return data and data sizes
        return X.T, N, n

    # def check_var_name():

    def get_data(self):  # noqa: D102
        # return data and data sizes
        return self.X, self.N, self.n

    def _load_h5_plom(self, filename):
        """Loading PLoM-formatted h5 database"""  # noqa: D400, D401
        try:
            store = pd.HDFStore(filename, 'r')
            for cur_var in store.keys():  # noqa: SIM118
                if cur_var in self.dbserver.get_item_adds() and ATTR_MAP[cur_var]:  # noqa: F405
                    # read in
                    cur_data = store[cur_var]
                    cur_dshape = tuple(
                        [x[0] for x in store['/DS_' + cur_var[1:]].values.tolist()]  # noqa: PD011
                    )
                    if cur_dshape == (1,):
                        item_value = np.array(sum(cur_data.values.tolist(), []))  # noqa: PD011, RUF017
                        col_headers = list(cur_data.columns)[0]  # noqa: RUF015
                    else:
                        item_value = cur_data.values  # noqa: PD011
                        col_headers = list(cur_data.columns)
                    self.dbserver.add_item(
                        item_name=cur_var.replace('/', ''),
                        col_names=col_headers,
                        item=item_value,
                        data_shape=cur_dshape,
                    )
                # constraints
                if cur_var == '/constraints_file':
                    cur_data = store[cur_var]
                    self.dbserver.add_item(
                        item=cur_data.values.tolist()[0],  # noqa: PD011
                        data_type='ConstraintsFile',
                    )
            store.close()

        except:  # noqa: E722
            self.logfile.write_msg(
                msg=f'PLoM._load_h5_plom: data in {filename} not compatible.',
                msg_type='ERROR',
                msg_level=0,
            )

    def _load_h5_data_X(self, filename):  # noqa: N802
        """Loading a h5 data which is expected to contain X data"""  # noqa: D400, D401
        try:
            store = pd.HDFStore(filename, 'r')
            # Note a table is expected for the variable
            self.X0 = store.get(store.keys()[0])
            store.close()
            self.dbserver.add_item(
                item_name='X0', col_name=list(self.X0.columns), item=self.X0
            )

            return self.X0.to_numpy()
        except:  # noqa: E722
            return None

    def _sync_data(self):
        """Sync database data to current attributes"""  # noqa: D400
        avail_name_list = self.dbserver.get_name_list()
        if not avail_name_list:
            # empty database
            self.logfile.write_msg(
                msg='PLoM._sync_data: database is empty - no data to sync.',
                msg_type='WARNING',
                msg_level=0,
            )
        else:
            for cur_item in avail_name_list:
                if cur_item.startswith('/DS_'):
                    # skipping the data-shape attributes
                    continue
                if type(ATTR_MAP[cur_item]) is str:  # noqa: F405
                    self.__setattr__(  # noqa: PLC2801
                        ATTR_MAP[cur_item],  # noqa: F405
                        self.dbserver.get_item(cur_item[1:]),
                    )
                    self.logfile.write_msg(
                        msg=f'PLoM._sync_data: self.{ATTR_MAP[cur_item]} synced.',  # noqa: F405
                        msg_type='RUNNING',
                        msg_level=0,
                    )
                else:
                    # None type (this is the 'basic' - skipped)
                    self.logfile.write_msg(
                        msg=f'PLoM._sync_data: data {cur_item} skipped.',
                        msg_type='RUNNING',
                        msg_level=0,
                    )

    def _sync_constraints(self):
        """Sync constraints from dbserver to the attributes"""  # noqa: D400
        avail_name_list = self.dbserver.get_name_list()
        if '/constraints_file' not in avail_name_list:
            # empty constraints
            self.logfile.write_msg(
                msg='PLoM._sync_data: no available constraint to sync.',
                msg_type='WARNING',
                msg_level=0,
            )
        else:
            # get constraints file path
            cfile = self.dbserver.get_item(data_type='ConstraintsFile')
            # add the constraints
            self.add_constraints(constraints_file=cfile)

    def load_h5(self, filename):
        """Loading h5 database"""  # noqa: D400, D401
        try:
            self._load_h5_plom(filename)
            self.logfile.write_msg(
                msg='PLoM.load_h5: h5 file loaded.', msg_type='RUNNING', msg_level=0
            )
            # sync data
            self._sync_data()
            self.logfile.write_msg(
                msg=f'PLoM.load_h5: data in {filename} synced.',
                msg_type='RUNNING',
                msg_level=0,
            )
            self._sync_constraints()
            self.logfile.write_msg(
                msg=f'PLoM.load_h5: constraints in {filename} synced.',
                msg_type='RUNNING',
                msg_level=0,
            )
            if '/X0' in self.dbserver.get_name_list():
                self.X0 = self.dbserver.get_item('X0', table_like=True)
                return self.X0.to_numpy()
            else:  # noqa: RET505
                self.logfile.write_msg(
                    msg='PLoM.load_h5: the original X0 data not found in the loaded data.',
                    msg_type='ERROR',
                    msg_level=0,
                )
                return None
        except:  # noqa: E722
            X = self._load_h5_data_X(filename)  # noqa: N806
            if X is None:
                self.logfile.write_msg(
                    msg=f'PLoM.load_h5: cannot load {filename}.',
                    msg_type='ERROR',
                    msg_level=0,
                )
                return None
            else:  # noqa: RET505
                return X

    def add_data(self, filename, separator=',', col_header=False):  # noqa: FBT002, D102
        # load new data
        new_X, new_N, new_n = self.load_data(filename, separator, col_header)  # noqa: N806
        # check data sizes
        if new_n != self.n:
            self.logfile.write_msg(
                msg=f'PLoM.add_data: incompatible column size when loading {filename}',
                msg_type='ERROR',
                msg_level=0,
            )
        else:
            # update the X and N
            self.X = np.concatenate((self.X, new_X))
            self.N = self.N + new_N  # noqa: PLR6104
            self.X0.append(pd.DataFrame(new_X.T, columns=list(self.X0.columns)))

        self.logfile.write_msg(
            msg=f'PLoM.add_data: current X0 size = ({self.N}, {self.n}).',
            msg_type='RUNNING',
            msg_level=0,
        )

    def initialize_data(  # noqa: D102
        self,
        filename,
        separator=',',
        col_header=False,  # noqa: FBT002
        constraints='',  # noqa: ARG002
    ):
        # initialize the data and data sizes
        try:
            self.X, self.N, self.n = self.load_data(filename, separator, col_header)
        except:  # noqa: E722
            self.logfile.write_msg(
                msg=f'PLoM.initialize_data: cannot initialize data with {filename}',
                msg_type='ERROR',
                msg_level=0,
            )
            return 1

        # Save to database
        self.dbserver.add_item(
            item_name='X0',
            col_names=list(self.X0.columns),
            item=self.X.T,
            data_shape=self.X.shape,
        )
        self.dbserver.add_item(item_name='N', item=np.array([self.N]))
        self.dbserver.add_item(item_name='n', item=np.array([self.n]))
        self.logfile.write_msg(
            msg=f'PLoM.initialize_data: current X0 size = ({self.N}, {self.n}).',
            msg_type='RUNNING',
            msg_level=0,
        )
        self.logfile.write_msg(
            msg='PLoM.initialize_data: X0 and X0_size saved to database.',
            msg_type='RUNNING',
            msg_level=0,
        )

        return 0

    def _init_indv_tasks(self):
        """Initializing tasks"""  # noqa: D400, D401
        for cur_task in FULL_TASK_LIST:  # noqa: F405
            self.__setattr__('task_' + cur_task, Task(task_name=cur_task))  # noqa: F405, PLC2801

    def ConfigTasks(self, task_list=FULL_TASK_LIST):  # noqa: C901, N802, F405
        """Creating a task list object
        - task_list: a string list of tasks to run
        """  # noqa: D205, D400, D401
        config_flag = True
        self.cur_task_list = task_list
        # check task orders
        if not all([x in FULL_TASK_LIST for x in self.cur_task_list]):  # noqa: C419, F405
            self.logfile.write_msg(
                msg='PLoM.config_tasks: task name not recognized.',
                msg_type='ERROR',
                msg_level=0,
            )
            self.logfile.write_msg(
                msg='PLoM.config_tasks: acceptable task names: {}.'.format(
                    ','.join(FULL_TASK_LIST)  # noqa: F405
                ),
                msg_type='WARNING',
                msg_level=0,
            )
            return False
        map_order = [FULL_TASK_LIST.index(x) for x in self.cur_task_list]  # noqa: F405
        if map_order != sorted(map_order):
            self.logfile.write_msg(
                msg='PLoM.config_tasks: task order error.',
                msg_type='ERROR',
                msg_level=0,
            )
            self.logfile.write_msg(
                msg='PLoM.config_tasks: please follow this order: {}.'.format(
                    '->'.join(FULL_TASK_LIST)  # noqa: F405
                ),
                msg_type='WARNING',
                msg_level=0,
            )
            return False
        if (max(map_order) - min(map_order) + 1) != len(map_order):
            # intermediate tasks missing -> since the jobs are in chain, so here the default is to automatically fill in any missing tasks in the middle
            self.cur_task_list = FULL_TASK_LIST[min(map_order) : max(map_order) + 1]  # noqa: F405
            self.logfile.write_msg(
                msg='PLoM.config_tasks: intermediate task(s) missing and being filled in automatically.',
                msg_type='WARNING',
                msg_level=0,
            )
            self.logfile.write_msg(
                msg='PLoM.config_tasks: the filled task list is: {}.'.format(
                    '->'.join(self.cur_task_list)
                ),
                msg_type='RUNNING',
                msg_level=0,
            )
        # initializing the task list
        self.task_list = TaskList()  # noqa: F405
        # initializing individual tasks and refreshing status
        self._init_indv_tasks()
        for cur_task in FULL_TASK_LIST:  # noqa: F405
            self.__getattribute__('task_' + cur_task).full_var_list = TASK_ITEM_MAP[  # noqa: F405, PLC2801
                cur_task
            ]
            for cur_item in TASK_ITEM_MAP[cur_task]:  # noqa: F405
                if '/' + cur_item in self.dbserver.get_name_list():
                    self.__getattribute__('task_' + cur_task).avail_var_list.append(  # noqa: PLC2801
                        cur_item
                    )
                self.__getattribute__('task_' + cur_task).refresh_status()  # noqa: PLC2801
        # create the task list
        for cur_task in self.cur_task_list:
            self.task_list.add_task(
                new_task=self.__getattribute__('task_' + cur_task)  # noqa: PLC2801
            )

        self.task_list.refresh_status()
        # need to check the task chain if all dependent tasks completed to go
        # otherwise, the current run could not be completed
        pre_task_list = FULL_TASK_LIST[: FULL_TASK_LIST.index(self.cur_task_list[0])]  # noqa: F405
        if len(pre_task_list):
            for cur_task in pre_task_list:
                if not self.__getattribute__('task_' + cur_task).refresh_status():  # noqa: PLC2801
                    config_flag = False
                    self.logfile.write_msg(
                        msg=f'PLoM.config_tasks: configuration failed with dependent task {cur_task} not completed.',
                        msg_type='ERROR',
                        msg_level=0,
                    )

        if config_flag:  # noqa: RET503
            self.logfile.write_msg(  # noqa: RET503
                msg='PLoM.config_tasks: the following tasks is configured to run: {}.'.format(
                    '->'.join(self.cur_task_list)
                ),
                msg_type='RUNNING',
                msg_level=0,
            )

    def RunAlgorithm(  # noqa: C901, N802
        self,
        n_mc=5,
        epsilon_pca=1e-6,
        epsilon_kde=25,
        tol_PCA2=1e-5,  # noqa: N803
        tol=1e-6,
        max_iter=50,
        plot_tag=False,  # noqa: FBT002
        runDiffMaps=None,  # noqa: N803
        seed_num=None,
        tolKDE=0.1,  # noqa: N803
    ):
        """Running the PLoM algorithm to train the model and generate new realizations
        - n_mc: realization/sample size ratio
        - epsilon_pca: tolerance for selecting the number of considered components in PCA
        - epsilon_kde: smoothing parameter in the kernel density estimation
        - tol: tolerance in the PLoM iterations
        - max_iter: maximum number of iterations of the PLoM algorithm
        """  # noqa: D205, D400, D401
        if runDiffMaps == None:  # noqa: E711
            runDiffMaps = self.runDiffMaps  # noqa: N806
        else:
            self.runDiffMaps = runDiffMaps

        if plot_tag:
            self.plot_tag = plot_tag
        cur_task = self.task_list.head_task
        while cur_task:
            if cur_task.task_name == 'DataNormalization':
                self.__getattribute__(  # noqa: PLC2801
                    'task_' + cur_task.task_name
                ).avail_var_list = []
                # data normalization
                self.X_scaled, self.alpha, self.x_min, self.x_mean = (
                    self.DataNormalization(self.X)
                )
                self.logfile.write_msg(
                    msg='PLoM.RunAlgorithm: data normalization completed.',
                    msg_type='RUNNING',
                    msg_level=0,
                )
                self.dbserver.add_item(
                    item_name='X_range', item=self.alpha, data_shape=self.alpha.shape
                )
                self.dbserver.add_item(
                    item_name='X_min',
                    col_names=list(self.X0.columns),
                    item=self.x_min.T,
                    data_shape=self.x_min.shape,
                )
                self.dbserver.add_item(
                    item_name='X_scaled',
                    col_names=list(self.X0.columns),
                    item=self.X_scaled.T,
                    data_shape=self.X_scaled.shape,
                )
                self.dbserver.add_item(
                    item_name='X_scaled_mean',
                    col_names=list(self.X0.columns),
                    item=self.x_mean.T,
                    data_shape=self.x_mean.shape,
                )
                self.logfile.write_msg(
                    msg='PLoM.RunAlgorithm: X_range, X_min, X_scaled and X_scaled_mean saved.',
                    msg_type='RUNNING',
                    msg_level=0,
                )
            elif cur_task.task_name == 'RunPCA':
                self.__getattribute__(  # noqa: PLC2801
                    'task_' + cur_task.task_name
                ).avail_var_list = []
                # PCA
                self.H, self.mu, self.phi, self.nu, self.errPCA = self.RunPCA(
                    self.X_scaled, epsilon_pca
                )
                self.logfile.write_msg(
                    msg='PLoM.RunAlgorithm: PCA completed.',
                    msg_type='RUNNING',
                    msg_level=0,
                )
                self.dbserver.add_item(
                    item_name='X_PCA',
                    col_names=[
                        'Component' + str(i + 1) for i in range(self.H.shape[0])
                    ],
                    item=self.H.T,
                    data_shape=self.H.shape,
                )
                self.dbserver.add_item(
                    item_name='EigenValue_PCA',
                    item=self.mu,
                    data_shape=self.mu.shape,
                )
                self.dbserver.add_item(
                    item_name='EigenVector_PCA',
                    col_names=['V' + str(i + 1) for i in range(self.phi.shape[1])],
                    item=self.phi,
                    data_shape=self.phi.shape,
                )
                self.dbserver.add_item(
                    item_name='NumComp_PCA', item=np.array([self.nu])
                )
                self.dbserver.add_item(
                    item_name='Error_PCA', item=np.array(self.errPCA)
                )
                self.logfile.write_msg(
                    msg='PLoM.RunAlgorithm: X_PCA, EigenValue_PCA and EigenVector_PCA saved.',
                    msg_type='RUNNING',
                    msg_level=0,
                )
            elif cur_task.task_name == 'RunKDE':
                self.__getattribute__(  # noqa: PLC2801
                    'task_' + cur_task.task_name
                ).avail_var_list = []
                # parameters KDE
                self.s_v, self.c_v, self.hat_s_v, self.K, self.b = self.RunKDE(
                    self.H, epsilon_kde
                )
                self.logfile.write_msg(
                    msg='PLoM.RunAlgorithm: kernel density estimation completed.',
                    msg_type='RUNNING',
                    msg_level=0,
                )
                self.dbserver.add_item(item_name='s_v', item=np.array([self.s_v]))
                self.dbserver.add_item(item_name='c_v', item=np.array([self.c_v]))
                self.dbserver.add_item(
                    item_name='hat_s_v', item=np.array([self.hat_s_v])
                )
                self.dbserver.add_item(
                    item_name='X_KDE', item=self.K, data_shape=self.K.shape
                )
                self.dbserver.add_item(
                    item_name='EigenValues_KDE', item=self.b, data_shape=self.b.shape
                )
                self.logfile.write_msg(
                    msg='PLoM.RunAlgorithm: KDE, X_KDE and EigenValues_KDE saved.',
                    msg_type='RUNNING',
                    msg_level=0,
                )
                # diff maps
                if runDiffMaps:
                    self.__getattribute__(  # noqa: PLC2801
                        'task_' + cur_task.task_name
                    ).avail_var_list = []
                    # diff maps
                    self.g, self.m, self.a, self.Z, self.eigenKDE = self.DiffMaps(
                        self.H, self.K, self.b, tol=tolKDE
                    )
                    self.logfile.write_msg(
                        msg='PLoM.RunAlgorithm: diffusion maps completed.',
                        msg_type='RUNNING',
                        msg_level=0,
                    )
                    self.dbserver.add_item(
                        item_name='KDE_g', item=self.g, data_shape=self.g.shape
                    )
                    self.dbserver.add_item(
                        item_name='KDE_m', item=np.array([self.m])
                    )
                    self.dbserver.add_item(
                        item_name='KDE_a', item=self.a, data_shape=self.a.shape
                    )
                    self.dbserver.add_item(
                        item_name='KDE_Z', item=self.Z, data_shape=self.Z.shape
                    )
                    self.dbserver.add_item(
                        item_name='KDE_Eigen',
                        item=self.eigenKDE,
                        data_shape=self.eigenKDE.shape,
                    )
                    self.logfile.write_msg(
                        msg='PLoM.RunAlgorithm: KDE_g, KDE_m, KDE_a, KDE_Z, and KDE_Eigen saved.',
                        msg_type='RUNNING',
                        msg_level=0,
                    )
                else:
                    self.g = np.identity(self.N)
                    self.m = self.N
                    self.a = self.g[:, : self.m].dot(
                        np.linalg.inv(
                            np.transpose(self.g[:, : self.m]).dot(
                                self.g[:, : self.m]
                            )
                        )
                    )
                    self.Z = self.H.dot(self.a)
                    self.eigenKDE = np.array([])
                    self.logfile.write_msg(
                        msg='PLoM.RunAlgorithm: diffusion map is inactivated.',
                        msg_type='RUNNING',
                        msg_level=0,
                    )
                    self.dbserver.add_item(
                        item_name='KDE_g', item=self.g, data_shape=self.g.shape
                    )
                    self.dbserver.add_item(
                        item_name='KDE_m', item=np.array([self.m])
                    )
                    self.dbserver.add_item(
                        item_name='KDE_a', item=self.a, data_shape=self.a.shape
                    )
                    self.dbserver.add_item(
                        item_name='KDE_Z', item=self.Z, data_shape=self.Z.shape
                    )
                    self.dbserver.add_item(
                        item_name='KDE_Eigen',
                        item=self.eigenKDE,
                        data_shape=self.eigenKDE.shape,
                    )
                    self.logfile.write_msg(
                        msg='PLoM.RunAlgorithm: KDE_g, KDE_m, KDE_a, KDE_Z, and KDE_Eigen saved.',
                        msg_type='RUNNING',
                        msg_level=0,
                    )
            elif cur_task.task_name == 'ISDEGeneration':
                self.__getattribute__(  # noqa: PLC2801
                    'task_' + cur_task.task_name
                ).avail_var_list = []
                # ISDE generation
                self.ISDEGeneration(
                    n_mc=n_mc,
                    tol_PCA2=tol_PCA2,
                    tol=tol,
                    max_iter=max_iter,
                    seed_num=seed_num,
                )
                self.logfile.write_msg(
                    msg='PLoM.RunAlgorithm: Realizations generated.',
                    msg_type='RUNNING',
                    msg_level=0,
                )
                self.dbserver.add_item(
                    item_name='X_new',
                    col_names=list(self.X0.columns),
                    item=self.Xnew.T,
                    data_shape=self.Xnew.shape,
                )
                self.logfile.write_msg(
                    msg='PLoM.RunAlgorithm: X_new saved.',
                    msg_type='RUNNING',
                    msg_level=0,
                )
            else:
                self.logfile.write_msg(
                    msg=f'PLoM.RunAlgorithm: task {cur_task.task_name} not found.',
                    msg_type='ERROR',
                    msg_level=0,
                )
                break
            # refresh status
            for cur_item in TASK_ITEM_MAP[cur_task.task_name]:  # noqa: F405
                if '/' + cur_item in self.dbserver.get_name_list():
                    self.__getattribute__(  # noqa: PLC2801
                        'task_' + cur_task.task_name
                    ).avail_var_list.append(cur_item)
            if not cur_task.refresh_status():
                self.logfile.write_msg(
                    msg=f'PLoM.RunAlgorithm: simulation stopped with task {cur_task.task_name} not fully completed.',
                    msg_type='ERROR',
                    msg_level=0,
                )
                break
            # move to the next task
            cur_task = cur_task.next_task

        if self.task_list.refresh_status():
            self.logfile.write_msg(
                msg='PLoM.RunAlgorithm: simulation completed with task(s) {} done.'.format(
                    '->'.join(self.cur_task_list)
                ),
                msg_type='RUNNING',
                msg_level=0,
            )
        else:
            self.logfile.write_msg(
                msg='PLoM.RunAlgorithm: simulation not fully completed.',
                msg_type='ERROR',
                msg_level=0,
            )

    def DataNormalization(self, X):  # noqa: N802, N803, PLR6301
        """Normalizing the X
        - X: the data matrix to be normalized
        """  # noqa: D205, D400, D401
        # scaling
        X_scaled, alpha, x_min = plom.scaling(X)  # noqa: N806
        x_mean = plom.mean(X_scaled)

        return X_scaled, alpha, x_min, x_mean

    def RunPCA(self, X_origin, epsilon_pca):  # noqa: N802, N803, D102
        # ...PCA...
        (H, mu, phi, errors) = plom.PCA(X_origin, epsilon_pca)  # noqa: N806
        nu = len(H)
        self.logfile.write_msg(
            msg=f'PLoM.RunPCA: considered number of PCA components = {nu}',
            msg_type='RUNNING',
            msg_level=0,
        )
        """
        if self.plot_tag:
            fig, ax = plt.subplots(figsize=(8,6))
            ctp = ax.contourf(plom.covariance(H), cmap=plt.cm.bone, levels=100)
            ax.set_xticks(list(range(nu)))
            ax.set_yticks(list(range(nu)))
            ax.set_xticklabels(['PCA-'+str(x+1) for x in range(nu)], fontsize=8, rotation=45)
            ax.set_yticklabels(['PCA-'+str(x+1) for x in range(nu)], fontsize=8, rotation=45)
            ax.set_title('Covariance matrix of PCA')
            cbar = fig.colorbar(ctp)
            plt.savefig(os.path.join(self.vl_path,'PCA_CovarianceMatrix.png'),dpi=480)
            self.logfile.write_msg(msg='PLoM: {} saved in {}.'.format('PCA_CovarianceMatrix.png',self.vl_path),msg_type='RUNNING',msg_level=0)
        """
        return H, mu, phi, nu, errors

    def RunKDE(self, X, epsilon_kde):  # noqa: N802, N803, PLR6301
        """Running Kernel Density Estimation
        - X: the data matrix to be reduced
        - epsilon_kde: smoothing parameter in the kernel density estimation
        """  # noqa: D205, D400, D401
        (s_v, c_v, hat_s_v) = plom.parameters_kde(X)
        K, b = plom.K(X, epsilon_kde)  # noqa: N806

        return s_v, c_v, hat_s_v, K, b

    def DiffMaps(self, H, K, b, tol=0.1):  # noqa: N802, N803, D102
        # ..diff maps basis...
        # self.Z = PCA(self.H)
        try:
            g, eigenvalues = plom.g(K, b)  # diffusion maps
            g = g.real
            m = plom.m(eigenvalues, tol=tol)
            a = g[:, 0:m].dot(np.linalg.inv(np.transpose(g[:, 0:m]).dot(g[:, 0:m])))
            Z = H.dot(a)  # noqa: N806
            """
            if self.plot_tag:
                fig, ax = plt.subplots(figsize=(6,4))
                ax.semilogy(np.arange(len(eigenvalues)), eigenvalues)
                ax.set_xlabel('Eigen number')
                ax.set_ylabel('Eigen value')
                ax.set_title('Eigen value (KDE)')
                plt.savefig(os.path.join(self.vl_path,'KDE_EigenValue.png'),dpi=480)
                self.logfile.write_msg(msg='PLoM: {} saved in {}.'.format('KDE_EigenValue.png',self.vl_path),msg_type='RUNNING',msg_level=0)
            """
        except:  # noqa: E722
            g = None
            m = 0
            a = None
            Z = None  # noqa: N806
            eigenvalues = []
            self.logfile.write_msg(
                msg='PLoM.DiffMaps: diffusion maps failed.',
                msg_type='ERROR',
                msg_level=0,
            )

        return g, m, a, Z, eigenvalues

    def ISDEGeneration(  # noqa: N802
        self,
        n_mc=5,
        tol_PCA2=1e-5,  # noqa: N803
        tol=0.02,
        max_iter=50,
        seed_num=None,
    ):
        """The construction of a nonlinear Ito Stochastic Differential Equation (ISDE) to generate realizations of random variable H"""  # noqa: D400, D401
        if seed_num:
            np.random.seed(seed_num)
        # constraints
        if self.g_c:
            self.C_h_hat_eta = plom.covariance(
                self.g_c(self.x_mean + (self.phi).dot(np.diag(self.mu)).dot(self.H))
            )

            # scaling beta
            # self.beta_c_normalized = self.beta_c_aux(self.beta_c, self.x_min, self.alpha)
            # KZ, 07/24
            self.beta_c_normalized = self.beta_c_aux(self.beta_c, self.X)
            self.b_c, self.psi = plom.PCA2(
                self.C_h_hat_eta, self.beta_c_normalized, tol_PCA2
            )
            self.nu_c = len(self.b_c)

            self.hessian = plom.hessian_gamma(
                self.H, self.psi, self.g_c, self.phi, self.mu, self.x_mean
            )
            self.inverse = plom.solve_inverse(self.hessian)

            self.gradient = plom.gradient_gamma(
                self.b_c, self.H, self.g_c, self.phi, self.mu, self.psi, self.x_mean
            )

            self.lambda_i = -(self.inverse).dot(self.gradient)

            self.errors = [plom.err(self.gradient, self.b_c)]
            iteration = 0
            nu_init = np.random.normal(size=(int(self.nu), int(self.N)))
            self.Y = nu_init.dot(self.a)

            error_ratio = 0
            increasing_iterations = 0

            while (
                iteration < max_iter
                and self.errors[iteration] > tol * self.errors[0]
                and (increasing_iterations < 3)  # noqa: PLR2004
            ):
                self.logfile.write_msg(
                    msg=f'PLoM.ISDEGeneration: running iteration {iteration + 1}.',
                    msg_type='RUNNING',
                    msg_level=0,
                )
                Hnewvalues, nu_lambda, x_, x_2 = plom.generator(  # noqa: N806
                    self.Z,
                    self.Y,
                    self.a,
                    n_mc,
                    self.x_mean,
                    self.H,
                    self.s_v,
                    self.hat_s_v,
                    self.mu,
                    self.phi,
                    self.g[:, 0 : int(self.m)],
                    psi=self.psi,
                    lambda_i=self.lambda_i,
                    g_c=self.g_c,
                    D_x_g_c=self.D_x_g_c,
                )  # solve the ISDE in n_mc iterations

                self.gradient = plom.gradient_gamma(
                    self.b_c,
                    Hnewvalues,
                    self.g_c,
                    self.phi,
                    self.mu,
                    self.psi,
                    self.x_mean,
                )
                self.hessian = plom.hessian_gamma(
                    Hnewvalues, self.psi, self.g_c, self.phi, self.mu, self.x_mean
                )
                self.inverse = plom.solve_inverse(self.hessian)

                self.lambda_i = self.lambda_i - 0.3 * (self.inverse).dot(  # noqa: PLR6104
                    self.gradient
                )

                self.Z = Hnewvalues[:, -self.N :].dot(self.a)
                self.Y = nu_lambda[:, -self.N :].dot(self.a)
                iteration += 1

                (self.errors).append(plom.err(self.gradient, self.b_c))

                if error_ratio > 1.00:
                    increasing_iterations += 1
                else:
                    increasing_iterations = 0

            # saving data
            self.dbserver.add_item(
                item_name='Errors',
                item=np.array(self.errors),
                data_shape=np.array(self.errors).shape,
            )

            if iteration == max_iter:
                self.logfile.write_msg(
                    msg='PLoM.ISDEGeneration: max. iteration reached and convergence not achieved.',
                    msg_type='WARNING',
                    msg_level=0,
                )

        # no constraints
        else:
            nu_init = np.random.normal(size=(int(self.nu), int(self.N)))
            self.Y = nu_init.dot(self.a)
            Hnewvalues, nu_lambda, x_, x_2 = plom.generator(  # noqa: F841, N806
                self.Z,
                self.Y,
                self.a,
                n_mc,
                self.x_mean,
                self.H,
                self.s_v,
                self.hat_s_v,
                self.mu,
                self.phi,
                self.g[:, 0 : int(self.m)],
                seed_num=seed_num,
            )  # solve the ISDE in n_mc iterations
            self.logfile.write_msg(
                msg='PLoM.ISDEGeneration: new generations are simulated.',
                msg_type='RUNNING',
                msg_level=0,
            )
            self.dbserver.add_item(item_name='Errors', item=np.array([0]))

            # saving data
            self.errors = []
            self.dbserver.add_item(
                item_name='Errors',
                item=np.array(self.errors),
                data_shape=np.array(self.errors).shape,
            )

        self.Xnew = self.x_mean + self.phi.dot(np.diag(self.mu)).dot(Hnewvalues)

        # unscale
        self.Xnew = np.diag(self.alpha).dot(self.Xnew) + self.x_min

    def export_results(self, data_list=[], file_format_list=['csv']):  # noqa: B006
        """Exporting results by the data names
        - data_list: list of data names
        - file_format_list: list of output formats
        """  # noqa: D205, D400, D401
        avail_name_list = self.dbserver.get_name_list()
        if not data_list:
            # print available data names
            avail_name_str = ','.join(avail_name_list)
            self.logfile.write_msg(
                msg=f'PLoM.export_results: available data {avail_name_str}.',
                msg_type='RUNNING',
                msg_level=0,
            )
        if not avail_name_list:
            # empty database
            self.logfile.write_msg(
                msg='PLoM.export_results: database is empty - no data exported.',
                msg_type='ERROR',
                msg_level=0,
            )
        else:
            for tag, data_i in enumerate(data_list):
                if data_i not in avail_name_list:
                    self.logfile.write_msg(
                        msg=f'PLoM.export_results: {data_i} is not found and skipped.',
                        msg_type='WARNING',
                        msg_level=0,
                    )
                else:
                    try:
                        ff_i = file_format_list[tag]
                    except:  # noqa: E722
                        ff_i = file_format_list[-1]
                    ex_flag = self.dbserver.export(
                        data_name=data_i, file_format=ff_i
                    )
                    if type(ex_flag) == int and ex_flat == 1:  # noqa: E721, F405
                        self.logfile.write_msg(
                            msg=f'PLoM.export_results: {data_i} is not found and skipped.',
                            msg_type='WARNING',
                            msg_level=0,
                        )
                    elif type(ex_flag) == int and ex_flag == 2:  # noqa: E721, PLR2004
                        self.logfile.write_msg(
                            msg=f'PLoM.export_results: {ff_i} is not supported yest.',
                            msg_type='ERROR',
                            msg_level=0,
                        )
                    else:
                        self.logfile.write_msg(
                            msg=f'PLoM.export_results: {data_i} is exported in {ex_flag}.',
                            msg_type='RUNNING',
                            msg_level=0,
                        )

    """
    def PostProcess():
    	#...output plots...

    	#plot some histograms
        import matplotlib.patches as mpatches
        plt.plot(Xnewvalues[0,:], Xnewvalues[1,:], 'rx')
        plt.plot(Xvalues[0], Xvalues[1], 'bo')
        plt.xlabel('x')
        plt.ylabel('y')
        red_patch = mpatches.Patch(color='red', label='X_c')
        blue_patch = mpatches.Patch(color='blue', label='X')
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()

        import matplotlib.patches as mpatches
        plt.xlabel('x1')
        plt.subplot(2,1,1)
        plt.hist(Xnewvalues[0], bins = 100, color = 'red')
        plt.subplot(2,1,2)
        plt.hist(Xvalues[0], bins = 100, color = 'blue')
        plt.title('Histogram')
        red_patch = mpatches.Patch(color='red', label='X_c')
        blue_patch = mpatches.Patch(color='blue', label='X')
        plt.legend(handles=[red_patch, blue_patch])
        plt.shlow()

        import matplotlib.patches as mpatches
        plt.xlabel('x2')
        plt.subplot(2,1,1)
        plt.hist(Xnewvalues[1], bins = 100, color = 'red')
        plt.subplot(2,1,2)
        plt.hist(Xvalues[1], bins = 100, color = 'blue')
        plt.title('Histogram')
        red_patch = mpatches.Patch(color='red', label='X new realizations')
        blue_patch = mpatches.Patch(color='blue', label='X')
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()

        import matplotlib.patches as mpatches
        plt.xlabel('x3')
        plt.subplot(2,1,1)
        plt.hist(Xnewvalues[2], bins = 100, color = 'red')
        plt.subplot(2,1,2)
        plt.hist(Xvalues[2], bins = 100, color = 'blue')
        plt.title('Histogram')
        red_patch = mpatches.Patch(color='red', label='X new realizations')
        blue_patch = mpatches.Patch(color='blue', label='X')
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()

        import matplotlib.patches as mpatches
        plt.xlabel('x4')
        plt.subplot(2,1,1)
        plt.hist(Xnewvalues[3], bins = 100, color = 'red')
        plt.subplot(2,1,2)
        plt.hist(Xvalues[3], bins = 100, color = 'blue')
        plt.title('Histogram')
        red_patch = mpatches.Patch(color='red', label='X new realizations')
        blue_patch = mpatches.Patch(color='blue', label='X')
        plt.legend(handles=[red_patch, blue_patch])
        plt.show()
    """  # noqa: E101
