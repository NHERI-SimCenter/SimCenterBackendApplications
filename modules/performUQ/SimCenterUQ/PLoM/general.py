# Constants, variables, and methods that are commonly used  # noqa: CPY001, D100

import os
from collections import Counter
from datetime import datetime

import pandas as pd

ITEM_LIST_DATANORM = ['X_range', 'X_min', 'X_scaled', 'X_scaled_mean']
ITEM_LIST_RUNPCA = [
    'X_PCA',
    'EigenValue_PCA',
    'EigenVector_PCA',
    'NumComp_PCA',
    'Error_PCA',
]
ITEM_LIST_RUNKDE = [
    's_v',
    'c_v',
    'hat_s_v',
    'X_KDE',
    'EigenValues_KDE',
    'KDE_g',
    'KDE_m',
    'KDE_a',
    'KDE_Z',
    'KDE_Eigen',
]
ITEM_LIST_ISDEGENE = ['Errors', 'X_new']
ITEM_LIST = (
    ['basic']  # noqa: RUF005
    + ['constraints_file']
    + ['X0', 'N', 'n']
    + ITEM_LIST_DATANORM
    + ITEM_LIST_RUNPCA
    + ITEM_LIST_RUNKDE
    + ITEM_LIST_ISDEGENE
)  # all variables in the database
ITEM_ADDS = ['/' + x for x in ITEM_LIST]  # HDFStore ABSOLUTE path-names
ATTR_LIST = [
    None,
    None,
    'X',
    'N',
    'n',
    'alpha',
    'x_min',
    'X_scaled',
    'x_mean',
    'H',
    'mu',
    'phi',
    'nu',
    'errPCA',
    's_v',
    'c_v',
    'hat_s_v',
    'K',
    'b',
    'g',
    'm',
    'a',
    'Z',
    'eigenKDE',
    'errors',
    'Xnew',
]
ATTR_MAP = dict(zip(ITEM_ADDS, ATTR_LIST))
FULL_TASK_LIST = ['DataNormalization', 'RunPCA', 'RunKDE', 'ISDEGeneration']
TASK_ITEM_MAP = {
    'DataNormalization': ITEM_LIST_DATANORM,
    'RunPCA': ITEM_LIST_RUNPCA,
    'RunKDE': ITEM_LIST_RUNKDE,
    'ISDEGeneration': ITEM_LIST_ISDEGENE,
}


class Logfile:  # noqa: D101
    def __init__(self, logfile_dir='./', logfile_name='plom.log', screen_msg=True):  # noqa: ANN001, ANN204, FBT002
        """Initializing the logfile
        - logfile_dir: default is the same path of the PLoM package
        - logfile_name: default is the "plom.log"
        - screen_msg: default is to show message on screen
        """  # noqa: D205, D400, D401
        self.logfile_dir = logfile_dir
        self.logfile_name = logfile_name
        self.logfile_path = os.path.join(self.logfile_dir, self.logfile_name)  # noqa: PTH118
        self.screen_msg = screen_msg
        # start the log
        self.write_msg(msg='--NEW LOG STARTING FROM THIS LINE--', mode='w')

    def write_msg(self, msg='', msg_type='RUNNING', msg_level=0, mode='a'):  # noqa: ANN001, ANN201
        """Writing running messages
        - msg: the message
        - msg_type: the type of message 'RUNNING', 'WARNING', 'ERROR'
        - msg_level: how many indent tags
        """  # noqa: D205, D400, D401
        indent_tabs = ''.join(['\t'] * msg_level)
        decorated_msg = f'{datetime.utcnow()} {indent_tabs} {msg_type}-MSG {msg} '  # noqa: DTZ003
        if self.screen_msg:
            print(decorated_msg)  # noqa: T201
        with open(self.logfile_path, mode) as f:  # noqa: PTH123
            f.write('\n' + decorated_msg)

    def delete_logfile(self):  # noqa: ANN201
        """Deleting the log file"""  # noqa: D400, D401
        if os.path.exists(self.logfile_path):  # noqa: PTH110
            os.remove(self.logfile_path)  # noqa: PTH107
        else:
            print(f'The logfile {self.logfile_path} does not exist.')  # noqa: T201


class DBServer:  # noqa: D101
    def __init__(self, db_dir='./', db_name='plom.h5'):  # noqa: ANN001, ANN204
        """Initializing the database
        - db_dir: default is the same path of the PLoM package
        - db_name: default is "plom.h5"
        """  # noqa: D205, D400, D401
        self.db_dir = db_dir
        self.db_name = db_name
        self.db_path = os.path.join(self.db_dir, self.db_name)  # noqa: PTH118
        if os.path.exists(self.db_path):  # noqa: PTH110
            # deleting the old database
            os.remove(self.db_path)  # noqa: PTH107
        self.init_time = datetime.utcnow()  # noqa: DTZ003
        self.item_name_list = []
        self.basic()
        self.dir_export = self._create_export_dir()
        self._item_list = ITEM_LIST
        self._item_adds = ITEM_ADDS

    def basic(self):  # noqa: ANN201
        """Writing basic info"""  # noqa: D400, D401
        df = pd.DataFrame.from_dict(  # noqa: PD901
            {
                'InitializedTime': [self.init_time],
                'LastEditedTime': [datetime.utcnow()],  # noqa: DTZ003
                'DBName': [self.db_name],
            },
            dtype=str,
        )
        store = pd.HDFStore(self.db_path, 'a')
        df.to_hdf(store, 'basic', mode='a')
        store.close()
        self.add_item(item=[''], data_type='ConstraintsFile')

    def _create_export_dir(self):  # noqa: ANN202
        """Creating a export folder"""  # noqa: D400, D401
        dir_export = os.path.join(self.db_dir, 'DataOut')  # noqa: PTH118
        try:
            os.makedirs(dir_export, exist_ok=True)  # noqa: PTH103
            return dir_export  # noqa: TRY300
        except:  # noqa: E722
            return None

    def get_item_adds(self):  # noqa: ANN201
        """Returning the full list of data items"""  # noqa: D400, D401
        return self._item_adds

    def add_item(  # noqa: ANN201
        self,
        item_name=None,  # noqa: ANN001
        col_names=None,  # noqa: ANN001
        item=[],  # noqa: ANN001, B006
        data_shape=None,  # noqa: ANN001
        data_type='Data',  # noqa: ANN001
    ):
        """Adding a new data item into database"""  # noqa: D400
        if data_type == 'Data':
            if item.size > 1:
                df = pd.DataFrame(item, columns=col_names)  # noqa: PD901
                dshape = pd.DataFrame(data_shape, columns=['DS_' + item_name])
            else:
                if col_names is None:
                    col_names = item_name
                df = pd.DataFrame.from_dict({col_names: item.tolist()})  # noqa: PD901
                dshape = pd.DataFrame.from_dict({'DS_' + col_names: (1,)})
            if item_name is not None:  # noqa: RET503
                store = pd.HDFStore(self.db_path, 'a')
                # data item
                df.to_hdf(store, item_name, mode='a')
                # data shape
                dshape.to_hdf(store, 'DS_' + item_name, mode='a')
                store.close()  # noqa: RET503
        elif data_type == 'ConstraintsFile':
            # constraints filename
            cf = pd.DataFrame.from_dict({'ConstraintsFile': item}, dtype=str)
            store = pd.HDFStore(self.db_path, 'a')
            cf.to_hdf(store, 'constraints_file', mode='a')
            store.close()  # noqa: RET503
        else:
            # Not supported data_type
            return False

    def get_item(self, item_name=None, table_like=False, data_type='Data'):  # noqa: ANN001, ANN201, FBT002
        """Getting a specific data item"""  # noqa: D400, D401
        if data_type == 'Data':  # noqa: RET503
            if item_name is not None:  # noqa: RET503
                store = pd.HDFStore(self.db_path, 'r')
                try:
                    item = store.get(item_name)
                    item_shape = tuple(
                        [
                            x[0]
                            for x in self.get_item_shape(  # noqa: PD011
                                item_name=item_name
                            ).values.tolist()
                        ]
                    )
                    if not table_like:
                        item = item.to_numpy().reshape(item_shape)
                except:  # noqa: E722
                    item = None
                finally:
                    store.close()

                return item
        elif data_type == 'ConstraintsFile':
            store = pd.HDFStore(self.db_path, 'r')
            try:
                item = store.get('/constraints_file')
            except:  # noqa: E722
                item = None
            finally:
                store.close()

            return item.values.tolist()[0][0]  # noqa: PD011

    def remove_item(self, item_name=None):  # noqa: ANN001, ANN201
        """Removing an item"""  # noqa: D400, D401
        if item_name is not None:
            store = pd.HDFStore(self.db_path, 'r')
            try:
                store.remove(item_name)
            except:  # noqa: E722
                item = None  # noqa: F841
            finally:
                store.close()

    def get_item_shape(self, item_name=None):  # noqa: ANN001, ANN201
        """Getting the shape of a specific data item"""  # noqa: D400, D401
        if item_name is not None:  # noqa: RET503
            store = pd.HDFStore(self.db_path, 'r')
            try:
                item_shape = store.get('DS_' + item_name)
            except:  # noqa: E722
                item_shape = None
            store.close()

            return item_shape

    def get_name_list(self):  # noqa: ANN201
        """Returning the keys of the database"""  # noqa: D400, D401
        store = pd.HDFStore(self.db_path, 'r')
        try:
            name_list = store.keys()
        except:  # noqa: E722
            name_list = []
        store.close()
        return name_list

    def export(self, data_name=None, filename=None, file_format='csv'):  # noqa: ANN001, ANN201
        """Exporting the specific data item
        - data_name: data tag
        - format: data format
        """  # noqa: D205, D400, D401
        d = self.get_item(item_name=data_name[1:], table_like=True)
        if d is None:
            return 1
        if filename is None:
            filename = os.path.join(  # noqa: PTH118
                self.dir_export, str(data_name).replace('/', '') + '.' + file_format
            )
        else:
            filename = os.path.join(  # noqa: PTH118
                self.dir_export, filename.split('.')[0] + '.' + file_format
            )
        if file_format == 'csv' or 'txt':  # noqa: SIM222
            d.to_csv(filename, header=True, index=True)
        elif file_format == 'json':
            with open(filename, 'w', encoding='utf-8') as f:  # noqa: PTH123
                json.dump(d, f)  # noqa: F821
        else:
            return 2
        return filename


class Task:
    """This is a class for managering an individual task in
    the PLoM running process
    """  # noqa: D205, D400, D404

    def __init__(self, task_name=None):  # noqa: ANN001, ANN204
        """Initialization
        - task_name: name of the task
        """  # noqa: D205, D400, D401
        self.task_name = task_name  # task name
        self.pre_task = None  # previous task
        self.next_task = None  # next task
        self.full_var_list = []  # key variable list
        self.avail_var_list = []  # current available variables
        self.status = False  # task status

    def refresh_status(self):  # noqa: ANN201
        """Refreshing the current status of the task
        If any of the previous tasks is not completed, the current task is also not reliable
        """  # noqa: D205, D400, D401
        # check the previous task if any
        if self.pre_task:
            if not self.pre_task.refresh_status():
                # previous task not completed -> this task also needs to rerun
                self.status = False

                return self.status

        # self-check
        if Counter(self.avail_var_list) == Counter(self.full_var_list) and len(
            self.avail_var_list
        ):
            self.status = True  # not finished
        else:
            self.status = False  # finished

        return self.status


class TaskList:
    """This is a class for managering a set of tasks
    in a specific order
    """  # noqa: D205, D400, D404

    def __init__(self):  # noqa: ANN204
        self.head_task = None  # first task
        self.tail_task = None  # last task
        self.status = False  # status

    def add_task(self, new_task=None):  # noqa: ANN001, ANN201, D102
        if new_task is None:
            self.head_task = None
            return
        elif self.head_task is None:  # noqa: RET505
            # first task
            self.head_task = new_task
            self.tail_task = new_task
        else:
            # adding a new to the current list
            new_task.pre_task = self.tail_task
            self.tail_task.next_task = new_task
            self.tail_task = new_task

    def refresh_status(self):  # noqa: ANN201
        """Refreshing the tasks' status"""  # noqa: D400, D401
        if self.head_task:  # noqa: RET503
            cur_task = self.head_task
            if not cur_task.status:
                self.status = False

                return self.status
            while cur_task.next_task:
                cur_task = cur_task.next_task
                if not cur_task.status:
                    self.status = False

                    return self.status

            self.status = True

            return self.status
