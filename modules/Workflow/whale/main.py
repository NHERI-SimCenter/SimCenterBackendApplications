# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 The Regents of the University of California
# Copyright (c) 2019 Leland Stanford Junior University
#
# This file is part of whale.
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
# whale. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Frank McKenna
# Adam Zsarnóczay
# Wael Elhaddad
# Michael Gardner
# Chaofeng Wang
# Stevan Gavrilovic
# Jinyan Zhao
# Sina Naeimi

"""
This module has classes and methods that handle everything at the moment.

.. rubric:: Contents

.. autosummary::

    ...

"""

from time import strftime
from datetime import datetime
import sys, os, json
import argparse
import importlib

import pprint
import shlex

import shutil
import subprocess

from copy import deepcopy

import warnings
import posixpath

import numpy as np
import pandas as pd

import platform
from pathlib import Path, PurePath

import shapely.wkt, shapely.geometry

# import posixpath
# import ntpath


pp = pprint.PrettyPrinter(indent=4)

# get the absolute path of the whale directory
whale_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def str2bool(v):
    # courtesy of Maxim @ stackoverflow

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Options(object):
    def __init__(self):
        self._log_show_ms = False
        self._print_log = False

        self.reset_log_strings()

    @property
    def log_show_ms(self):
        return self._log_show_ms

    @log_show_ms.setter
    def log_show_ms(self, value):
        self._log_show_ms = bool(value)

        self.reset_log_strings()

    @property
    def log_pref(self):
        return self._log_pref

    @property
    def log_div(self):
        return self._log_div

    @property
    def log_time_format(self):
        return self._log_time_format

    @property
    def log_file(self):
        return globals()['log_file']

    @log_file.setter
    def log_file(self, value):
        if value is None:
            globals()['log_file'] = value

        else:
            filepath = Path(value).resolve()

            try:
                globals()['log_file'] = str(filepath)

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write('')

            except:
                raise ValueError(
                    f'The filepath provided does not point to a '
                    f'valid location: {filepath}'
                )

    @property
    def print_log(self):
        return self._print_log

    @print_log.setter
    def print_log(self, value):
        self._print_log = str2bool(value)

    def reset_log_strings(self):
        if self._log_show_ms:
            self._log_time_format = '%H:%M:%S:%f'
            self._log_pref = (
                ' ' * 16
            )  # the length of the time string in the log file
            self._log_div = '-' * (
                80 - 17
            )  # to have a total length of 80 with the time added
        else:
            self._log_time_format = '%H:%M:%S'
            self._log_pref = ' ' * 9
            self._log_div = '-' * (80 - 9)


options = Options()

log_file = None


def set_options(config_options):
    if config_options is not None:
        for key, value in config_options.items():
            if key == 'LogShowMS':
                options.log_show_ms = value
            elif key == 'LogFile':
                options.log_file = value
            elif key == 'PrintLog':
                options.print_log = value


# Monkeypatch warnings to get prettier messages
def _warning(message, category, filename, lineno, file=None, line=None):
    if '\\' in filename:
        file_path = filename.split('\\')
    elif '/' in filename:
        file_path = filename.split('/')
    python_file = '/'.join(file_path[-3:])
    print('WARNING in {} at line {}\n{}\n'.format(python_file, lineno, message))


warnings.showwarning = _warning


def log_div(prepend_timestamp=False, prepend_blank_space=True):
    """
    Print a divider line to the log file

    """

    if prepend_timestamp:
        msg = options.log_div

    elif prepend_blank_space:
        msg = options.log_div

    else:
        msg = '-' * 80

    log_msg(
        msg,
        prepend_timestamp=prepend_timestamp,
        prepend_blank_space=prepend_blank_space,
    )


def log_msg(msg='', prepend_timestamp=True, prepend_blank_space=True):
    """
    Print a message to the screen with the current time as prefix

    The time is in ISO-8601 format, e.g. 2018-06-16T20:24:04Z

    Parameters
    ----------
    msg: string
       Message to print.

    """

    msg_lines = msg.split('\n')

    for msg_i, msg_line in enumerate(msg_lines):
        if prepend_timestamp and (msg_i == 0):
            formatted_msg = '{} {}'.format(
                datetime.now().strftime(options.log_time_format), msg_line
            )
        elif prepend_timestamp:
            formatted_msg = options.log_pref + msg_line
        elif prepend_blank_space:
            formatted_msg = options.log_pref + msg_line
        else:
            formatted_msg = msg_line

        if options.print_log:
            print(formatted_msg)

        if globals()['log_file'] is not None:
            with open(globals()['log_file'], 'a', encoding='utf-8') as f:
                f.write('\n' + formatted_msg)


def log_error(msg):
    """
    Print an error message to the screen

    Parameters
    ----------
    msg: string
       Message to print.
    """

    log_div()
    log_msg('' * (80 - 21 - 6) + ' ERROR')
    log_msg(msg)
    log_div()


def print_system_info():
    log_msg(
        'System Information:', prepend_timestamp=False, prepend_blank_space=False
    )
    log_msg(
        f'  local time zone: {datetime.utcnow().astimezone().tzinfo}\n'
        f'  start time: {datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}\n'
        f'  python: {sys.version}\n'
        f'  numpy: {np.__version__}\n'
        f'  pandas: {pd.__version__}\n',
        prepend_timestamp=False,
        prepend_blank_space=False,
    )


def create_command(command_list, enforced_python=None):
    """
    Short description

    Long description

    Parameters
    ----------
    command_list: array of unicode strings
        Explain...
    """
    if command_list[0] == 'python':
        # replace python with...
        if enforced_python is None:
            # the full path to the python interpreter
            python_exe = sys.executable
        else:
            # the prescribed path to the python interpreter
            python_exe = enforced_python

        command = '"{}" "{}" '.format(
            python_exe, command_list[1]
        )  # + ' '.join(command_list[2:])

        for command_arg in command_list[2:]:
            command += '"{}" '.format(command_arg)
    else:
        command = '"{}" '.format(command_list[0])  # + ' '.join(command_list[1:])

        for command_arg in command_list[1:]:
            command += '"{}" '.format(command_arg)

    return command


def run_command(command):
    """
    Short description

    Long description

    Parameters
    ----------
    command_list: array of unicode strings
        Explain...

    """

    # If it is a python script, we do not run it, but rather import the main
    # function. This ensures that the script is run using the same python
    # interpreter that this script uses and it is also faster because we do not
    # need to run multiple python interpreters simultaneously.
    Frank_trusts_this_approach = False
    if command[:6] == 'python' and Frank_trusts_this_approach:
        import importlib  # only import this when it's needed

        command_list = command.split()[1:]
        # py_args = command_list[1:]

        # get the dir and file name
        py_script_dir, py_script_file = os.path.split(command_list[0][1:-1])

        # add the dir to the path
        sys.path.insert(0, py_script_dir)

        # import the file
        py_script = importlib.__import__(
            py_script_file[:-3],
            globals(),
            locals(),
            [
                'main',
            ],
            0,
        )

        # remove the quotes from the arguments
        arg_list = [c[1:-1] for c in command_list[1:]]

        py_script.main(arg_list)

        return '', ''

    else:
        # fmk with Shell=True not working on older windows machines, new approach needed for quoted command .. turn into a list
        command = shlex.split(command)

        try:
            result = subprocess.check_output(
                command, stderr=subprocess.STDOUT, text=True
            )
            returncode = 0
        except subprocess.CalledProcessError as e:
            result = e.output
            returncode = e.returncode

        if returncode != 0:
            log_error('return code: {}'.format(returncode))

        # if platform.system() == 'Windows':
        #     return result.decode(sys.stdout.encoding), returncode
        # else:
        #     #print(result, returncode)
        #     return str(result), returncode

        return result, returncode


def show_warning(warning_msg):
    warnings.warn(UserWarning(warning_msg))


def resolve_path(target_path, ref_path):
    ref_path = Path(ref_path)

    target_path = str(target_path).strip()

    while target_path.startswith('/') or target_path.startswith('\\'):
        target_path = target_path[1:]

    if target_path == '':
        target_path = ref_path

    else:
        target_path = Path(target_path)

        if not target_path.exists():
            target_path = Path(ref_path) / target_path

        if target_path.exists():
            target_path = target_path.resolve()
        else:
            # raise ValueError(
            #    f"{target_path} does not point to a valid location")
            print(f'{target_path} does not point to a valid location')

    return target_path


def _parse_app_registry(registry_path, app_types, list_available_apps=False):
    """
    Load the information about available workflow applications.

    Parameters
    ----------
    registry_path: string
        Path to the JSON file with the app registry information. By default,
        this file is stored at applications/Workflow/WorkflowApplications.json
    app_types: list of strings
        List of application types (e.g., Assets, Modeling, DL) to parse from the
        registry
    list_available_apps: bool, optional, default: False
        If True, all available applications of the requested types are printed
        in the log file.

    Returns
    -------
    app_registry: dict
        A dictionary with WorkflowApplication objects. Primary keys are
        the type of application (e.g., Assets, Modeling, DL); secondary keys
        are the name of the specific application (e.g, MDOF-LU). See the
        documentation for more details.
    default_values: dict
        Default values of filenames used to pass data between applications. Keys
        are the placeholder names (e.g., filenameAIM) and values are the actual
        filenames (e.g,. AIM.json)
    """

    log_msg('Parsing application registry file')

    # open the registry file
    log_msg('Loading the json file...', prepend_timestamp=False)
    with open(registry_path, 'r', encoding='utf-8') as f:
        app_registry_data = json.load(f)
    log_msg('  OK', prepend_timestamp=False)

    # initialize the app registry
    app_registry = dict([(a, dict()) for a in app_types])

    log_msg('Loading default values...', prepend_timestamp=False)

    default_values = app_registry_data.get('DefaultValues', None)

    log_msg('  OK', prepend_timestamp=False)

    log_msg('Collecting application data...', prepend_timestamp=False)
    # for each application type
    for app_type in sorted(app_registry.keys()):
        # if the registry contains information about it
        app_type_long = app_type + 'Applications'
        if app_type_long in app_registry_data:
            # get the list of available applications
            available_apps = app_registry_data[app_type_long]['Applications']
            api_info = app_registry_data[app_type_long]['API']

            # add the default values to the API info
            if default_values is not None:
                api_info.update({'DefaultValues': default_values})

            # and create a workflow application for each app of this type
            for app in available_apps:
                app_registry[app_type][app['Name']] = WorkflowApplication(
                    app_type=app_type, app_info=app, api_info=api_info
                )

    log_msg('  OK', prepend_timestamp=False)

    if list_available_apps:
        log_msg('Available applications:', prepend_timestamp=False)

        for app_type, app_list in app_registry.items():
            for app_name, app_object in app_list.items():
                log_msg(
                    '  {} : {}'.format(app_type, app_name), prepend_timestamp=False
                )

    # pp.pprint(self.app_registry)

    log_msg('Successfully parsed application registry', prepend_timestamp=False)
    log_div()

    return app_registry, default_values


class WorkFlowInputError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class WorkflowApplication(object):
    """
    Short description.


    Longer description.

    Parameters
    ----------

    """

    def __init__(self, app_type, app_info, api_info):
        # print('APP_TYPE', app_type)
        # print('APP_INFO', app_info)
        # print('API_INFO', api_info)
        # print('APP_RELPATH', app_info['ExecutablePath'])

        self.name = app_info['Name']
        self.app_type = app_type
        self.rel_path = app_info['ExecutablePath']

        if 'RunsParallel' in app_info.keys():
            self.runsParallel = app_info['RunsParallel']
        else:
            self.runsParallel = False

        self.app_spec_inputs = app_info.get('ApplicationSpecificInputs', [])

        self.inputs = api_info['Inputs']
        self.outputs = api_info['Outputs']
        if 'DefaultValues' in api_info.keys():
            self.defaults = api_info['DefaultValues']
        else:
            self.defaults = None

    def set_pref(self, preferences, ref_path):
        """
        Short description

        Parameters
        ----------
        preferences: dictionary
            Explain...
        """
        self.pref = preferences

        # parse the relative paths (if any)
        ASI = [inp['id'] for inp in self.app_spec_inputs]
        for preference in list(self.pref.keys()):
            if preference in ASI:
                input_id = np.where([preference == asi for asi in ASI])[0][0]
                input_type = self.app_spec_inputs[input_id]['type']

                if input_type == 'path':
                    if 'PelicunDefault' in self.pref[preference]:
                        continue

                    self.pref[preference] = resolve_path(
                        self.pref[preference], ref_path
                    )

    def get_command_list(self, app_path, force_posix=False):
        """
        Short description

        Parameters
        ----------
        app_path: Path
            Explain...
        """

        abs_path = Path(app_path) / self.rel_path

        # abs_path = posixpath.join(app_path, self.rel_path)

        arg_list = []

        if str(abs_path).endswith('.py'):
            arg_list.append('python')

        if force_posix:
            arg_list.append('{}'.format(abs_path.as_posix()))
        else:
            arg_list.append('{}'.format(abs_path))

        for in_arg in self.inputs:
            arg_list.append('--{}'.format(in_arg['id']))

            # Default values are protected, they cannot be overwritten simply
            # by providing application specific inputs in the config file
            if in_arg['type'] == 'workflowDefault':
                arg_value = self.defaults[in_arg['id']]

                # If the user also provided an input, let them know that their
                # input is invalid
                if in_arg['id'] in self.pref.keys():
                    log_msg(
                        '\nWARNING: Application specific parameters cannot '
                        'overwrite default workflow\nparameters. See the '
                        'documentation on how to edit workflowDefault '
                        'inputs.\n',
                        prepend_timestamp=False,
                        prepend_blank_space=False,
                    )

            elif in_arg['id'] in self.pref.keys():
                arg_value = self.pref[in_arg['id']]

            else:
                arg_value = in_arg['default']

            if isinstance(arg_value, Path) and force_posix:
                arg_list.append('{}'.format(arg_value.as_posix()))
            else:
                arg_list.append('{}'.format(arg_value))

        for out_arg in self.outputs:
            out_id = '--{}'.format(out_arg['id'])

            if out_id not in arg_list:
                arg_list.append(out_id)

                # Default values are protected, they cannot be overwritten simply
                # by providing application specific inputs in the config file
                if out_arg['type'] == 'workflowDefault':
                    arg_value = self.defaults[out_arg['id']]

                    # If the user also provided an input, let them know that
                    # their input is invalid
                    if out_arg['id'] in self.pref.keys():
                        log_msg(
                            '\nWARNING: Application specific parameters '
                            'cannot overwrite default workflow\nparameters. '
                            'See the documentation on how to edit '
                            'workflowDefault inputs.\n',
                            prepend_timestamp=False,
                            prepend_blank_space=False,
                        )

                elif out_arg['id'] in self.pref.keys():
                    arg_value = self.pref[out_arg['id']]

                else:
                    arg_value = out_arg['default']

                if isinstance(arg_value, Path) and force_posix:
                    arg_list.append('{}'.format(arg_value.as_posix()))
                else:
                    arg_list.append('{}'.format(arg_value))

        ASI_list = [inp['id'] for inp in self.app_spec_inputs]
        for pref_name, pref_value in self.pref.items():
            # only pass those input arguments that are in the registry
            if pref_name in ASI_list:
                pref_id = '--{}'.format(pref_name)
                if pref_id not in arg_list:
                    arg_list.append(pref_id)

                    if isinstance(pref_value, Path) and force_posix:
                        arg_list.append('{}'.format(pref_value.as_posix()))
                    else:
                        arg_list.append('{}'.format(pref_value))

        # pp.pprint(arg_list)

        return arg_list


class Workflow(object):
    """
    A class that collects methods common to all workflows developed by the
    SimCenter. Child-classes will be introduced later if needed.

    Parameters
    ----------

    run_type: string
        Explain...
    input_file: string
        Explain...
    app_registry: string
        Explain...

    """

    def __init__(
        self,
        run_type,
        input_file,
        app_registry,
        app_type_list,
        reference_dir=None,
        working_dir=None,
        app_dir=None,
        parType='seqRUN',
        mpiExec='mpiExec',
        numProc=8,
    ):
        log_msg('Inputs provided:')
        log_msg(
            'workflow input file: {}'.format(input_file), prepend_timestamp=False
        )
        log_msg(
            'application registry file: {}'.format(app_registry),
            prepend_timestamp=False,
        )
        log_msg('run type: {}'.format(run_type), prepend_timestamp=False)
        log_div()

        self.optional_apps = ['RegionalEvent', 'Modeling', 'EDP', 'UQ', 'DL', 'FEM']

        # Create the asset registry
        self.asset_type_list = [
            'Buildings',
            'WaterDistributionNetwork',
            'TransportationNetwork',
        ]
        self.asset_registry = dict([(a, dict()) for a in self.asset_type_list])

        self.run_type = run_type
        self.input_file = input_file
        self.app_registry_file = app_registry
        self.modifiedRun = False  # ADAM to fix
        self.parType = parType
        self.mpiExec = mpiExec
        self.numProc = numProc

        # if parallel setup, open script file to run
        self.inputFilePath = os.path.dirname(input_file)
        parCommandFileName = os.path.join(self.inputFilePath, 'sc_parScript.sh')
        if parType == 'parSETUP':
            self.parCommandFile = open(parCommandFileName, 'w')
            self.parCommandFile.write('#!/bin/sh' + '\n')

        print(
            'WF: parType, mpiExec, numProc: ',
            self.parType,
            self.mpiExec,
            self.numProc,
        )
        self.numP = 1
        self.procID = 0
        self.doParallel = False
        if parType == 'parRUN':
            mpi_spec = importlib.util.find_spec('mpi4py')
            found = mpi_spec is not None
            if found:
                import mpi4py
                from mpi4py import MPI

                self.comm = MPI.COMM_WORLD
                self.numP = self.comm.Get_size()
                self.procID = self.comm.Get_rank()
                if self.numP < 2:
                    self.doParallel = False
                    self.numP = 1
                    self.procID = 0
                else:
                    self.doParallel = True

        print(
            'WF: parType, mpiExec, numProc, do? numP, procID: ',
            self.parType,
            self.mpiExec,
            self.numProc,
            self.doParallel,
            self.numP,
            self.procID,
        )

        if reference_dir is not None:
            self.reference_dir = Path(reference_dir)
        else:
            self.reference_dir = None

        if working_dir is not None:
            self.working_dir = Path(working_dir)
        else:
            self.working_dir = None

        if app_dir is not None:
            self.app_dir_local = Path(app_dir)
        else:
            self.app_dir_local = None

        self.app_type_list = app_type_list

        if self.run_type == 'parSETUP':
            self.app_dir_local = self.app_dir_remote

        # parse the application registry
        self.app_registry, self.default_values = _parse_app_registry(
            registry_path=self.app_registry_file, app_types=self.app_type_list
        )

        # parse the input file
        self.workflow_apps = {}
        self.workflow_assets = {}
        self._parse_inputs()

    def __del__(self):
        # if parallel setup, add command to run this scipt with parellel option
        if self.parType == 'parSETUP':
            inputArgs = sys.argv
            length = len(inputArgs)
            i = 0
            while i < length:
                if 'parSETUP' == inputArgs[i]:
                    inputArgs[i] = 'parRUN'
                i += 1

            inputArgs.insert(0, 'python')
            command = create_command(inputArgs)

            self.parCommandFile.write(
                '\n# Writing Final command to run this application in parallel\n'
            )
            self.parCommandFile.write(
                self.mpiExec + ' -n ' + str(self.numProc) + ' ' + command
            )
            self.parCommandFile.close()

    def _register_app_type(self, app_type, app_dict, sub_app=''):
        """
        Function to register the applications provided in the input file into
        memory, i.e., the 'App registry'

        Parameters
        ----------

        app_type - the type of application

        app_dict - dictionary containing app data

        """

        if type(app_dict) is not dict:
            return
        else:
            for itmKey, itm in app_dict.items():
                self._register_app_type(app_type, itm, itmKey)

        # The provided application
        app_in = app_dict.get('Application')

        # Check to ensure the applications key is provided in the input
        if app_in == None:
            return
            err = "Need to provide the 'Application' key in " + app_type
            raise WorkFlowInputError(err)

        # Check to see if the app type is in the application registry
        app_type_obj = self.app_registry.get(app_type)

        if app_in == None:
            return

        if app_in == 'None':
            return

        if app_type_obj == None:
            err = 'The application ' + app_type + ' is not found in the app registry'
            raise WorkFlowInputError(err)

        # Finally check to see if the app registry contains the provided application
        if app_type_obj.get(app_in) == None:
            err = (
                'Could not find the provided application in the internal app registry, app name: '
                + app_in
            )
            print('Error', app_in)
            raise WorkFlowInputError(err)

        appData = app_dict['ApplicationData']
        #
        #        for itmKey, itm in  appData.items() :
        #            self._register_app_type(app_type,itm,itmKey)

        # Make a deep copy of the app object
        app_object = deepcopy(app_type_obj.get(app_in))

        # Check if the app object was created successfully
        if app_object is None:
            raise WorkFlowInputError(
                'Application deep copy failed for {}'.format(app_type)
            )

        # only assign the app to the workflow if it has an executable
        if app_object.rel_path is None:
            log_msg(
                f'{app_dict["Application"]} is '
                'a passive application (i.e., it does not invoke '
                'any calculation within the workflow.',
                prepend_timestamp=False,
            )

        else:
            app_object.set_pref(appData, self.reference_dir)

            if len(sub_app) == 0:
                log_msg(f'For {app_type}', prepend_timestamp=False)
                self.workflow_apps[app_type] = app_object
            else:
                if self.workflow_apps.get(app_type, None) is None:
                    self.workflow_apps[app_type] = {}

                log_msg(f'For {sub_app} in {app_type}', prepend_timestamp=False)
                self.workflow_apps[app_type][sub_app] = app_object

            log_msg(
                f'  Registering application {app_dict["Application"]} ',
                prepend_timestamp=False,
            )

    def _register_asset(self, asset_type, asset_dict):
        """
        Function to register the assets provided in the input file into memory

        Parameters
        ----------

        asset_type - the type of asset, e.g., buildings, water pipelines

        app_dict - dictionary containing asset data

        """

        # Check to see if the app type is in the application registry
        asset_object = self.asset_registry.get(asset_type)

        if asset_object is None:
            err = (
                'The asset '
                + asset_type
                + ' is not found in the asset registry. Supported assets are '
                + ' '.join(self.asset_type_list)
            )
            raise WorkFlowInputError(err)

        # Add the incoming asset to the workflow assets
        self.workflow_assets[asset_type] = asset_dict

        log_msg(f'Found asset: {asset_type} ', prepend_timestamp=False)

    def _parse_inputs(self):
        """
        Load the information about the workflow to run

        """

        log_msg('Parsing workflow input file')

        # open input file
        log_msg('Loading the json file...', prepend_timestamp=False)
        with open(self.input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        log_msg('  OK', prepend_timestamp=False)

        # store the specified units (if available)
        if 'units' in input_data:
            self.units = input_data['units']

            log_msg('The following units were specified: ', prepend_timestamp=False)
            for key, unit in self.units.items():
                log_msg('  {}: {}'.format(key, unit), prepend_timestamp=False)
        else:
            self.units = None
            log_msg(
                'No units specified; using Standard units.', prepend_timestamp=False
            )

        # store the specified output types
        self.output_types = input_data.get('outputs', None)

        if self.output_types is None:
            default_output_types = {
                'AIM': False,
                'EDP': True,
                'DM': True,
                'DV': True,
                'every_realization': False,
            }

            log_msg(
                'Missing output type specification, using default ' 'settings.',
                prepend_timestamp=False,
            )
            self.output_types = default_output_types

        else:
            log_msg(
                'The following output_types were requested: ',
                prepend_timestamp=False,
            )
            for out_type, flag in self.output_types.items():
                if flag:
                    log_msg(f'  {out_type}', prepend_timestamp=False)

        # replace the default values, if needed
        default_values = input_data.get('DefaultValues', None)

        if default_values is None:
            default_values = {}

        # workflow input is input file
        default_values['workflowInput'] = os.path.basename(self.input_file)
        if default_values is not None:
            log_msg(
                'The following workflow defaults were overwritten:',
                prepend_timestamp=False,
            )

            for key, value in default_values.items():
                if key in self.default_values.keys():
                    self.default_values[key] = value

                else:
                    self.default_values.update({key: value})

                log_msg(f'  {key}: {value}', prepend_timestamp=False)

        # parse the shared data in the input file
        self.shared_data = {}
        for shared_key in [
            'RegionalEvent',
        ]:
            value = input_data.get(shared_key, None)
            if value != None:
                self.shared_data.update({shared_key: value})

        # parse the location of the run_dir
        if self.working_dir is not None:
            self.run_dir = self.working_dir
        elif 'runDir' in input_data:
            self.run_dir = Path(input_data['runDir'])
        # else:
        #    raise WorkFlowInputError('Need a runDir entry in the input file')

        # parse the location(s) of the applications directory
        if 'localAppDir' in input_data:
            self.app_dir_local = input_data['localAppDir']
        # else:
        #    raise WorkFlowInputError('Need a localAppDir entry in the input file')

        if 'remoteAppDir' in input_data:
            self.app_dir_remote = Path(input_data['remoteAppDir'])
        else:
            self.app_dir_remote = self.app_dir_local
            log_msg(
                'remoteAppDir not specified. Using the value provided for '
                'localAppDir instead.',
                prepend_timestamp=False,
            )

        if self.app_dir_local == '' and self.app_dir_remote != '':
            self.app_dir_local = self.app_dir_remote

        if self.app_dir_remote == '' and self.app_dir_local != '':
            self.app_dir_remote = self.app_dir_local

        if 'referenceDir' in input_data:
            self.reference_dir = input_data['referenceDir']

        for loc_name, loc_val in zip(
            [
                'Run dir',
                'Local applications dir',
                'Remote applications dir',
                'Reference dir',
            ],
            [
                self.run_dir,
                self.app_dir_local,
                self.app_dir_remote,
                self.reference_dir,
            ],
        ):
            log_msg('{} : {}'.format(loc_name, loc_val), prepend_timestamp=False)

        # get the list of requested applications
        log_msg(
            '\nParsing the requested list of applications...',
            prepend_timestamp=False,
        )

        if 'Applications' in input_data:
            requested_apps = input_data['Applications']
        else:
            raise WorkFlowInputError('Need an Applications entry in the input file')

        # create the requested applications

        # Events are special because they are in an array
        if 'Events' in requested_apps:
            if len(requested_apps['Events']) > 1:
                raise WorkFlowInputError(
                    'Currently, WHALE only supports a single event.'
                )
            for event in requested_apps['Events'][
                :1
            ]:  # this limitation can be relaxed in the future
                if 'EventClassification' in event:
                    eventClassification = event['EventClassification']
                    if eventClassification in [
                        'Earthquake',
                        'Wind',
                        'Hurricane',
                        'Flood',
                        'Hydro',
                        'Tsunami',
                        'Surge',
                        'Lahar',
                    ]:
                        app_object = deepcopy(
                            self.app_registry['Event'].get(event['Application'])
                        )

                        if app_object is None:
                            raise WorkFlowInputError(
                                'Application entry missing for {}'.format('Events')
                            )

                        app_object.set_pref(
                            event['ApplicationData'], self.reference_dir
                        )
                        self.workflow_apps['Event'] = app_object

                    else:
                        raise WorkFlowInputError(
                            (
                                'Currently, only earthquake and wind events are supported. '
                                'EventClassification must be Earthquake, not {}'
                            ).format(eventClassification)
                        )
                else:
                    raise WorkFlowInputError('Need Event Classification')

        # Figure out what types of assets are coming into the analysis
        assetObjs = requested_apps.get('Assets', None)

        # Check if an asset object exists
        if assetObjs != None:
            # raise WorkFlowInputError('Need to define the assets for analysis')

            # Check if asset list is not empty
            if len(assetObjs) == 0:
                raise WorkFlowInputError('The provided asset object is empty')

            # Iterate through the asset objects
            for assetObj in assetObjs:
                self._register_asset(assetObj, assetObjs[assetObj])

        # Iterate through the app type list which is set when you instantiate the workflow
        for app_type in self.app_type_list:
            # If the app_type is not an event
            if app_type == 'Event':
                continue

            # Check to make sure the required app type is in the list of requested apps
            # i.e., the apps in provided in the input.json file
            if app_type in requested_apps:
                self._register_app_type(app_type, requested_apps[app_type])

        for app_type in self.optional_apps:
            if (app_type not in self.app_registry) and (
                app_type in self.app_type_list
            ):
                self.app_type_list.remove(app_type)

        def recursiveLog(app_type, app_object):
            if type(app_object) is dict:
                for sub_app_type, sub_object in app_object.items():
                    log_msg('   {} : '.format(app_type), prepend_timestamp=False)
                    recursiveLog(sub_app_type, sub_object)
            else:
                log_msg(
                    '       {} : {}'.format(app_type, app_object.name),
                    prepend_timestamp=False,
                )

        log_msg('\nRequested workflow:', prepend_timestamp=False)

        for app_type, app_object in self.workflow_apps.items():
            recursiveLog(app_type, app_object)

        log_msg('\nSuccessfully parsed workflow inputs', prepend_timestamp=False)
        log_div()

    def create_asset_files(self):
        """
        Short description

        Longer description

        Parameters
        ----------

        """

        log_msg('Creating files for individual assets')

        # Open the input file - we'll need it later
        with open(self.input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        # Get the workflow assets
        assetsWfapps = self.workflow_apps.get('Assets', None)
        assetWfList = self.workflow_assets.keys()

        # TODO: not elegant code, fix later
        os.chdir(self.run_dir)

        assetFilesList = {}

        # Iterate through the asset workflow apps
        for asset_type, asset_app in assetsWfapps.items():
            asset_folder = posixpath.join(self.run_dir, asset_type)

            # Make a new directory for each asset
            os.mkdir(asset_folder)

            asset_file = posixpath.join(asset_folder, asset_type) + '.json'

            assetPrefs = asset_app.pref

            # filter assets (if needed)
            asset_filter = asset_app.pref.get('filter', None)
            if asset_filter == '':
                del asset_app.pref['filter']
                asset_filter = None

            if asset_filter is not None:
                atag = [bs.split('-') for bs in asset_filter.split(',')]

                asset_file = Path(
                    str(asset_file).replace(
                        '.json', f'{atag[0][0]}-{atag[-1][-1]}.json'
                    )
                )

            # store the path to the asset file

            assetFilesList[asset_type] = str(asset_file)

            for output in asset_app.outputs:
                if output['id'] == 'assetFile':
                    output['default'] = asset_file

            asset_command_list = asset_app.get_command_list(
                app_path=self.app_dir_local
            )

            # The GEOJSON_TO_ASSET application is special because it can be used
            # for multiple asset types. "asset_type" needs to be added so the app
            # knows which asset_type it's processing.
            if asset_app.name == 'GEOJSON_TO_ASSET' or asset_app.name == 'INP_FILE':
                asset_command_list = asset_command_list + [
                    '--assetType',
                    asset_type,
                    '--inputJsonFile',
                    self.input_file,
                ]

            asset_command_list.append('--getRV')

            # Create the asset command list
            command = create_command(asset_command_list)

            if self.parType == 'parSETUP':
                log_msg(
                    '\nWriting Asset Info command for asset type: '
                    + asset_type
                    + ' to script',
                    prepend_timestamp=False,
                )
                self.parCommandFile.write(
                    '\n# Perform Asset File Creation for type: ' + asset_type + ' \n'
                )

                if asset_app.runsParallel == False:
                    self.parCommandFile.write(command + '\n')
                else:
                    self.parCommandFile.write(
                        self.mpiExec
                        + ' -n '
                        + str(self.numProc)
                        + ' '
                        + command
                        + '\n'
                    )

            else:
                log_msg(
                    '\nCreating initial asset information model (AIM) files for '
                    + asset_type,
                    prepend_timestamp=False,
                )
                log_msg(
                    '\n{}\n'.format(command),
                    prepend_timestamp=False,
                    prepend_blank_space=False,
                )

                result, returncode = run_command(command)

                # Check if the command was completed successfully
                if returncode != 0:
                    print(result)
                    raise WorkFlowInputError(
                        'Failed to create the AIM file for ' + asset_type
                    )
                else:
                    log_msg(
                        'AIM files created for ' + asset_type + '\n',
                        prepend_timestamp=False,
                    )

                log_msg(
                    'Output: ' + str(returncode),
                    prepend_timestamp=False,
                    prepend_blank_space=False,
                )
                log_msg(
                    '\n{}\n'.format(result),
                    prepend_timestamp=False,
                    prepend_blank_space=False,
                )
                log_msg(
                    '\nAsset Information Model (AIM) files successfully created.',
                    prepend_timestamp=False,
                )

        log_div()

        return assetFilesList

    def augment_asset_files(self):
        """
        Short description

        Longer description

        Parameters
        ----------

        """

        log_msg('Augmenting files for individual assets for Workflow')

        # print('INPUT FILE:', self.input_file)

        # Open the input file - we'll need it later
        with open(self.input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        # Get the workflow assets
        assetsWfapps = self.workflow_apps.get('Assets', None)
        assetWfList = self.workflow_assets.keys()

        # TODO: not elegant code, fix later
        os.chdir(self.run_dir)

        assetFilesList = {}

        # Iterate through the asset workflow apps
        for asset_type, asset_app in assetsWfapps.items():
            asset_folder = posixpath.join(self.run_dir, asset_type)

            asset_file = posixpath.join(asset_folder, asset_type) + '.json'

            assetPrefs = asset_app.pref

            # filter assets (if needed)
            asset_filter = asset_app.pref.get('filter', None)
            if asset_filter == '':
                del asset_app.pref['filter']
                asset_filter = None

            if asset_filter is not None:
                atag = [bs.split('-') for bs in asset_filter.split(',')]

                asset_file = Path(
                    str(asset_file).replace(
                        '.json', f'{atag[0][0]}-{atag[-1][-1]}.json'
                    )
                )

            # store the path to the asset file
            assetFilesList[asset_type] = str(asset_file)

            for output in asset_app.outputs:
                if output['id'] == 'assetFile':
                    output['default'] = asset_file

            # Check if the command was completed successfully
            # FMK check AIM file exists

            # Append workflow settings to the BIM file
            log_msg('Appending additional settings to the AIM files...\n')

            with open(asset_file, 'r', encoding='utf-8') as f:
                asset_data = json.load(f)

            # extract the extra information from the input file for this asset type
            extra_input = {'Applications': {}}

            if self.parType == 'parRUN':
                extra_input['parType'] = self.parType
                extra_input['mpiExec'] = self.mpiExec
                extra_input['numProc'] = self.numProc

            apps_of_interest = [
                'Events',
                'Modeling',
                'EDP',
                'Simulation',
                'UQ',
                'DL',
            ]
            for app_type in apps_of_interest:
                # Start with the app data under Applications
                if app_type in input_data['Applications'].keys():
                    if app_type == 'Events':
                        # Events are stored in an array, so they require special treatment
                        app_data_array = input_data['Applications'][app_type]

                        extra_input['Applications'][app_type] = []

                        for app_data in app_data_array:
                            if 'Application' in app_data:
                                app_info = app_data
                            elif asset_type in app_data:
                                app_info = app_data[asset_type]

                            extra_input['Applications'][app_type].append(app_info)

                    else:
                        # Every other app type has a single app in it per asset type
                        app_data = input_data['Applications'][app_type]

                        if 'Application' in app_data:
                            app_info = app_data
                        elif asset_type in app_data:
                            app_info = app_data[asset_type]

                        extra_input['Applications'][app_type] = app_info

                # Then, look at the app data in the root of the input json
                if app_type in input_data.keys():
                    if app_type == 'Events':
                        # Events are stored in an array, so they require special treatment
                        app_data_array = input_data[app_type]

                        extra_input[app_type] = []

                        for app_data in app_data_array:
                            if asset_type in app_data:
                                extra_input[app_type].append(app_data[asset_type])

                    else:
                        # Every other app type has a single app in it per asset type
                        app_data = input_data[app_type]

                        if asset_type in app_data:
                            extra_input[app_type] = app_data[asset_type]

            count = 0
            for asst in asset_data:
                if count % self.numP == self.procID:
                    AIM_file = asst['file']

                    # Open the AIM file and add the unit information to it
                    # print(count, self.numP, self.procID, AIM_file)

                    with open(AIM_file, 'r', encoding='utf-8') as f:
                        AIM_data = json.load(f)

                    if 'DefaultValues' in input_data.keys():
                        AIM_data.update(
                            {'DefaultValues': input_data['DefaultValues']}
                        )

                    if 'commonFileDir' in input_data.keys():
                        commonFileDir = input_data['commonFileDir']
                        if self.inputFilePath not in commonFileDir:
                            commonFileDir = os.path.join(
                                self.inputFilePath, input_data['commonFileDir']
                            )
                        AIM_data.update({'commonFileDir': commonFileDir})

                    if 'remoteAppDir' in input_data.keys():
                        AIM_data.update({'remoteAppDir': input_data['remoteAppDir']})

                    if 'localAppDir' in input_data.keys():
                        AIM_data.update({'localAppDir': input_data['localAppDir']})

                    if self.units != None:
                        AIM_data.update({'units': self.units})

                    # TODO: remove this after all apps have been updated to use the
                    # above location to get units
                    AIM_data['GeneralInformation'].update({'units': self.units})

                    AIM_data.update({'outputs': self.output_types})

                    for key, value in self.shared_data.items():
                        AIM_data[key] = value

                    # Save the asset type
                    AIM_data['assetType'] = asset_type

                    AIM_data.update(extra_input)

                    with open(AIM_file, 'w', encoding='utf-8') as f:
                        json.dump(AIM_data, f, indent=2)

                count = count + 1

        log_msg(
            '\nAsset Information Model (AIM) files successfully augmented.',
            prepend_timestamp=False,
        )
        log_div()

        return assetFilesList

    def perform_system_performance_assessment(self, asset_type):
        """
        For an asset type run the system level performance assesment application

        Longer description

        Parameters
        ----------
        asset_type: string
           Asset type to run perform system assessment of

        """

        if 'SystemPerformance' in self.workflow_apps.keys():
            performance_app = self.workflow_apps['SystemPerformance'][asset_type]
        else:
            log_msg(
                f'No Performance application to run for asset type: {asset_type}.',
                prepend_timestamp=False,
            )
            log_div()
            return False

        if performance_app.rel_path == None:
            log_msg(
                f'No Performance application to run for asset type: {asset_type}.',
                prepend_timestamp=False,
            )
            log_div()
            return False

        log_msg(
            'Performing System Performance Application for asset type: '
            + asset_type,
            prepend_timestamp=False,
        )
        log_div()

        app_command_list = performance_app.get_command_list(
            app_path=self.app_dir_local
        )

        #
        # defaults added to a system performance app are asset_type, input_dir and running_parallel (default False)
        #

        # app_command_list.append('--asset_type')
        # app_command_list.append(asset_type)
        app_command_list.append('--input')
        app_command_list.append(self.input_file)
        # app_command_list.append('--working_dir')
        # app_command_list.append(self.working_dir)

        # Sina added this part for parallel run in REWET
        if self.parType == 'parSETUP':
            log_msg(
                '\nParallel settings for System Performance for asset type:'
                + asset_type,
                prepend_timestamp=False,
            )
            app_command_list.append('--par')

        command = create_command(app_command_list)

        log_msg('Output: ', prepend_timestamp=False, prepend_blank_space=False)
        log_msg(
            '\n{}\n'.format(command),
            prepend_timestamp=False,
            prepend_blank_space=False,
        )

        result, returncode = run_command(command)
        log_msg(
            '\n{}\n'.format(result),
            prepend_timestamp=False,
            prepend_blank_space=False,
        )

        log_msg(
            'System Performance Application Completed for asset type: ' + asset_type,
            prepend_timestamp=False,
        )

        #  end of Sina's odifications for parallel run

        # if (self.parType == 'parSETUP'):

        #     log_msg('\nWriting System Performance application for asset type:' + asset_type, prepend_timestamp=False)
        #     self.parCommandFile.write("\n# Writing System Performance application for asset type:" + asset_type +"\n")

        #     if performance_app.runsParallel == False:
        #         self.parCommandFile.write(command + "\n")
        #     else:
        #         self.parCommandFile.write(self.mpiExec + " -n " + str(self.numProc) + " " + command + " --running_parallel True\n")

        # else:

        #     log_msg('\n{}\n'.format(command), prepend_timestamp=False,
        #             prepend_blank_space=False)

        #     result, returncode = run_command(command)

        #     log_msg('Output: ', prepend_timestamp=False, prepend_blank_space=False)
        #     log_msg('\n{}\n'.format(result), prepend_timestamp=False, prepend_blank_space=False)

        #     log_msg('System Performance Application Completed for asset type: ' + asset_type, prepend_timestamp=False)

        log_div()
        return True

    def perform_regional_event(self):
        """
        Run an application to simulate a regional-scale hazard event.

        Longer description

        Parameters
        ----------

        """

        log_msg('Simulating regional event...')

        if 'RegionalEvent' in self.workflow_apps.keys():
            reg_event_app = self.workflow_apps['RegionalEvent']
        else:
            log_msg('No Regional Event Application to run.', prepend_timestamp=False)
            log_div()
            return

        if reg_event_app.rel_path == None:
            log_msg('No regional Event Application to run.', prepend_timestamp=False)
            log_div()
            return

        reg_event_command_list = reg_event_app.get_command_list(
            app_path=self.app_dir_local
        )

        command = create_command(reg_event_command_list)

        if self.parType == 'parSETUP':
            log_msg(
                '\nWriting Regional Event Command to script', prepend_timestamp=False
            )
            self.parCommandFile.write('\n# Perform Regional Event Simulation\n')

            if reg_event_app.runsParallel == False:
                self.parCommandFile.write(command + '\n')
            else:
                self.parCommandFile.write(
                    self.mpiExec + ' -n ' + str(self.numProc) + ' ' + command + '\n'
                )

        else:
            log_msg(
                '\n{}\n'.format(command),
                prepend_timestamp=False,
                prepend_blank_space=False,
            )

            result, returncode = run_command(command)

            log_msg('Output: ', prepend_timestamp=False, prepend_blank_space=False)
            log_msg(
                '\n{}\n'.format(result),
                prepend_timestamp=False,
                prepend_blank_space=False,
            )

            log_msg(
                'Regional event successfully simulated.', prepend_timestamp=False
            )
        log_div()

    def perform_regional_recovery(self, asset_keys):
        """
        Run an application to simulate regional recovery

        Longer description

        Parameters
        ----------

        """

        log_msg('Simulating Regional Recovery ...')

        if 'Recovery' in self.workflow_apps.keys():
            reg_recovery_app = self.workflow_apps['Recovery']
        else:
            log_msg('No Recovery Application to run.', prepend_timestamp=False)
            log_div()
            return

        if reg_recovery_app.rel_path == None:
            log_msg('No regional Event Application to run.', prepend_timestamp=False)
            log_div()
            return

        reg_recovery_command_list = reg_recovery_app.get_command_list(
            app_path=self.app_dir_local
        )

        command = create_command(reg_recovery_command_list)

        if self.parType == 'parSETUP':
            log_msg(
                '\nWriting Regional Event Command to script', prepend_timestamp=False
            )
            self.parCommandFile.write('\n# Perform Regional Recovery Simulation\n')

            if reg_recovery_app.runsParallel == False:
                self.parCommandFile.write(command + '\n')
            else:
                self.parCommandFile.write(
                    self.mpiExec + ' -n ' + str(self.numProc) + ' ' + command + '\n'
                )

        else:
            log_msg(
                '\n{}\n'.format(command),
                prepend_timestamp=False,
                prepend_blank_space=False,
            )

            result, returncode = run_command(command)

            log_msg('Output: ', prepend_timestamp=False, prepend_blank_space=False)
            log_msg(
                '\n{}\n'.format(result),
                prepend_timestamp=False,
                prepend_blank_space=False,
            )

            log_msg(
                'Regional Recovery Successfully Simulated.', prepend_timestamp=False
            )
        log_div()

    def perform_regional_mapping(self, AIM_file_path, assetType, doParallel=True):
        """
        Performs the regional mapping between the asset and a hazard event.


        Parameters
        ----------

        """

        log_msg('', prepend_timestamp=False, prepend_blank_space=False)
        log_msg('Creating regional mapping...')

        reg_mapping_app = self.workflow_apps['RegionalMapping'][assetType]

        # TODO: not elegant code, fix later
        for input_ in reg_mapping_app.inputs:
            if input_['id'] == 'assetFile':
                input_['default'] = str(AIM_file_path)

        reg_mapping_app.inputs.append(
            {
                'id': 'filenameEVENTgrid',
                'type': 'path',
                'default': resolve_path(
                    self.shared_data['RegionalEvent']['eventFile'],
                    self.reference_dir,
                ),
            }
        )

        reg_mapping_command_list = reg_mapping_app.get_command_list(
            app_path=self.app_dir_local
        )

        command = create_command(reg_mapping_command_list)

        log_msg(
            '\n{}\n'.format(command),
            prepend_timestamp=False,
            prepend_blank_space=False,
        )

        if self.parType == 'parSETUP':
            self.parCommandFile.write(
                '\n# Regional Mapping for asset type: ' + assetType + ' \n'
            )

            if reg_mapping_app.runsParallel == False:
                self.parCommandFile.write(command + '\n')
            else:
                self.parCommandFile.write(
                    self.mpiExec
                    + ' -n '
                    + str(self.numProc)
                    + ' '
                    + command
                    + ' --doParallel '
                    + str(doParallel)
                    + ' -m '
                    + self.mpiExec
                    + ' --numP '
                    + str(self.numProc)
                    + '\n'
                )

            log_msg(
                'Regional mapping command added to parallel script.',
                prepend_timestamp=False,
            )

        else:
            result, returncode = run_command(command)

            log_msg(
                'Output: ' + str(returncode),
                prepend_timestamp=False,
                prepend_blank_space=False,
            )
            log_msg(
                '\n{}\n'.format(result),
                prepend_timestamp=False,
                prepend_blank_space=False,
            )

            log_msg(
                'Regional mapping successfully created.', prepend_timestamp=False
            )

        log_div()

    def init_simdir(self, asst_id=None, AIM_file_path='AIM.json'):
        """
        Initializes the simulation directory for each asset.

        In the current directory where the Asset Information Model (AIM) file resides, e.g., ./Buildings/2000-AIM.json, a new directory is created with the asset id, e.g., ./Buildings/2000, and within that directory a template directory is created (templatedir) ./Buildings/2000/templatedir. The AIM file is copied over to the template dir. It is within this template dir that the analysis is run for the individual asset.

        Parameters
        ----------

        asst_id - the asset id
        AIM_file  - file path to the existing AIM file
        """
        log_msg('Initializing the simulation directory\n')

        aimDir = os.path.dirname(AIM_file_path)
        aimFileName = os.path.basename(AIM_file_path)

        # If the path is not provided, assume the AIM file is in the run dir
        if os.path.exists(aimDir) == False:
            aimDir = self.run_dir
            aimFileName = AIM_file_path

        os.chdir(aimDir)

        if asst_id is not None:
            # if the directory already exists, remove its contents
            if asst_id in os.listdir(aimDir):
                shutil.rmtree(asst_id, ignore_errors=True)

            # create the asset_id dir and the template dir
            os.mkdir(asst_id)
            os.chdir(asst_id)
            os.mkdir('templatedir')
            os.chdir('templatedir')

            # Make a copy of the AIM file
            src = posixpath.join(aimDir, aimFileName)
            dst = posixpath.join(aimDir, f'{asst_id}/templatedir/{aimFileName}')
            # dst = posixpath.join(aimDir, f'{asst_id}/templatedir/AIM.json')

            try:
                shutil.copy(src, dst)

                print('Copied AIM file to: ', dst)
                # os.remove(src)

            except:
                print('Error occurred while copying file: ', dst)

        else:
            for dir_or_file in os.listdir(os.getcwd()):
                if dir_or_file not in ['log.txt', 'templatedir', 'input_data']:
                    if os.path.isdir(dir_or_file):
                        shutil.rmtree(dir_or_file)
                    else:
                        os.remove(dir_or_file)

            os.chdir(
                'templatedir'
            )  # TODO: we might want to add a generic id dir to be consistent with the regional workflow here

            # Remove files with .j extensions that might be there from previous runs
            for file in os.listdir(os.getcwd()):
                if file.endswith('.j'):
                    os.remove(file)

            # Make a copy of the input file and rename it to AIM.json
            # This is a temporary fix, will be removed eventually.
            dst = Path(os.getcwd()) / AIM_file_path
            # dst = posixpath.join(os.getcwd(),AIM_file)
            if AIM_file_path != self.input_file:
                shutil.copy(src=self.input_file, dst=dst)

        log_msg(
            'Simulation directory successfully initialized.\n',
            prepend_timestamp=False,
        )
        log_div()
        return dst

    def cleanup_simdir(self, asst_id):
        """
        Short description

        Longer description

        Parameters
        ----------

        """
        log_msg('Cleaning up the simulation directory.')

        os.chdir(self.run_dir)

        if asst_id is not None:
            os.chdir(asst_id)

        workdirs = os.listdir(os.getcwd())
        for workdir in workdirs:
            if 'workdir' in workdir:
                shutil.rmtree(workdir, ignore_errors=True)

        log_msg(
            'Simulation directory successfully cleaned up.', prepend_timestamp=False
        )
        log_div()

    def init_workdir(self):
        """
        Short description

        Longer description

        Parameters
        ----------

        """
        log_msg('Initializing the working directory.')

        os.chdir(self.run_dir)

        for dir_or_file in os.listdir(os.getcwd()):
            if dir_or_file != 'log.txt':
                if os.path.isdir(dir_or_file):
                    shutil.rmtree(dir_or_file)
                else:
                    os.remove(dir_or_file)

        log_msg(
            'Working directory successfully initialized.', prepend_timestamp=False
        )
        log_div()

    def cleanup_workdir(self):
        """
        Short description

        Longer description

        Parameters
        ----------

        """
        log_msg('Cleaning up the working directory.')

        os.chdir(self.run_dir)

        workdir_contents = os.listdir(self.run_dir)
        for file_or_dir in workdir_contents:
            if (self.run_dir / file_or_dir).is_dir():
                # if os.path.isdir(posixpath.join(self.run_dir, file_or_dir)):
                shutil.rmtree(file_or_dir, ignore_errors=True)

        log_msg('Working directory successfully cleaned up.')
        log_div()

    def preprocess_inputs(
        self, app_sequence, AIM_file_path='AIM.json', asst_id=None, asset_type=None
    ):
        """
        Short description

        Longer description

        Parameters
        ----------

        """

        log_msg('Running preprocessing step random variables')

        # Get the directory to the asset class dir, e.g., buildings
        aimDir = os.path.dirname(AIM_file_path)
        aimFileName = os.path.basename(AIM_file_path)

        # If the path is not provided, assume the AIM file is in the run dir
        if os.path.exists(aimDir) == False:
            aimDir = self.run_dir

        os.chdir(aimDir)

        if asst_id is not None:
            os.chdir(asst_id)

        # Change the directory to the templatedir that was previously created in init_simdir
        os.chdir('templatedir')

        for app_type in self.optional_apps:
            if (app_type in app_sequence) and (
                app_type not in self.workflow_apps.keys()
            ):
                app_sequence.remove(app_type)

        for app_type in app_sequence:
            workflow_app = self.workflow_apps[app_type]

            if app_type != 'FEM':
                if AIM_file_path is not None:
                    if type(workflow_app) is dict:
                        for itemKey, item in workflow_app.items():
                            if asset_type is not None and asset_type != itemKey:
                                continue

                            item.defaults['filenameAIM'] = AIM_file_path

                            command_list = item.get_command_list(
                                app_path=self.app_dir_local
                            )

                            command_list.append('--getRV')

                            command = create_command(command_list)

                            log_msg(
                                '\nRunning {} app at preprocessing step...'.format(
                                    app_type
                                ),
                                prepend_timestamp=False,
                            )
                            log_msg(
                                '\n{}\n'.format(command),
                                prepend_timestamp=False,
                                prepend_blank_space=False,
                            )

                            result, returncode = run_command(command)

                            log_msg(
                                'Output: ' + str(returncode),
                                prepend_timestamp=False,
                                prepend_blank_space=False,
                            )
                            log_msg(
                                '\n{}\n'.format(result),
                                prepend_timestamp=False,
                                prepend_blank_space=False,
                            )

                            # if returncode==0:
                            #    log_msg('Preprocessing successfully completed.', prepend_timestamp=False)
                            # else:
                            #    log_msg('Error in the preprocessor.', prepend_timestamp=False)
                            #    exit(-1)

                            log_div()

                    else:
                        workflow_app.defaults['filenameAIM'] = AIM_file_path

                        command_list = workflow_app.get_command_list(
                            app_path=self.app_dir_local
                        )

                        command_list.append('--getRV')

                        command = create_command(command_list)

                        log_msg(
                            '\nRunning {} app at preprocessing step...'.format(
                                app_type
                            ),
                            prepend_timestamp=False,
                        )
                        log_msg(
                            '\n{}\n'.format(command),
                            prepend_timestamp=False,
                            prepend_blank_space=False,
                        )

                        result, returncode = run_command(command)

                        log_msg(
                            'Output: ' + str(returncode),
                            prepend_timestamp=False,
                            prepend_blank_space=False,
                        )
                        log_msg(
                            '\n{}\n'.format(result),
                            prepend_timestamp=False,
                            prepend_blank_space=False,
                        )

                        # if returncode==0:
                        #    log_msg('Preprocessing successfully completed.', prepend_timestamp=False)
                        # else:
                        #    log_msg('Error in the preprocessor.', prepend_timestamp=False)
                        #    exit(-1)

                        log_div()

            else:
                old_command_list = workflow_app.get_command_list(
                    app_path=self.app_dir_local
                )
                old_command_list.append('--appKey')
                old_command_list.append('FEM')
                if old_command_list[0] == 'python':
                    if self.run_type in ['set_up', 'runningRemote']:
                        old_command_list.append('--runType')
                        old_command_list.append('runningRemote')
                        old_command_list.append('--osType')
                        old_command_list.append('MacOS')
                    else:
                        old_command_list.append('--runType')
                        old_command_list.append('runningLocal')
                        if any(platform.win32_ver()):
                            old_command_list.append('--osType')
                            old_command_list.append('Windows')
                        else:
                            old_command_list.append('--osType')
                            old_command_list.append('MacOS')

                    command = create_command(old_command_list)

                else:
                    #
                    # FMK to modify C++ applications to take above
                    #

                    command_list = []
                    command_list.append(old_command_list[0])
                    command_list.append(self.input_file)

                    if self.run_type in ['set_up', 'runningRemote']:
                        command_list.append('runningRemote')
                        command_list.append('MacOS')
                    else:
                        command_list.append('runningLocal')
                        if any(platform.win32_ver()):
                            command_list.append('Windows')
                        else:
                            command_list.append('MacOS')

                    command_list.append(old_command_list[4])

                    command = create_command(command_list)

                log_msg('\nRunning FEM app', prepend_timestamp=False)
                log_msg(
                    '\n{}\n'.format(command),
                    prepend_timestamp=False,
                    prepend_blank_space=False,
                )

                result, returncode = run_command(command)

                log_msg(
                    'Output: ' + str(returncode),
                    prepend_timestamp=False,
                    prepend_blank_space=False,
                )
                log_msg(
                    '\n{}\n'.format(result),
                    prepend_timestamp=False,
                    prepend_blank_space=False,
                )

                # sy - trying adding exit command
                # if platform.system() == 'Windows':
                #    with open("driver.bat","r", encoding="utf-8") as f:
                #        lines = f.readlines()
                #    #lines.append(r'if %errorlevel% neq 0 exit /b -1')
                #    with open("driver.bat","w", encoding="utf-8") as f:
                #        f.writelines(lines)
                # else:
                #    pass

                log_msg(
                    'Successfully Created Driver File for Workflow.',
                    prepend_timestamp=False,
                )
                log_div()

    def gather_workflow_inputs(self, asst_id=None, AIM_file_path='AIM.json'):
        log_msg('Gathering Workflow Inputs.', prepend_timestamp=False)

        if 'UQ' in self.workflow_apps.keys():
            # Get the directory to the asset class dir, e.g., buildings
            aimDir = os.path.dirname(AIM_file_path)

            # If the path is not provided, assume the AIM file is in the run dir
            if os.path.exists(aimDir) == False:
                aimDir = self.run_dir

            os.chdir(aimDir)

            if asst_id is not None:
                os.chdir(asst_id)

            os.chdir('templatedir')

            relPathCreateCommon = (
                'applications/performUQ/common/createStandardUQ_Input'
            )
            abs_path = Path(self.app_dir_local) / relPathCreateCommon

            arg_list = []
            arg_list.append('{}'.format(abs_path.as_posix()))
            # arg_list.append(u'{}'.format(abs_path))

            # inputFilePath = os.path.dirname(self.input_file)
            inputFilePath = os.getcwd()
            inputFilename = os.path.basename(self.input_file)
            pathToScFile = posixpath.join(inputFilePath, 'sc_' + inputFilename)

            # arg_list.append(u'{}'.format(self.input_file))
            arg_list.append('{}'.format(AIM_file_path))
            arg_list.append('{}'.format(pathToScFile))
            arg_list.append('{}'.format(self.default_values['driverFile']))
            arg_list.append('{}'.format('sc_' + self.default_values['driverFile']))
            arg_list.append('{}'.format(self.run_type))

            if any(platform.win32_ver()):
                arg_list.append('Windows')
            else:
                arg_list.append('MacOS')

            self.default_values['workflowInput'] = pathToScFile
            # self.default_values['driverFile']='sc_'+self.default_values['driverFile']
            self.default_values['modDriverFile'] = (
                'sc_' + self.default_values['driverFile']
            )
            # self.default_values['driverFile']='driver'

            self.modifiedRun = True  # ADAM to fix
            command = create_command(arg_list)

            # print('FMK- gather command:', command)

            result, returncode = run_command(command)

            log_msg('Output: ', prepend_timestamp=False, prepend_blank_space=False)
            log_msg(
                '\n{}\n'.format(result),
                prepend_timestamp=False,
                prepend_blank_space=False,
            )

            log_msg('Successfully Gathered Inputs.', prepend_timestamp=False)
            log_div()

    def create_driver_file(
        self, app_sequence, asst_id=None, AIM_file_path='AIM.json'
    ):
        """
        This functipon creates a UQ driver file. This is only done if UQ is in the workflow apps

        Parameters
        ----------
        """

        if 'UQ' in self.workflow_apps.keys():
            log_msg('Creating the workflow driver file')
            # print('ASSET_ID', asst_id)
            # print('AIM_FILE_PATH', AIM_file_path)

            aimDir = os.path.dirname(AIM_file_path)
            aimFile = os.path.basename(AIM_file_path)

            # If the path is not provided, assume the AIM file is in the run dir
            if os.path.exists(aimDir) == False:
                aimDir = self.run_dir

            os.chdir(aimDir)

            if asst_id is not None:
                os.chdir(asst_id)

            os.chdir('templatedir')

            # print('PWD', os.getcwd())

            driver_script = ''

            for app_type in self.optional_apps:
                if (app_type in app_sequence) and (
                    app_type not in self.workflow_apps.keys()
                ):
                    app_sequence.remove(app_type)

            for app_type in app_sequence:
                workflow_app = self.workflow_apps[app_type]

                # print('FMK runtype', self.run_type)
                if self.run_type in ['set_up', 'runningRemote', 'parSETUP']:
                    if type(workflow_app) is dict:
                        for itemKey, item in workflow_app.items():
                            command_list = item.get_command_list(
                                app_path=self.app_dir_remote, force_posix=True
                            )
                            driver_script += (
                                create_command(
                                    command_list, enforced_python='python3'
                                )
                                + '\n'
                            )

                    else:
                        command_list = workflow_app.get_command_list(
                            app_path=self.app_dir_remote, force_posix=True
                        )
                        driver_script += (
                            create_command(command_list, enforced_python='python3')
                            + '\n'
                        )

                else:
                    if type(workflow_app) is dict:
                        for itemKey, item in workflow_app.items():
                            command_list = item.get_command_list(
                                app_path=self.app_dir_local
                            )
                            driver_script += create_command(command_list) + '\n'

                    else:
                        command_list = workflow_app.get_command_list(
                            app_path=self.app_dir_local
                        )

                        driver_script += create_command(command_list) + '\n'

                # sy - trying adding exit command

                # if platform.system() == 'Windows':
                #   #driver_script += 'if %errorlevel% neq 0 exit /b -1 \n'
                #   pass
                # else:
                #   pass

            # log_msg('Workflow driver script:', prepend_timestamp=False)
            # log_msg('\n{}\n'.format(driver_script), prepend_timestamp=False, prepend_blank_space=False)

            driverFile = self.default_values['driverFile']

            # KZ: for windows, to write bat
            if platform.system() == 'Windows':
                driverFile = driverFile + '.bat'
            log_msg(driverFile)
            with open(driverFile, 'w', newline='\n', encoding='utf-8') as f:
                f.write(driver_script)

            log_msg(
                'Workflow driver file successfully created.', prepend_timestamp=False
            )
            log_div()
        else:
            log_msg('No UQ requested, workflow driver is not needed.')
            log_div()

    def simulate_response(self, AIM_file_path='AIM.json', asst_id=None):
        """
        Short description

        Longer description

        Parameters
        ----------
        """

        # Get the directory to the asset class dir, e.g., buildings
        aimDir = os.path.dirname(AIM_file_path)
        aimFileName = os.path.basename(AIM_file_path)

        # If the path is not provided, assume the AIM file is in the run dir
        if os.path.exists(aimDir) == False:
            aimDir = self.run_dir

        os.chdir(aimDir)

        if asst_id is not None:
            os.chdir(asst_id)

        if 'UQ' in self.workflow_apps.keys():
            log_msg('Running response simulation')

            os.chdir('templatedir')

            workflow_app = self.workflow_apps['UQ']

            # FMK
            if asst_id is not None:
                workflow_app = workflow_app['Buildings']

            if AIM_file_path is not None:
                workflow_app.defaults['filenameAIM'] = AIM_file_path
                # for input_var in workflow_app.inputs:
                #    if input_var['id'] == 'filenameAIM':
                #        input_var['default'] = AIM_file_path

            command_list = workflow_app.get_command_list(app_path=self.app_dir_local)

            # ADAM to fix FMK
            if self.modifiedRun:
                command_list[3] = self.default_values['workflowInput']

                command_list[5] = self.default_values['modDriverFile']

            # add the run type to the uq command list
            command_list.append('--runType')
            command_list.append('{}'.format(self.run_type))

            # if ('rvFiles' in self.default_values.keys()):
            #    command_list.append('--filesWithRV')
            #    rvFiles = self.default_values['rvFiles']
            #    for rvFile in rvFiles:
            #        command_list.append(rvFile)

            # if ('edpFiles' in self.default_values.keys()):
            #    command_list.append('--filesWithEDP')
            #    edpFiles = self.default_values['edpFiles']
            #    for edpFile in edpFiles:
            #        command_list.append(edpFile)

            command = create_command(command_list)

            log_msg('Simulation command:', prepend_timestamp=False)
            log_msg(
                '\n{}\n'.format(command),
                prepend_timestamp=False,
                prepend_blank_space=False,
            )

            result, returncode = run_command(command)

            if self.run_type in ['run', 'runningLocal']:
                log_msg(
                    'Output: ', prepend_timestamp=False, prepend_blank_space=False
                )
                log_msg(
                    '\n{}\n'.format(result),
                    prepend_timestamp=False,
                    prepend_blank_space=False,
                )

                # create the response.csv file from the dakotaTab.out file
                os.chdir(aimDir)

                if asst_id is not None:
                    os.chdir(asst_id)

                try:
                    # sy, abs - added try-statement because dakota-reliability does not write DakotaTab.out
                    dakota_out = pd.read_csv(
                        'dakotaTab.out', sep=r'\s+', header=0, index_col=0
                    )

                    # if the DL is coupled with response estimation, we need to sort the results
                    DL_app = self.workflow_apps.get('DL', None)

                    # FMK
                    # if asst_id is not None:
                    # KZ: 10/19/2022, minor patch
                    if asst_id is not None and DL_app is not None:
                        DL_app = DL_app['Buildings']

                    if DL_app is not None:
                        is_coupled = DL_app.pref.get('coupled_EDP', None)

                        if is_coupled:
                            if 'eventID' in dakota_out.columns:
                                events = dakota_out['eventID'].values
                                events = [int(e.split('x')[-1]) for e in events]
                                sorter = np.argsort(events)
                                dakota_out = dakota_out.iloc[sorter, :]
                                dakota_out.index = np.arange(dakota_out.shape[0])

                    dakota_out.to_csv('response.csv')

                    # log_msg('Response simulation finished successfully.', prepend_timestamp=False)# sy - this message was showing up when quoFEM analysis failed

                except:
                    log_msg(
                        'dakotaTab.out not found. Response.csv not created.',
                        prepend_timestamp=False,
                    )

            elif self.run_type in ['set_up', 'runningRemote']:
                log_msg(
                    'Response simulation set up successfully',
                    prepend_timestamp=False,
                )

            log_div()

        else:
            log_msg('No UQ requested, response simulation step is skipped.')

            # copy the response.csv from the templatedir to the run dir
            shutil.copy(src='templatedir/response.csv', dst='response.csv')

            log_div()

    def perform_asset_performance(asset_type):
        performanceWfapps = self.workflow_apps.get('Performance', None)
        performance_app = performanceWfapps[asset_type]
        app_command_list = performance_app.get_command_list(
            app_path=self.app_dir_local
        )
        command = create_command(app_command_list)
        result, returncode = run_command(command)

    def estimate_losses(
        self,
        AIM_file_path='AIM.json',
        asst_id=None,
        asset_type=None,
        input_file=None,
        copy_resources=False,
    ):
        """
        Short description

        Longer description

        Parameters
        ----------
        """

        if 'DL' in self.workflow_apps.keys():
            log_msg('Running damage and loss assessment')

            # Get the directory to the asset class dir, e.g., buildings
            aimDir = os.path.dirname(AIM_file_path)
            aimFileName = os.path.basename(AIM_file_path)

            # If the path is not provided, assume the AIM file is in the run dir
            if os.path.exists(aimDir) == False:
                aimDir = self.run_dir
                aimFileName = AIM_file_path

            os.chdir(aimDir)

            if 'Assets' not in self.app_type_list:
                # Copy the dakota.json file from the templatedir to the run_dir so that
                # all the required inputs are in one place.
                input_file = PurePath(input_file).name
                # input_file = ntpath.basename(input_file)
                shutil.copy(
                    src=aimDir / f'templatedir/{input_file}',
                    dst=posixpath.join(aimDir, aimFileName),
                )
                # src = posixpath.join(self.run_dir,'templatedir/{}'.format(input_file)),
                # dst = posixpath.join(self.run_dir,AIM_file_path))
            else:
                src = posixpath.join(aimDir, aimFileName)
                dst = posixpath.join(aimDir, f'{asst_id}/{aimFileName}')

                # copy the AIM file from the main dir to the building dir
                shutil.copy(src, dst)

                # src = posixpath.join(self.run_dir, AIM_file_path),
                # dst = posixpath.join(self.run_dir,
                #                     '{}/{}'.format(asst_id, AIM_file_path)))
                os.chdir(str(asst_id))

            workflow_app = self.workflow_apps['DL']

            if type(workflow_app) is dict:
                for itemKey, item in workflow_app.items():
                    if AIM_file_path is not None:
                        item.defaults['filenameDL'] = AIM_file_path
                        # for input_var in workflow_app.inputs:
                        #    if input_var['id'] == 'filenameDL':
                        #        input_var['default'] = AIM_file_path

                    if asset_type != itemKey:
                        continue

                    command_list = item.get_command_list(app_path=self.app_dir_local)

                    if copy_resources:
                        command_list.append('--resource_dir')
                        command_list.append(self.working_dir)

                    command_list.append('--dirnameOutput')
                    # Only add asset id if we are running a regional assessment
                    if asst_id != None:
                        command_list.append(f'{aimDir}/{asst_id}')
                    else:
                        command_list.append(f'{aimDir}')

                    command = create_command(command_list)

                    log_msg(
                        'Damage and loss assessment command (1):',
                        prepend_timestamp=False,
                    )
                    log_msg(
                        '\n{}\n'.format(command),
                        prepend_timestamp=False,
                        prepend_blank_space=False,
                    )

                    result, returncode = run_command(command)

                    log_msg(result, prepend_timestamp=False)

                    # if multiple buildings are analyzed, copy the pelicun_log file to the root dir
                    if 'Assets' in self.app_type_list:
                        try:
                            shutil.copy(
                                src=aimDir / f'{asst_id}/{"pelicun_log.txt"}',
                                dst=aimDir / f'pelicun_log_{asst_id}.txt',
                            )

                            # src = posixpath.join(self.run_dir, '{}/{}'.format(asst_id, 'pelicun_log.txt')),
                            # dst = posixpath.join(self.run_dir, 'pelicun_log_{}.txt'.format(asst_id)))
                        except:
                            pass

            else:
                if AIM_file_path is not None:
                    workflow_app.defaults['filenameDL'] = AIM_file_path
                    # for input_var in workflow_app.inputs:
                    #    if input_var['id'] == 'filenameDL':
                    #        input_var['default'] = AIM_file_path

                command_list = self.workflow_apps['DL'].get_command_list(
                    app_path=self.app_dir_local
                )

                command_list.append('--dirnameOutput')
                # Only add asset id if we are running a regional assessment
                if asst_id != None:
                    command_list.append(f'{aimDir}/{asst_id}')
                else:
                    command_list.append(f'{aimDir}')

                if copy_resources:
                    command_list.append('--resource_dir')
                    command_list.append(self.working_dir)

                command = create_command(command_list)

                log_msg(
                    'Damage and loss assessment command (2):',
                    prepend_timestamp=False,
                )
                log_msg(
                    '\n{}\n'.format(command),
                    prepend_timestamp=False,
                    prepend_blank_space=False,
                )

                result, returncode = run_command(command)

                log_msg(result, prepend_timestamp=False)

                # if multiple buildings are analyzed, copy the pelicun_log file to the root dir
                if 'Building' in self.app_type_list:
                    try:
                        shutil.copy(
                            src=self.run_dir / f'{asst_id}/{"pelicun_log.txt"}',
                            dst=self.run_dir / f'pelicun_log_{asst_id}.txt',
                        )
                        # src = posixpath.join(self.run_dir, '{}/{}'.format(asst_id, 'pelicun_log.txt')),
                        # dst = posixpath.join(self.run_dir, 'pelicun_log_{}.txt'.format(asst_id)))
                    except:
                        pass
            # Remove the copied AIM since it is not used anymore
            try:
                dst = posixpath.join(aimDir, f'{asst_id}/{aimFileName}')
                os.remove(dst)
            except:
                pass
            log_msg(
                'Damage and loss assessment finished successfully.',
                prepend_timestamp=False,
            )
            log_div()

        else:
            log_msg('No DL requested, loss assessment step is skipped.')

            # Only regional simulations send in a asst id
            if asst_id != None:
                EDP_df = pd.read_csv('response.csv', header=0, index_col=0)

                col_info = []
                for col in EDP_df.columns:
                    try:
                        # KZ: 10/19/2022, patches for masking dummy edps (TODO: this part could be optimized)
                        if col in ['dummy']:
                            col_info.append(['dummy', '1', '1'])
                            continue
                        split_col = col.split('-')
                        if len(split_col[1]) == 3:
                            col_info.append(split_col[1:])
                    except:
                        continue

                col_info = np.transpose(col_info)

                EDP_types = np.unique(col_info[0])
                EDP_locs = np.unique(col_info[1])
                EDP_dirs = np.unique(col_info[2])

                MI = pd.MultiIndex.from_product(
                    [EDP_types, EDP_locs, EDP_dirs, ['median', 'beta']],
                    names=['type', 'loc', 'dir', 'stat'],
                )

                df_res = pd.DataFrame(
                    columns=MI,
                    index=[
                        0,
                    ],
                )
                if ('PID', '0') in df_res.columns:
                    del df_res[('PID', '0')]

                # store the EDP statistics in the output DF
                for col in np.transpose(col_info):
                    # KZ: 10/19/2022, patches for masking dummy edps (TODO: this part could be optimized)
                    if 'dummy' in col:
                        df_res.loc[0, (col[0], col[1], col[2], 'median')] = EDP_df[
                            'dummy'
                        ].median()
                        df_res.loc[0, (col[0], col[1], col[2], 'beta')] = np.log(
                            EDP_df['dummy']
                        ).std()
                        continue
                    df_res.loc[0, (col[0], col[1], col[2], 'median')] = EDP_df[
                        '1-{}-{}-{}'.format(col[0], col[1], col[2])
                    ].median()
                    df_res.loc[0, (col[0], col[1], col[2], 'beta')] = np.log(
                        EDP_df['1-{}-{}-{}'.format(col[0], col[1], col[2])]
                    ).std()

                df_res.dropna(axis=1, how='all', inplace=True)

                df_res = df_res.astype(float)

                # save the output
                df_res.to_csv('EDP.csv')

            log_div()

    def estimate_performance(
        self,
        AIM_file_path='AIM.json',
        asst_id=None,
        asset_type=None,
        input_file=None,
        copy_resources=False,
    ):
        if 'Performance' not in self.workflow_apps.keys():
            log_msg(
                'No performance assessment requested, performance assessment step is skipped.'
            )
            log_div()
            return

        log_msg('Running performance assessment')

        # Get the directory to the asset class dir, e.g., buildings
        aimDir = os.path.dirname(AIM_file_path)
        aimFileName = os.path.basename(AIM_file_path)

        # If the path is not provided, assume the AIM file is in the run dir
        if os.path.exists(aimDir) == False:
            aimDir = self.run_dir
            aimFileName = AIM_file_path

        os.chdir(aimDir)

        workflow_app = self.workflow_apps['Performance']

        command_list = workflow_app.get_command_list(app_path=self.app_dir_local)

        command_list.append('--dirnameOutput')

        # Only add asset id if we are running a regional assessment
        if asst_id != None:
            command_list.append(f'{aimDir}/{asst_id}')
        else:
            command_list.append(f'{aimDir}')

        command = create_command(command_list)

        log_msg('Performance assessment command:', prepend_timestamp=False)
        log_msg(
            '\n{}\n'.format(command),
            prepend_timestamp=False,
            prepend_blank_space=False,
        )

        result, returncode = run_command(command)

        log_msg(result, prepend_timestamp=False)

        log_msg('Performance assessment finished.', prepend_timestamp=False)
        log_div()

    def aggregate_results(
        self,
        asst_data,
        asset_type='',
        # out_types = ['IM', 'BIM', 'EDP', 'DM', 'DV', 'every_realization'],
        out_types=['AIM', 'EDP', 'DMG', 'DV', 'every_realization'],
        headers=None,
    ):
        """
        Short description

        Longer description

        Parameters
        ----------
        """

        log_msg('Collecting ' + asset_type + ' damage and loss results')

        R2D_res_out_types = []
        with open(self.input_file, 'r') as f:
            input_data = json.load(f)
        requested_output = input_data['outputs']
        for key, item in requested_output.items():
            if item:
                R2D_res_out_types.append(key)

        run_path = self.run_dir

        if asset_type != '':
            run_path = posixpath.join(run_path, asset_type)

        os.chdir(run_path)

        min_id = min(
            [int(x['id']) for x in asst_data]
        )  # min_id = int(asst_data[0]['id'])
        max_id = max(
            [int(x['id']) for x in asst_data]
        )  # max_id = int(asst_data[0]['id'])

        #
        # TODO: ugly, ugly, I know.
        # Only temporary solution while we have both Pelicuns in parallel
        # FMK - bug fix adding check on DL, not in siteResponse input file
        #

        if (
            'DL' in self.workflow_apps
            and self.workflow_apps['DL'][asset_type].name == 'Pelicun3'
        ):
            initialize_dicts = True
            for a_i, asst in enumerate(asst_data):
                bldg_dir = Path(os.path.dirname(asst_data[a_i]['file'])).resolve()
                main_dir = bldg_dir
                assetTypeHierarchy = [bldg_dir.name]
                while main_dir.parent.name != 'Results':
                    main_dir = bldg_dir.parent
                    assetTypeHierarchy = [main_dir.name] + assetTypeHierarchy

                asset_id = asst['id']
                asset_dir = bldg_dir / asset_id

                # always get the AIM info

                AIM_file = None

                if f'{asset_id}-AIM_ap.json' in os.listdir(asset_dir):
                    AIM_file = asset_dir / f'{asset_id}-AIM_ap.json'

                elif f'{asset_id}-AIM.json' in os.listdir(asset_dir):
                    AIM_file = asset_dir / f'{asset_id}-AIM.json'

                else:
                    # skip this asset if there is no AIM file available
                    show_warning(
                        f"Couldn't find AIM file for {assetTypeHierarchy[-1]} {asset_id}"
                    )
                    continue

                with open(AIM_file, 'r', encoding='utf-8') as f:
                    AIM_data_i = json.load(f)

                sample_size = AIM_data_i['Applications']['DL']['ApplicationData'][
                    'Realizations'
                ]

                # initialize the output dict if this is the first asset
                if initialize_dicts:
                    # We assume all assets have the same output sample size
                    # Variable sample size doesn't seem to make sense
                    realizations = {
                        rlz_i: {asset_type: {}} for rlz_i in range(sample_size)
                    }

                    # We also create a dict to collect deterministic info, i.e.,
                    # data that is identical for all realizations
                    deterministic = {asset_type: {}}
                    initialize_dicts = False

                # Check if the asset type hierarchy exist in deterministic and
                # realizations. Create a hierarchy if it doesn't exist.
                deter_pointer = deterministic
                rlzn_pointer = {
                    rlz_i: realizations[rlz_i] for rlz_i in range(sample_size)
                }
                for assetTypeIter in assetTypeHierarchy:
                    if assetTypeIter not in deter_pointer.keys():
                        deter_pointer.update({assetTypeIter: {}})
                    deter_pointer = deter_pointer[assetTypeIter]
                    for rlz_i in range(sample_size):
                        if assetTypeIter not in rlzn_pointer[rlz_i].keys():
                            rlzn_pointer[rlz_i].update({assetTypeIter: {}})
                        rlzn_pointer[rlz_i] = rlzn_pointer[rlz_i][assetTypeIter]

                # Currently, all GI data is deterministic
                GI_data_i_det = AIM_data_i['GeneralInformation']

                # TODO: later update this to handle probabilistic GI attributes
                GI_data_i_prob = {}

                for rlz_i in range(sample_size):
                    rlzn_pointer[rlz_i].update(
                        {asset_id: {'GeneralInformation': GI_data_i_prob}}
                    )

                deter_pointer.update(
                    {asset_id: {'GeneralInformation': GI_data_i_det}}
                )
                deter_pointer[asset_id].update({'R2Dres': {}})

                if 'EDP' in out_types:
                    edp_out_file_i = 'DEM_sample.json'

                    if edp_out_file_i not in os.listdir(asset_dir):
                        show_warning(
                            f"Couldn't find EDP file for {assetTypeHierarchy[-1]} {asset_id}"
                        )

                    else:
                        with open(
                            asset_dir / edp_out_file_i, 'r', encoding='utf-8'
                        ) as f:
                            edp_data_i = json.load(f)

                        # remove the ONE demand
                        edp_data_i.pop('ONE-0-1')

                        # extract EDP unit info
                        edp_units = edp_data_i['Units']
                        del edp_data_i['Units']

                        # parse the demand data into a DataFrame
                        # we assume demands are stored in JSON with a SimpleIndex
                        edp_data_i = pd.DataFrame(edp_data_i)

                        # convert to a realization-by-realization format
                        edp_output = {
                            int(rlz_i): {
                                col: float(edp_data_i.loc[rlz_i, col])
                                for col in edp_data_i.columns
                            }
                            for rlz_i in edp_data_i.index
                        }

                        # save the EDP intensities in each realization
                        for rlz_i in range(sample_size):
                            rlzn_pointer[rlz_i][asset_id].update(
                                {'Demand': edp_output[rlz_i]}
                            )

                        # save the EDP units
                        deter_pointer[asset_id].update(
                            {'Demand': {'Units': edp_units}}
                        )
                        if 'EDP' in R2D_res_out_types:
                            pass
                            # meanValues = edp_data_i.mean()
                            # stdValues = edp_data_i.std()
                            # r2d_res_edp = dict()
                            # for key in edp_data_i.columns:
                            #     meanKey = f'R2Dres_mean_{key}_{edp_units[key]}'
                            #     stdKey = f'R2Dres_std_{key}_{edp_units[key]}'
                            #     r2d_res_edp.update({meanKey:meanValues[key],\
                            #                         stdKey:stdValues[key]})
                            # r2d_res_i =  deter_pointer[asset_id].get('R2Dres', {})
                            # r2d_res_i.update(r2d_res_edp)
                            # deter_pointer[asset_id].update({
                            #     "R2Dres":r2d_res_i
                            # })
                if 'DMG' in out_types:
                    dmg_out_file_i = 'DMG_grp.json'

                    if dmg_out_file_i not in os.listdir(asset_dir):
                        show_warning(
                            f"Couldn't find DMG file for {assetTypeHierarchy[-1]} {asset_id}"
                        )

                    else:
                        with open(
                            asset_dir / dmg_out_file_i, 'r', encoding='utf-8'
                        ) as f:
                            dmg_data_i = json.load(f)

                        # remove damage unit info
                        del dmg_data_i['Units']

                        # parse damage data into a DataFrame
                        dmg_data_i = pd.DataFrame(dmg_data_i)

                        # convert to realization-by-realization format
                        dmg_output = {}
                        for rlz_i in dmg_data_i.index:
                            rlz_output = {}

                            for col in dmg_data_i.columns:
                                if not pd.isna(dmg_data_i.loc[rlz_i, col]):
                                    rlz_output.update(
                                        {col: int(dmg_data_i.loc[rlz_i, col])}
                                    )

                            dmg_output.update({rlz_i: rlz_output})

                        # we assume that damage information is condensed
                        # TODO: implement condense_ds flag in DL_calc
                        for rlz_i in range(sample_size):
                            rlzn_pointer[rlz_i][asset_id].update(
                                {'Damage': dmg_output[rlz_i]}
                            )
                        if 'DM' in R2D_res_out_types:
                            # use forward fill in case of multiple modes
                            meanValues = dmg_data_i.mode().ffill().mean()
                            stdValues = dmg_data_i.std()
                            r2d_res_dmg = dict()
                            # for key in dmg_data_i.columns:
                            #     meanKey = f'R2Dres_mode_{key}'
                            #     stdKey = f'R2Dres_std_{key}'
                            #     r2d_res_dmg.update({meanKey:meanValues[key],\
                            #                         stdKey:stdValues[key]})
                            r2d_res_dmg.update(
                                {
                                    'R2Dres_MostLikelyCriticalDamageState': dmg_data_i.max(
                                        axis=1
                                    )
                                    .mode()
                                    .mean()
                                }
                            )
                            r2d_res_i = deter_pointer[asset_id].get('R2Dres', {})
                            r2d_res_i.update(r2d_res_dmg)
                            deter_pointer[asset_id].update({'R2Dres': r2d_res_i})

                if 'DV' in out_types:
                    dv_out_file_i = 'DV_repair_grp.json'

                    if dv_out_file_i not in os.listdir(asset_dir):
                        show_warning(
                            f"Couldn't find DV file for {assetTypeHierarchy[-1]} {asset_id}"
                        )

                    else:
                        with open(
                            asset_dir / dv_out_file_i, 'r', encoding='utf-8'
                        ) as f:
                            dv_data_i = json.load(f)

                        # extract DV unit info
                        dv_units = dv_data_i['Units']
                        del dv_data_i['Units']

                        # parse decision variable data into a DataFrame
                        dv_data_i = pd.DataFrame(dv_data_i)

                        # get a list of dv types
                        dv_types = np.unique(
                            [col.split('-')[0] for col in dv_data_i.columns]
                        )

                        # convert to realization-by-realization format
                        dv_output = {
                            int(rlz_i): {
                                dv_type: {
                                    col[len(dv_type) + 1 :]: float(
                                        dv_data_i.loc[rlz_i, col]
                                    )
                                    for col in dv_data_i.columns
                                    if col.startswith(dv_type)
                                }
                                for dv_type in dv_types
                            }
                            for rlz_i in dv_data_i.index
                        }

                        # save loss data
                        for rlz_i in range(sample_size):
                            rlzn_pointer[rlz_i][asset_id].update(
                                {'Loss': {'Repair': dv_output[rlz_i]}}
                            )

                        # save DV units
                        deter_pointer[asset_id].update({'Loss': {'Units': dv_units}})

                        if 'DV' in R2D_res_out_types:
                            r2d_res_dv = dict()
                            cost_columns = [
                                col
                                for col in dv_data_i.columns
                                if col.startswith('Cost')
                            ]
                            if len(cost_columns) != 0:
                                cost_data = dv_data_i[cost_columns].mean()
                                cost_data_std = dv_data_i[cost_columns].std()
                                cost_key = cost_data.idxmax()
                                meanKey = (
                                    f'R2Dres_mean_RepairCost_{dv_units[cost_key]}'
                                )
                                stdKey = (
                                    f'R2Dres_std_RepairCost_{dv_units[cost_key]}'
                                )
                                r2d_res_dv.update(
                                    {
                                        meanKey: cost_data[cost_key],
                                        stdKey: cost_data_std[cost_key],
                                    }
                                )
                            time_columns = [
                                col
                                for col in dv_data_i.columns
                                if col.startswith('Time')
                            ]
                            if len(time_columns) != 0:
                                time_data = dv_data_i[time_columns].mean()
                                time_data_std = dv_data_i[time_columns].std()
                                time_key = time_data.idxmax()
                                meanKey = (
                                    f'R2Dres_mean_RepairTime_{dv_units[time_key]}'
                                )
                                stdKey = (
                                    f'R2Dres_std_RepairTime_{dv_units[time_key]}'
                                )
                                r2d_res_dv.update(
                                    {
                                        meanKey: time_data[time_key],
                                        stdKey: time_data_std[time_key],
                                    }
                                )

                            r2d_res_i = deter_pointer[asset_id].get('R2Dres', {})
                            r2d_res_i.update(r2d_res_dv)
                            deter_pointer[asset_id].update({'R2Dres': r2d_res_i})

            # This is also ugly but necessary for backward compatibility so that
            # file structure created from apps other than GeoJSON_TO_ASSET can be
            # dealt with
            if len(assetTypeHierarchy) == 1:
                if assetTypeHierarchy[0] == 'Buildings':
                    deterministic = {
                        'Buildings': {'Building': deterministic['Buildings']}
                    }
                    for rlz_i in realizations.keys():
                        realizations[rlz_i] = {
                            'Buildings': {
                                'Building': realizations[rlz_i]['Buildings']
                            }
                        }
                else:
                    deterministic = {assetTypeHierarchy[0]: deterministic}
                    for rlz_i in realizations.keys():
                        realizations[rlz_i] = {
                            assetTypeHierarchy[0]: realizations[rlz_i]
                        }

            # save outputs to JSON files
            for rlz_i, rlz_data in realizations.items():
                with open(
                    main_dir / f'{asset_type}_{rlz_i}.json', 'w', encoding='utf-8'
                ) as f:
                    json.dump(rlz_data, f, indent=2)

            with open(
                main_dir / f'{asset_type}_det.json', 'w', encoding='utf-8'
            ) as f:
                json.dump(deterministic, f, indent=2)

        else:
            # This is legacy for Pelicun 2 runs
            out_types = ['IM', 'BIM', 'EDP', 'DM', 'DV', 'every_realization']

            if headers is None:
                headers = dict(
                    IM=[0, 1, 2, 3],
                    AIM=[
                        0,
                    ],
                    EDP=[0, 1, 2, 3],
                    DM=[0, 1, 2],
                    DV=[0, 1, 2, 3],
                )

            for out_type in out_types:
                if (self.output_types is None) or (
                    self.output_types.get(out_type, False)
                ):
                    if out_type == 'every_realization':
                        realizations_EDP = None
                        realizations_DL = None

                        for asst in asst_data:
                            print('ASSET', asst)
                            asst_file = asst['file']

                            # Get the folder containing the results
                            aimDir = os.path.dirname(asst_file)

                            asst_id = asst['id']
                            min_id = min(int(asst_id), min_id)
                            max_id = max(int(asst_id), max_id)

                            # save all EDP realizations

                            df_i = pd.read_csv(
                                aimDir + '/' + asst_id + '/response.csv',
                                header=0,
                                index_col=0,
                            )

                            if realizations_EDP == None:
                                realizations_EDP = dict(
                                    [(col, []) for col in df_i.columns]
                                )

                            for col in df_i.columns:
                                vals = df_i.loc[:, col].to_frame().T
                                vals.index = [
                                    asst_id,
                                ]
                                realizations_EDP[col].append(vals)

                            # If damage and loss assessment is part of the workflow
                            # then save the DL outputs too
                            if 'DL' in self.workflow_apps.keys():
                                try:
                                    # if True:
                                    df_i = pd.read_csv(
                                        aimDir + '/' + asst_id + f'/DL_summary.csv',
                                        header=0,
                                        index_col=0,
                                    )

                                    if realizations_DL == None:
                                        realizations_DL = dict(
                                            [(col, []) for col in df_i.columns]
                                        )

                                    for col in df_i.columns:
                                        vals = df_i.loc[:, col].to_frame().T
                                        vals.index = [
                                            asst_id,
                                        ]
                                        realizations_DL[col].append(vals)

                                except:
                                    log_msg(
                                        f'Error reading DL realization data for asset {asset_type} {asst_id}',
                                        prepend_timestamp=False,
                                    )

                        for d_type in realizations_EDP.keys():
                            d_agg = pd.concat(
                                realizations_EDP[d_type], axis=0, sort=False
                            )

                            with warnings.catch_warnings():
                                warnings.simplefilter(action='ignore')

                                d_agg.to_hdf(
                                    f'realizations_{min_id}-{max_id}.hdf',
                                    f'EDP-{d_type}',
                                    mode='a',
                                    format='fixed',
                                )

                        if 'DL' in self.workflow_apps.keys():
                            for d_type in realizations_DL.keys():
                                d_agg = pd.concat(
                                    realizations_DL[d_type], axis=0, sort=False
                                )
                                # d_agg.sort_index(axis=0, inplace=True)

                                with warnings.catch_warnings():
                                    warnings.simplefilter(action='ignore')

                                    d_agg.to_hdf(
                                        f'realizations_{min_id}-{max_id}.hdf',
                                        f'DL-{d_type}',
                                        mode='a',
                                        format='fixed',
                                    )

                    else:
                        out_list = []
                        count = 0
                        for asst in asst_data:
                            if count % self.numP == self.procID:
                                print('ASSET', self.procID, self.numP, asst['file'])
                                asst_file = asst['file']

                                # Get the folder containing the results
                                aimDir = os.path.dirname(asst_file)

                                asst_id = asst['id']
                                min_id = min(int(asst_id), min_id)
                                max_id = max(int(asst_id), max_id)

                                try:
                                    # if True:

                                    csvPath = (
                                        aimDir + '/' + asst_id + f'/{out_type}.csv'
                                    )

                                    # EDP data
                                    df_i = pd.read_csv(
                                        csvPath,
                                        header=headers[out_type],
                                        index_col=0,
                                    )

                                    df_i.index = [
                                        asst_id,
                                    ]

                                    out_list.append(df_i)

                                except:
                                    log_msg(
                                        f'Error reading {out_type} data for asset {asset_type} {asst_id}',
                                        prepend_timestamp=False,
                                    )

                            # increment counter
                            count = count + 1

                        # save the collected DataFrames as csv files
                        if self.procID == 0:
                            outPath = posixpath.join(run_path, f'{out_type}.csv')
                        else:
                            outPath = posixpath.join(
                                run_path, f'{out_type}_tmp_{self.procID}.csv'
                            )

                        # if not P0 output file & barrier
                        if self.procID != 0:
                            out_agg = (
                                pd.DataFrame()
                                if len(out_list) < 1
                                else pd.concat(out_list, axis=0, sort=False)
                            )
                            out_agg.to_csv(outPath)
                            self.comm.Barrier()

                        else:
                            # P0 if parallel & parallel, barrier then read other, and merge
                            if self.numP > 1:
                                self.comm.Barrier()

                                # fileList = []
                                for i in range(1, self.numP):
                                    fileToAppend = posixpath.join(
                                        run_path, f'{out_type}_tmp_{i}.csv'
                                    )
                                    # fileList.append(fileToAppend)
                                    out_list.append(
                                        pd.read_csv(
                                            fileToAppend,
                                            header=headers[out_type],
                                            index_col=0,
                                        )
                                    )

                            # write file
                            out_agg = (
                                pd.DataFrame()
                                if len(out_list) < 1
                                else pd.concat(out_list, axis=0, sort=False)
                            )
                            out_agg.to_csv(outPath)

        log_msg(
            'Damage and loss results collected successfully.',
            prepend_timestamp=False,
        )
        log_div()

    def compile_r2d_results_geojson(self, asset_files):
        run_path = self.run_dir
        with open(self.input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        with open(run_path / 'Results_det.json', encoding='utf-8') as f:
            res_det = json.load(f)
        metadata = {
            'Name': input_data['Name'],
            'Units': input_data['units'],
            'Author': input_data['Author'],
            'WorkflowType': input_data['WorkflowType'],
            'Time': datetime.now().strftime('%m-%d-%Y %H:%M:%S'),
        }
        ## create the geojson for R2D visualization
        geojson_result = {
            'type': 'FeatureCollection',
            'crs': {
                'type': 'name',
                'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'},
            },
            'metadata': metadata,
            'features': [],
        }
        for asset_type in asset_files.keys():
            for assetSubtype, subtypeResult in res_det[asset_type].items():
                allAssetIds = sorted([int(x) for x in subtypeResult.keys()])
                for asset_id in allAssetIds:
                    ft = {'type': 'Feature'}
                    asst_GI = subtypeResult[str(asset_id)][
                        'GeneralInformation'
                    ].copy()
                    asst_GI.update({'assetType': asset_type})
                    try:
                        if 'geometry' in asst_GI:
                            asst_geom = shapely.wkt.loads(asst_GI['geometry'])
                            asst_geom = shapely.geometry.mapping(asst_geom)
                            asst_GI.pop('geometry')
                        elif 'Footprint' in asst_GI:
                            asst_geom = json.loads(asst_GI['Footprint'])['geometry']
                            asst_GI.pop('Footprint')
                        else:
                            # raise ValueError("No valid geometric information in GI.")
                            asst_lat = asst_GI['location']['latitude']
                            asst_lon = asst_GI['location']['longitude']
                            asst_geom = {
                                'type': 'Point',
                                'coordinates': [asst_lon, asst_lat],
                            }
                            asst_GI.pop('location')
                    except:
                        warnings.warn(
                            UserWarning(
                                f'Geospatial info is missing in {assetSubtype} {asset_id}'
                            )
                        )
                        continue
                    if asst_GI.get('units', None) is not None:
                        asst_GI.pop('units')
                    ft.update({'geometry': asst_geom})
                    ft.update({'properties': asst_GI})
                    ft['properties'].update(subtypeResult[str(asset_id)]['R2Dres'])
                    geojson_result['features'].append(ft)
        with open(run_path / 'R2D_results.geojson', 'w', encoding='utf-8') as f:
            json.dump(geojson_result, f, indent=2)

    def combine_assets_results(self, asset_files):
        asset_types = list(asset_files.keys())
        for asset_type in asset_types:
            if self.workflow_apps['DL'][asset_type].name != 'Pelicun3':
                # isPelicun3 = False
                asset_files.pop(asset_type)
        if asset_files:  # If any asset_type uses Pelicun3 as DL app
            with open(self.input_file, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            sample_size = []
            for asset_type, assetIt in asset_files.items():
                sample_size.append(
                    input_data['Applications']['DL'][asset_type]['ApplicationData'][
                        'Realizations'
                    ]
                )
            sample_size = min(sample_size)
            ## Create the Results_det.json and Results_rlz_i.json for recoverary
            deterministic = {}
            realizations = {rlz_i: {} for rlz_i in range(sample_size)}
            for asset_type in asset_files.keys():
                asset_dir = self.run_dir / asset_type
                determine_file = asset_dir / f'{asset_type}_det.json'
                with open(determine_file, 'r', encoding='utf-8') as f:
                    determ_i = json.load(f)
                deterministic.update(determ_i)
                for rlz_i in range(sample_size):
                    rlz_i_file = asset_dir / f'{asset_type}_{rlz_i}.json'
                    with open(rlz_i_file, 'r', encoding='utf-8') as f:
                        rlz_i_i = json.load(f)
                    realizations[rlz_i].update(rlz_i_i)

            determine_file = self.run_dir / 'Results_det.json'
            with open(determine_file, 'w', encoding='utf-8') as f:
                json.dump(deterministic, f, indent=2)
            for rlz_i, rlz_data in realizations.items():
                with open(
                    self.run_dir / f'Results_{rlz_i}.json', 'w', encoding='utf-8'
                ) as f:
                    json.dump(rlz_data, f, indent=2)
        else:
            pass
            # print("Visualizing results of asset types besides buildings is only supported when Pelicun3 is used as the DL for all asset types")
