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

import pprint

import shutil
import subprocess

from copy import deepcopy

import warnings

import numpy as np
import pandas as pd

import platform
from pathlib import Path, PurePath

#import posixpath
#import ntpath


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

                with open(filepath, 'w') as f:
                    f.write('')

            except:
                raise ValueError(f"The filepath provided does not point to a "
                                 f"valid location: {filepath}")

    @property
    def print_log(self):
        return self._print_log

    @print_log.setter
    def print_log(self, value):
        self._print_log = str2bool(value)

    def reset_log_strings(self):

        if self._log_show_ms:
            self._log_time_format = '%H:%M:%S:%f'
            self._log_pref = ' ' * 16 # the length of the time string in the log file
            self._log_div = '-' * (80 - 17) # to have a total length of 80 with the time added
        else:
            self._log_time_format = '%H:%M:%S'
            self._log_pref = ' ' * 9
            self._log_div = '-' * (80 - 9)

options = Options()

log_file = None

def set_options(config_options):

    if config_options is not None:

        for key, value in config_options.items():

            if key == "LogShowMS":
                options.log_show_ms = value
            elif key == "LogFile":
                options.log_file = value
            elif key == "PrintLog":
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

    log_msg(msg, prepend_timestamp = prepend_timestamp,
            prepend_blank_space = prepend_blank_space)


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

        if (prepend_timestamp and (msg_i==0)):
            formatted_msg = '{} {}'.format(
                datetime.now().strftime(options.log_time_format), msg_line)
        elif prepend_timestamp:
            formatted_msg = options.log_pref + msg_line
        elif prepend_blank_space:
            formatted_msg = options.log_pref + msg_line
        else:
            formatted_msg = msg_line

        if options.print_log:
            print(formatted_msg)

        if globals()['log_file'] is not None:
            with open(globals()['log_file'], 'a') as f:
                f.write('\n'+formatted_msg)

def log_error(msg):
    """
    Print an error message to the screen

    Parameters
    ----------
    msg: string
       Message to print.
    """

    log_div()
    log_msg(''*(80-21-6) + ' ERROR')
    log_msg(msg)
    log_div()

def print_system_info():

    log_msg('System Information:',
            prepend_timestamp=False, prepend_blank_space=False)
    log_msg(f'  local time zone: {datetime.utcnow().astimezone().tzinfo}\n'
            f'  start time: {datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}\n'
            f'  python: {sys.version}\n'
            f'  numpy: {np.__version__}\n'
            f'  pandas: {pd.__version__}\n',
            prepend_timestamp=False, prepend_blank_space=False)

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

        command = '"{}" "{}" '.format(python_exe, command_list[1])# + ' '.join(command_list[2:])

        for command_arg in command_list[2:]:
            command += '"{}" '.format(command_arg)
    else:
        command = '"{}" '.format(command_list[0])# + ' '.join(command_list[1:])

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

        import importlib # only import this when it's needed

        command_list = command.split()[1:]
        #py_args = command_list[1:]

        # get the dir and file name
        py_script_dir, py_script_file = os.path.split(command_list[0][1:-1])

        # add the dir to the path
        sys.path.insert(0, py_script_dir)

        # import the file
        py_script = importlib.__import__(
            py_script_file[:-3], globals(), locals(), ['main',], 0)

        # remove the quotes from the arguments
        arg_list = [c[1:-1] for c in command_list[1:]]

        py_script.main(arg_list)

        return "", ""

    else:

        try:
            result = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
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
        return result.decode(sys.stdout.encoding), returncode

def show_warning(warning_msg):
    warnings.warn(UserWarning(warning_msg))

def resolve_path(target_path, ref_path):

    ref_path = Path(ref_path)

    target_path = str(target_path).strip()

    while target_path.startswith('/') or target_path.startswith('\\'):
        target_path = target_path[1:]

    if target_path == "":
        target_path = ref_path

    else:
        target_path = Path(target_path)

        if not target_path.exists():
            target_path = Path(ref_path) / target_path

        if target_path.exists():
            target_path = target_path.resolve()
        else:
            raise ValueError(
                f"{target_path} does not point to a valid location")

    return target_path

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

        self.name = app_info['Name']
        self.app_type = app_type
        self.rel_path = app_info['ExecutablePath']
        self.app_spec_inputs = app_info.get('ApplicationSpecificInputs',[])

        self.inputs = api_info['Inputs']
        self.outputs = api_info['Outputs']
        self.defaults = api_info['DefaultValues']

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

                    self.pref[preference] = resolve_path(
                        self.pref[preference], ref_path)

    def get_command_list(self, app_path, force_posix=False):
        """
        Short description

        Parameters
        ----------
        app_path: Path
            Explain...
        """

        abs_path = Path(app_path) / self.rel_path
        #abs_path = posixpath.join(app_path, self.rel_path)

        arg_list = []

        if str(abs_path).endswith('.py'):
            arg_list.append('python')

        if force_posix:
            arg_list.append(u'{}'.format(abs_path.as_posix()))
        else:
            arg_list.append(u'{}'.format(abs_path))

        for in_arg in self.inputs:
            arg_list.append(u'--{}'.format(in_arg['id']))

            # Default values are protected, they cannot be overwritten simply
            # by providing application specific inputs in the config file
            if in_arg['type'] == 'workflowDefault':
                arg_value = self.defaults[in_arg['id']]

                # If the user also provided an input, let them know that their
                # input is invalid
                if in_arg['id'] in self.pref.keys():
                    log_msg('\nWARNING: Application specific parameters cannot '
                            'overwrite default workflow\nparameters. See the '
                            'documentation on how to edit workflowDefault '
                            'inputs.\n', prepend_timestamp=False,
                            prepend_blank_space=False)

            elif in_arg['id'] in self.pref.keys():
                arg_value = self.pref[in_arg['id']]

            else:
                arg_value = in_arg['default']

            if isinstance(arg_value, Path) and force_posix:
                arg_list.append(u'{}'.format(arg_value.as_posix()))
            else:
                arg_list.append(u'{}'.format(arg_value))

        for out_arg in self.outputs:
            out_id = u'--{}'.format(out_arg['id'])
            if out_id not in arg_list:
                arg_list.append(out_id)

                # Default values are protected, they cannot be overwritten simply
                # by providing application specific inputs in the config file
                if out_arg['type'] == 'workflowDefault':
                    arg_value = self.defaults[out_arg['id']]

                    # If the user also provided an input, let them know that
                    # their input is invalid
                    if out_arg['id'] in self.pref.keys():
                        log_msg('\nWARNING: Application specific parameters '
                                'cannot overwrite default workflow\nparameters. '
                                'See the documentation on how to edit '
                                'workflowDefault inputs.\n',
                                prepend_timestamp=False,
                                prepend_blank_space=False)

                elif out_arg['id'] in self.pref.keys():
                    arg_value = self.pref[out_arg['id']]

                else:
                    arg_value = out_arg['default']

                if isinstance(arg_value, Path) and force_posix:
                    arg_list.append(u'{}'.format(arg_value.as_posix()))
                else:
                    arg_list.append(u'{}'.format(arg_value))

        ASI_list =  [inp['id'] for inp in self.app_spec_inputs]
        for pref_name, pref_value in self.pref.items():
            # only pass those input arguments that are in the registry
            if pref_name in ASI_list:
                pref_id = u'--{}'.format(pref_name)
                if pref_id not in arg_list:
                    arg_list.append(pref_id)

                    if isinstance(pref_value, Path) and force_posix:
                        arg_list.append(u'{}'.format(pref_value.as_posix()))
                    else:
                        arg_list.append(u'{}'.format(pref_value))

        #pp.pprint(arg_list)

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

    def __init__(self, run_type, input_file, app_registry, app_type_list,
        reference_dir=None, working_dir=None, app_dir=None):

        log_msg('Inputs provided:')
        log_msg('workflow input file: {}'.format(input_file),
            prepend_timestamp=False)
        log_msg('application registry file: {}'.format(app_registry),
            prepend_timestamp=False)
        log_msg('run type: {}'.format(run_type),
            prepend_timestamp=False)
        log_div()

        self.optional_apps = ['RegionalEvent', 'Modeling', 'EDP', 'UQ', 'DL']

        self.run_type = run_type
        self.input_file = input_file
        self.app_registry_file = app_registry

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

        # initialize app registry
        self._init_app_registry()

        # parse the application registry
        self._parse_app_registry()

        # parse the input file
        self.workflow_apps = {}
        self._parse_inputs()

    def _init_app_registry(self):
        """
        Initialize the dictionary where we keep the data on available apps.

        """
        self.app_registry = dict([(a, dict()) for a in self.app_type_list])

    def _parse_app_registry(self):
        """
        Load the information about available workflow applications.

        """

        log_msg('Parsing application registry file')

        # open the registry file
        log_msg('Loading the json file...', prepend_timestamp=False)
        with open(self.app_registry_file, 'r') as f:
            app_registry_data = json.load(f)
        log_msg('  OK', prepend_timestamp=False)

        # initialize the app registry
        self._init_app_registry()

        log_msg('Loading default values...', prepend_timestamp=False)

        self.default_values = app_registry_data.get('DefaultValues', None)

        log_msg('  OK', prepend_timestamp=False)

        log_msg('Collecting application data...', prepend_timestamp=False)
        # for each application type
        for app_type in sorted(self.app_registry.keys()):

            # if the registry contains information about it
            app_type_long = app_type+'Applications'
            if app_type_long in app_registry_data:

                # get the list of available applications
                available_apps = app_registry_data[app_type_long]['Applications']
                api_info = app_registry_data[app_type_long]['API']

                # add the default values to the API info
                if self.default_values is not None:
                    api_info.update({'DefaultValues': self.default_values})

                # and store their name and executable location
                for app in available_apps:
                    self.app_registry[app_type][app['Name']] = WorkflowApplication(
                         app_type=app_type, app_info=app, api_info=api_info)

        log_msg('  OK', prepend_timestamp=False)

        log_msg('Available applications:', prepend_timestamp=False)

        for app_type, app_list in self.app_registry.items():
            for app_name, app_object in app_list.items():
                log_msg('  {} : {}'.format(app_type, app_name),
                        prepend_timestamp=False)

        #pp.pprint(self.app_registry)

        log_msg('Successfully parsed application registry',
                prepend_timestamp=False)
        log_div()

    def _parse_inputs(self):
        """
        Load the information about the workflow to run

        """

        log_msg('Parsing workflow input file')

        # open input file
        log_msg('Loading the json file...', prepend_timestamp=False)
        with open(self.input_file, 'r') as f:
            input_data = json.load(f)
        log_msg('  OK', prepend_timestamp=False)

        # store the specified units (if available)
        if 'units' in input_data:
            self.units = input_data['units']

            log_msg('The following units were specified: ',
                prepend_timestamp=False)
            for key, unit in self.units.items():
                log_msg('  {}: {}'.format(key, unit), prepend_timestamp=False)
        else:
            self.units = None
            log_msg('No units specified; using Standard units.',
                    prepend_timestamp=False)

        # store the specified output types
        self.output_types = input_data.get('outputs', None)

        if self.output_types is None:
            default_output_types = {
                "BIM": False,
                "EDP": True,
                "DM": True,
                "DV": True,
                "every_realization": False
            }

            log_msg("Missing output type specification, using default "
                    "settings.", prepend_timestamp=False)
            self.output_types = default_output_types

        else:
            log_msg("The following output_types were requested: ",
                prepend_timestamp=False)
            for out_type, flag in self.output_types.items():
                if flag:
                    log_msg(f'  {out_type}', prepend_timestamp=False)

        # replace the default values, if needed
        default_values = input_data.get('DefaultValues', None)

        if default_values is not None:
            log_msg("The following workflow defaults were overwritten:",
                prepend_timestamp=False)

            for key, value in default_values.items():

                if key in self.default_values.keys():
                    self.default_values[key] = value

                else:
                    self.default_values.update({key: value})

                log_msg(f"  {key}: {value}", prepend_timestamp=False)

        # parse the shared data in the input file
        self.shared_data = {}
        for shared_key in ['RegionalEvent',]:
            value = input_data.get(shared_key, None)
            if value != None:
                self.shared_data.update({shared_key: value})

        # parse the location of the run_dir
        if self.working_dir is not None:
            self.run_dir = self.working_dir
        elif 'runDir' in input_data:
            self.run_dir = Path(input_data['runDir'])
        #else:
        #    raise WorkFlowInputError('Need a runDir entry in the input file')

        # parse the location(s) of the applications directory
        if 'localAppDir' in input_data:
            self.app_dir_local = input_data['localAppDir']
        #else:
        #    raise WorkFlowInputError('Need a localAppDir entry in the input file')

        if 'remoteAppDir' in input_data:
            self.app_dir_remote = Path(input_data['remoteAppDir'])
        else:
            self.app_dir_remote = self.app_dir_local
            log_msg('remoteAppDir not specified. Using the value provided for '
                'localAppDir instead.', prepend_timestamp=False)

        if 'referenceDir' in input_data:
            self.reference_dir = input_data['referenceDir']

        for loc_name, loc_val in zip(
            ['Run dir', 'Local applications dir','Remote applications dir',
             'Reference dir'],
            [self.run_dir, self.app_dir_local, self.app_dir_remote,
             self.reference_dir]):
            log_msg('{} : {}'.format(loc_name, loc_val),
                    prepend_timestamp=False)

        if 'Building' in self.app_type_list:
            self.building_file_name = "buildings.json"
            #if 'buildingFile' in input_data:
            #    self.building_file_name = input_data['buildingFile']
            #else:
            #    self.building_file_name = "buildings.json"
            #log_msg('\tbuilding file name: {}'.format(self.building_file_name))


        # get the list of requested applications
        log_msg('\nParsing the requested list of applications...',
                prepend_timestamp=False)
        if 'Applications' in input_data:
            requested_apps = input_data['Applications']
        else:
            raise WorkFlowInputError('Need an Applications entry in the input file')

        # create the requested applications

        # Events are special because they are in an array
        if 'Events' in requested_apps:
            if len(requested_apps['Events']) > 1:
                raise WorkFlowInputError('Currently, WHALE only supports a single event.')
            for event in requested_apps['Events'][:1]: #this limitation can be relaxed in the future
                if 'EventClassification' in event:
                    eventClassification = event['EventClassification']
                    if eventClassification in ['Earthquake', 'Wind', 'Hurricane', 'Flood','Hydro'] :

                        app_object = deepcopy(
                            self.app_registry['Event'].get(event['Application']))

                        if app_object is None:
                            raise WorkFlowInputError(
                                'Application entry missing for {}'.format('Events'))

                        app_object.set_pref(event['ApplicationData'],
                                            self.reference_dir)
                        self.workflow_apps['Event'] = app_object

                    else:
                        raise WorkFlowInputError(
                            ('Currently, only earthquake and wind events are supported. '
                             'EventClassification must be Earthquake, not {}'
                             ).format(eventClassification))
                else:
                    raise WorkFlowInputError('Need Event Classification')
        else:
            raise WorkFlowInputError('Need an Events Entry in Applications')

        for app_type in self.app_type_list:
            if app_type != 'Event':
                if app_type in requested_apps:

                    app_object = deepcopy(
                        self.app_registry[app_type].get(
                            requested_apps[app_type]['Application']))

                    if app_object is None:
                        raise WorkFlowInputError(
                            'Application entry missing for {}'.format(app_type))

                    # only assign the app to the workflow if it has an executable
                    if app_object.rel_path is None:
                        log_msg(
                            f'  {requested_apps[app_type]["Application"]} is '
                            'a passive application (i.e., it does not invoke '
                            'any calculation within the workflow.',
                            prepend_timestamp=False)

                    else:
                        app_object.set_pref(requested_apps[app_type]['ApplicationData'],
                                            self.reference_dir)
                        self.workflow_apps[app_type] = app_object

                else:
                    if app_type in self.optional_apps:
                        self.app_registry.pop(app_type, None)
                        log_msg(f'  No {app_type} among requested applications.',
                                prepend_timestamp=False)
                    else:
                        raise WorkFlowInputError(
                            f'Need {app_type} entry in Applications')

        for app_type in self.optional_apps:
            if (app_type not in self.app_registry) and (app_type in self.app_type_list):
                self.app_type_list.remove(app_type)

        log_msg('\nRequested workflow:', prepend_timestamp=False)
        for app_type, app_object in self.workflow_apps.items():
            log_msg('  {} : {}'.format(app_type, app_object.name),
                    prepend_timestamp=False)

        log_msg('\nSuccessfully parsed workflow inputs', prepend_timestamp=False)
        log_div()

    def create_building_files(self):
        """
        Short description

        Longer description

        Parameters
        ----------

        """

        log_msg('Creating files for individual buildings')

        building_file = self.run_dir / self.building_file_name
        #building_file = posixpath.join(self.run_dir, self.building_file_name)

        bldg_app = self.workflow_apps['Building']

        # TODO: not elegant code, fix later
        os.chdir(self.run_dir)

        # filter buildings (if needed)
        bldg_filter = bldg_app.pref.get('filter', None)
        if bldg_filter == "":
            del bldg_app.pref['filter']
            bldg_filter = None

        if bldg_filter is not None:
            bldgs = [bs.split('-') for bs in bldg_filter.split(',')]

            building_file = Path(str(building_file).replace(
                ".json", f"{bldgs[0][0]}-{bldgs[-1][-1]}.json"))

        # store the path to the building file
        self.building_file_path = building_file

        for output in bldg_app.outputs:
            if output['id'] == 'buildingFile':
                output['default'] = building_file

        bldg_command_list = bldg_app.get_command_list(
            app_path = self.app_dir_local)

        bldg_command_list.append(u'--getRV')

        command = create_command(bldg_command_list)

        log_msg('Creating initial building files...',
                prepend_timestamp=False)
        log_msg('\n{}\n'.format(command), prepend_timestamp=False,
                prepend_blank_space=False)

        result, returncode = run_command(command)

        log_msg('Output: ', prepend_timestamp=False, prepend_blank_space=False)
        log_msg('\n{}\n'.format(result), prepend_timestamp=False,
                prepend_blank_space=False)

        # Append workflow settings to the BIM file
        log_msg('Appending additional settings to the BIM files...')

        with open(building_file, 'r') as f:
            bldg_data = json.load(f)

        for bldg in bldg_data:

            BIM_file = bldg['file']

            # Open the BIM file and add the unit information to it
            with open(BIM_file, 'r') as f:
                BIM_data = json.load(f)

            if self.units != None:
                BIM_data.update({'units': self.units})

                # TODO: remove this after all apps have been updated to use the
                # above location to get units
                BIM_data['GeneralInformation'].update({'units': self.units})

            BIM_data.update({'outputs': self.output_types})

            for key, value in self.shared_data.items():
                BIM_data[key] = value

            with open(BIM_file, 'w') as f:
                json.dump(BIM_data, f, indent=2)

        log_msg('Building files successfully created.', prepend_timestamp=False)
        log_div()

        return building_file

    def perform_regional_event(self):
        """
        Run an application to simulate a regional-scale hazard event.

        Longer description

        Parameters
        ----------

        """

        log_msg('Simulating regional event...')

        reg_event_app = self.workflow_apps['RegionalEvent']

        reg_event_command_list = reg_event_app.get_command_list(
            app_path = self.app_dir_local)

        command = create_command(reg_event_command_list)

        log_msg('\n{}\n'.format(command), prepend_timestamp=False,
                prepend_blank_space=False)

        result, returncode = run_command(command)

        log_msg('Output: ', prepend_timestamp=False, prepend_blank_space=False)
        log_msg('\n{}\n'.format(result), prepend_timestamp=False,
                prepend_blank_space=False)

        log_msg('Regional event successfully simulated.',
                prepend_timestamp=False)
        log_div()

    def perform_regional_mapping(self, building_file):
        """
        Short description

        Longer description

        Parameters
        ----------

        """

        log_msg('Creating regional mapping...')

        reg_mapping_app = self.workflow_apps['RegionalMapping']

        # TODO: not elegant code, fix later
        for input_ in reg_mapping_app.inputs:
            if input_['id'] == 'buildingFile':
                input_['default'] = str(building_file)

        reg_mapping_app.inputs.append({
            'id': 'filenameEVENTgrid',
            'type': 'path',
            'default': resolve_path(
                self.shared_data['RegionalEvent']['eventFile'],
                self.reference_dir)
            })

        reg_mapping_command_list = reg_mapping_app.get_command_list(
            app_path = self.app_dir_local)

        command = create_command(reg_mapping_command_list)

        log_msg('\n{}\n'.format(command), prepend_timestamp=False,
                prepend_blank_space=False)

        result, returncode = run_command(command)

        log_msg('Output: ', prepend_timestamp=False,
                prepend_blank_space=False)
        log_msg('\n{}\n'.format(result), prepend_timestamp=False,
                prepend_blank_space=False)

        log_msg('Regional mapping successfully created.',
                prepend_timestamp=False)
        log_div()

    def init_simdir(self, bldg_id=None, BIM_file = 'BIM.json'):
        """
        Short description

        Longer description

        Parameters
        ----------

        """
        log_msg('Initializing the simulation directory')

        os.chdir(self.run_dir)

        if bldg_id is not None:

            # if the directory already exists, remove its contents
            if bldg_id in os.listdir(self.run_dir):
                shutil.rmtree(bldg_id, ignore_errors=True)

            # create the building_id dir and the template dir
            os.mkdir(bldg_id)
            os.chdir(bldg_id)
            os.mkdir('templatedir')
            os.chdir('templatedir')

            # Make a copy of the BIM file
            shutil.copy(
                src = self.run_dir / BIM_file,
                dst = self.run_dir / f'{bldg_id}/templatedir/{BIM_file}')

            #src = posixpath.join(self.run_dir, BIM_file),
            #dst = posixpath.join(
            #    self.run_dir,
            #    '{}/templatedir/{}'.format(bldg_id, BIM_file)))

        else:

            for dir_or_file in os.listdir(os.getcwd()):
                if dir_or_file not in ['log.txt', 'templatedir']:
                    if os.path.isdir(dir_or_file):
                        shutil.rmtree(dir_or_file)
                    else:
                        os.remove(dir_or_file)

            os.chdir('templatedir') #TODO: we might want to add a generic id dir to be consistent with the regional workflow here

            # Make a copy of the input file and rename it to BIM.json
            # This is a temporary fix, will be removed eventually.
            dst = Path(os.getcwd()) / BIM_file
            #dst = posixpath.join(os.getcwd(),BIM_file)
            if BIM_file != self.input_file:
                shutil.copy(src = self.input_file, dst = dst)

        log_msg('Simulation directory successfully initialized.',
                prepend_timestamp=False)
        log_div()

    def cleanup_simdir(self, bldg_id):
        """
        Short description

        Longer description

        Parameters
        ----------

        """
        log_msg('Cleaning up the simulation directory.')

        os.chdir(self.run_dir)

        if bldg_id is not None:
            os.chdir(bldg_id)

        workdirs = os.listdir(os.getcwd())
        for workdir in workdirs:
            if 'workdir' in workdir:
                shutil.rmtree(workdir, ignore_errors=True)

        log_msg('Simulation directory successfully cleaned up.',
                prepend_timestamp=False)
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

        log_msg('Working directory successfully initialized.',
                prepend_timestamp=False)
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
            #if os.path.isdir(posixpath.join(self.run_dir, file_or_dir)):
                shutil.rmtree(file_or_dir, ignore_errors=True)

        log_msg('Working directory successfully cleaned up.')
        log_div()


    def create_RV_files(self, app_sequence, BIM_file = 'BIM.json', bldg_id=None): # we will probably need to rename this one
        """
        Short description

        Longer description

        Parameters
        ----------

        """

        log_msg('Creating files with random variables')

        os.chdir(self.run_dir)

        if bldg_id is not None:
            os.chdir(bldg_id)

        os.chdir('templatedir')

        for app_type in self.optional_apps:
            if ((app_type in app_sequence) and
                (app_type not in self.workflow_apps.keys())):
                app_sequence.remove(app_type)

        for app_type in app_sequence:

            workflow_app = self.workflow_apps[app_type]

            if BIM_file is not None:
                workflow_app.defaults['filenameBIM'] = BIM_file
                #for input_var in workflow_app.inputs:
                #    if input_var['id'] == 'filenameBIM':
                #        input_var['default'] = BIM_file

            command_list = workflow_app.get_command_list(
                app_path = self.app_dir_local)

            command_list.append(u'--getRV')

            command = create_command(command_list)

            log_msg('\nRunning {} app for RV...'.format(app_type),
                    prepend_timestamp=False)
            log_msg('\n{}\n'.format(command), prepend_timestamp=False,
                    prepend_blank_space=False)

            result, returncode = run_command(command)

            log_msg('Output: ', prepend_timestamp=False,
                    prepend_blank_space=False)
            log_msg('\n{}\n'.format(result), prepend_timestamp=False,
                    prepend_blank_space=False)

        log_msg('Files with random variables successfully created.',
                prepend_timestamp=False)
        log_div()


    def create_driver_file(self, app_sequence, bldg_id=None):
        """
        Short description

        Longer description

        Parameters
        ----------
        """

        if 'UQ' in self.workflow_apps.keys():
            log_msg('Creating the workflow driver file')

            os.chdir(self.run_dir)

            if bldg_id is not None:
                os.chdir(bldg_id)

            os.chdir('templatedir')

            driver_script = u''

            for app_type in self.optional_apps:
                if ((app_type in app_sequence) and
                    (app_type not in self.workflow_apps.keys())):
                    app_sequence.remove(app_type)

            for app_type in app_sequence:

                if self.run_type in ['set_up', 'runningRemote']:
                    command_list = self.workflow_apps[app_type].get_command_list(
                        app_path = self.app_dir_remote, force_posix = True)

                    driver_script += create_command(command_list, enforced_python='python3') + u'\n'
                else:
                    command_list = self.workflow_apps[app_type].get_command_list(
                        app_path = self.app_dir_local)

                    driver_script += create_command(command_list) + u'\n'

            log_msg('Workflow driver script:', prepend_timestamp=False)
            log_msg('\n{}\n'.format(driver_script), prepend_timestamp=False,
                    prepend_blank_space=False)

            with open('driver','w') as f:
                f.write(driver_script)

            log_msg('Workflow driver file successfully created.',
                    prepend_timestamp=False)
            log_div()
        else:
            log_msg('No UQ requested, workflow driver is not needed.')
            log_div()

    def simulate_response(self, BIM_file = 'BIM.json', bldg_id=None):
        """
        Short description

        Longer description

        Parameters
        ----------
        """

        if 'UQ' in self.workflow_apps.keys():
            log_msg('Running response simulation')

            os.chdir(self.run_dir)

            if bldg_id is not None:
                os.chdir(bldg_id)

            os.chdir('templatedir')

            workflow_app = self.workflow_apps['UQ']

            if BIM_file is not None:
                workflow_app.defaults['filenameBIM'] = BIM_file
                #for input_var in workflow_app.inputs:
                #    if input_var['id'] == 'filenameBIM':
                #        input_var['default'] = BIM_file

            command_list = workflow_app.get_command_list(
                app_path=self.app_dir_local)

            # add the run type to the uq command list
            command_list.append(u'--runType')
            command_list.append(u'{}'.format(self.run_type))

            command = create_command(command_list)

            log_msg('Simulation command:', prepend_timestamp=False)
            log_msg('\n{}\n'.format(command), prepend_timestamp=False,
                    prepend_blank_space=False)

            result, returncode = run_command(command)

            if self.run_type in ['run', 'runningLocal']:

                log_msg('Output: ', prepend_timestamp=False,
                        prepend_blank_space=False)
                log_msg('\n{}\n'.format(result), prepend_timestamp=False,
                        prepend_blank_space=False)

                # create the response.csv file from the dakotaTab.out file
                os.chdir(self.run_dir)
                if bldg_id is not None:
                    os.chdir(bldg_id)
                dakota_out = pd.read_csv('dakotaTab.out', sep=r'\s+', header=0, index_col=0)

                # if the DL is coupled with response estimation, we need to sort the results
                DL_app = self.workflow_apps.get('DL', None)
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

                log_msg('Response simulation finished successfully.',
                        prepend_timestamp=False)

            elif self.run_type in ['set_up', 'runningRemote']:

                log_msg('Response simulation set up successfully',
                        prepend_timestamp=False)

            log_div()

        else:
            log_msg('No UQ requested, response simulation step is skipped.')

            # copy the response.csv from the templatedir to the run dir
            os.chdir(self.run_dir)
            if bldg_id is not None:
                os.chdir(bldg_id)
            shutil.copy(src = 'templatedir/response.csv', dst = 'response.csv')

            log_div()

    def estimate_losses(self, BIM_file = 'BIM.json', bldg_id = None,
        input_file = None, copy_resources=False):
        """
        Short description

        Longer description

        Parameters
        ----------
        """

        if 'DL' in self.workflow_apps.keys():
            log_msg('Running damage and loss assessment')

            os.chdir(self.run_dir)

            if 'Building' not in self.app_type_list:
                # Copy the dakota.json file from the templatedir to the run_dir so that
                # all the required inputs are in one place.
                input_file = PurePath(input_file).name
                #input_file = ntpath.basename(input_file)
                shutil.copy(
                    src = self.run_dir / f'templatedir/{input_file}',
                    dst = self.run_dir / BIM_file)
                #src = posixpath.join(self.run_dir,'templatedir/{}'.format(input_file)),
                #dst = posixpath.join(self.run_dir,BIM_file))
            else:
                # copy the BIM file from the main dir to the building dir
                shutil.copy(
                    src = self.run_dir / BIM_file,
                    dst = self.run_dir / f'{bldg_id}/{BIM_file}')
                #src = posixpath.join(self.run_dir, BIM_file),
                #dst = posixpath.join(self.run_dir,
                #                     '{}/{}'.format(bldg_id, BIM_file)))
                os.chdir(str(bldg_id))

            workflow_app = self.workflow_apps['DL']

            if BIM_file is not None:
                workflow_app.defaults['filenameDL'] = BIM_file
                #for input_var in workflow_app.inputs:
                #    if input_var['id'] == 'filenameDL':
                #        input_var['default'] = BIM_file

            command_list = self.workflow_apps['DL'].get_command_list(
                app_path=self.app_dir_local)

            if copy_resources:
                command_list.append('--resource_dir')
                command_list.append(self.working_dir)

            command = create_command(command_list)

            log_msg('Damage and loss assessment command:',
                    prepend_timestamp=False)
            log_msg('\n{}\n'.format(command), prepend_timestamp=False,
                    prepend_blank_space=False)

            result, returncode = run_command(command)

            log_msg(result, prepend_timestamp=False)

            # if multiple buildings are analyzed, copy the pelicun_log file to the root dir
            if 'Building' in self.app_type_list:

                try:
                    shutil.copy(
                        src = self.run_dir / f'{bldg_id}/{"pelicun_log.txt"}',
                        dst = self.run_dir / f'pelicun_log_{bldg_id}.txt')
                    #src = posixpath.join(self.run_dir, '{}/{}'.format(bldg_id, 'pelicun_log.txt')),
                    #dst = posixpath.join(self.run_dir, 'pelicun_log_{}.txt'.format(bldg_id)))
                except:
                    pass

            log_msg('Damage and loss assessment finished successfully.',
                    prepend_timestamp=False)
            log_div()

        else:
            log_msg('No DL requested, loss assessment step is skipped.')

            # Only regional simulations send in a bldg id
            if bldg_id != None:

                EDP_df = pd.read_csv('response.csv', header=0, index_col=0)

                col_info = []
                for col in EDP_df.columns:
                    try:
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
                    names=['type', 'loc', 'dir', 'stat'])

                df_res = pd.DataFrame(columns=MI, index=[0, ])
                if ('PID', '0') in df_res.columns:
                    del df_res[('PID', '0')]

                # store the EDP statistics in the output DF
                for col in np.transpose(col_info):
                    df_res.loc[0, (col[0], col[1], col[2], 'median')] = EDP_df[
                        '1-{}-{}-{}'.format(col[0], col[1], col[2])].median()
                    df_res.loc[0, (col[0], col[1], col[2], 'beta')] = np.log(
                        EDP_df['1-{}-{}-{}'.format(col[0], col[1], col[2])]).std()

                df_res.dropna(axis=1, how='all', inplace=True)

                df_res = df_res.astype(float)

                # save the output
                df_res.to_csv('EDP.csv')

            log_div()

    def aggregate_results(self, bldg_data):
        """
        Short description

        Longer description

        Parameters
        ----------
        """

        log_msg('Collecting damage and loss results')

        os.chdir(self.run_dir)

        min_id = int(bldg_data[0]['id'])
        max_id = int(bldg_data[0]['id'])

        out_types = ['BIM', 'EDP', 'DM', 'DV', 'every_realization']

        headers = dict(
            BIM = [0, ],
            EDP = [0, 1, 2, 3],
            DM = [0, 1, 2],
            DV = [0, 1, 2, 3])

        for out_type in out_types:
            if ((self.output_types is None) or
                (self.output_types.get(out_type, False))):

                if out_type == 'every_realization':

                        realizations_EDP = None
                        realizations_DL = None

                        for bldg in bldg_data:
                            bldg_id = bldg['id']
                            min_id = min(int(bldg_id), min_id)
                            max_id = max(int(bldg_id), max_id)

                            # save all EDP realizations

                            df_i = pd.read_csv(bldg_id+'/response.csv', header=0, index_col=0)

                            if realizations_EDP == None:
                                realizations_EDP = dict([(col, []) for col in df_i.columns])

                            for col in df_i.columns:
                                vals = df_i.loc[:,col].to_frame().T
                                vals.index = [bldg_id,]
                                realizations_EDP[col].append(vals)

                            # If damage and loss assessment is part of the workflow
                            # then save the DL outputs too
                            if 'DL' in self.workflow_apps.keys():

                                try:
                                #if True:
                                    df_i = pd.read_csv(bldg_id+f'/DL_summary.csv',
                                                       header=0, index_col=0)

                                    if realizations_DL == None:
                                        realizations_DL = dict([(col, []) for col in df_i.columns])

                                    for col in df_i.columns:
                                        vals = df_i.loc[:,col].to_frame().T
                                        vals.index = [bldg_id,]
                                        realizations_DL[col].append(vals)

                                except:
                                    log_msg(f'Error reading DL realization data for building {bldg_id}',
                                            prepend_timestamp=False)

                        for d_type in realizations_EDP.keys():
                            d_agg = pd.concat(realizations_EDP[d_type], axis=0, sort=False)

                            d_agg.to_hdf(f'realizations_{min_id}-{max_id}.hdf', f'EDP-{d_type}', mode='a', format='fixed')

                        if 'DL' in self.workflow_apps.keys():
                            for d_type in realizations_DL.keys():
                                d_agg = pd.concat(realizations_DL[d_type], axis=0, sort=False)
                                #d_agg.sort_index(axis=0, inplace=True)

                                d_agg.to_hdf(f'realizations_{min_id}-{max_id}.hdf', f'DL-{d_type}', mode='a', format='fixed')



                else:
                    out_list = []

                    for bldg in bldg_data:
                        bldg_id = bldg['id']
                        min_id = min(int(bldg_id), min_id)
                        max_id = max(int(bldg_id), max_id)

                        try:
                        #if True:
                            # EDP data
                            df_i = pd.read_csv(bldg_id+f'/{out_type}.csv',
                                               header=headers[out_type], index_col=0)
                            df_i.index = [bldg_id,]
                            out_list.append(df_i)

                        except:
                            log_msg(f'Error reading {out_type} data for building {bldg_id}',
                                    prepend_timestamp=False)

                    #out_agg = pd.concat(out_list, axis=0, sort=False)
                    out_agg = pd.DataFrame() if len(out_list) < 1 else pd.concat(out_list, axis=0, sort=False)
                    #out_agg.sort_index(axis=0, inplace=True)

                    # save the collected DataFrames as csv files
                    out_agg.to_csv(f'{out_type}_{min_id}-{max_id}.csv')

        log_msg('Damage and loss results collected successfully.',
                prepend_timestamp=False)
        log_div()