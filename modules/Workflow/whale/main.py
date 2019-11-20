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

# import functions for Python 2.X support
from __future__ import division, print_function
import sys
if sys.version.startswith('2'): 
    range=xrange
    string_types = basestring
else:
    string_types = str

from time import gmtime, strftime
import json
import pprint
import posixpath
import ntpath
import os
import shutil
import importlib
from copy import deepcopy
import subprocess
import warnings
import pandas as pd

pp = pprint.PrettyPrinter(indent=4)

log_file = None

log_div = '-' * (80-21)  # 21 to have a total length of 80 with the time added

# get the absolute path of the whale directory
whale_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Monkeypatch warnings to get prettier messages
def _warning(message, category, filename, lineno, file=None, line=None):
    if '\\' in filename:
        file_path = filename.split('\\')
    elif '/' in filename:
        file_path = filename.split('/')
    python_file = '/'.join(file_path[-3:])
    print('WARNING in {} at line {}\n{}\n'.format(python_file, lineno, message))

warnings.showwarning = _warning

def log_msg(msg, prepend_timestamp=True):
    """
    Print a message to the screen with the current time as prefix

    The time is in ISO-8601 format, e.g. 2018-06-16T20:24:04Z

    Parameters
    ----------
    msg: string
       Message to print.

    """
    if prepend_timestamp:
        formatted_msg = '{} {}'.format(strftime('%Y-%m-%dT%H:%M:%SZ', gmtime()), msg)
    else:
        formatted_msg = msg

    print(formatted_msg)

    global log_file
    if log_file is not None:
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

    log_msg(log_div)
    log_msg(''*(80-21-6) + ' ERROR')
    log_msg(msg)
    log_msg(log_div)

def create_command(command_list, run_type):
    """
    Short description

    Long description

    Parameters
    ----------
    command_list: array of unicode strings
        Explain...
    """
    if command_list[0] == 'python':

        if run_type != 'set_up':
            # replace python with the full path to the python interpreter
            python_exe = sys.executable
        else:
            python_exe = 'python'

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
        
        #return result.decode(sys.stdout.encoding), returncode
        print(result, returncode)
        return str(result), returncode

def show_warning(warning_msg):
    warnings.warn(UserWarning(warning_msg))

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

    def set_pref(self, preferences):
        """
        Short description

        Parameters
        ----------
        preferences: dictionary
            Explain...
        """
        self.pref = preferences

    def get_command_list(self, app_path):
        """
        Short description

        Parameters
        ----------
        app_path: string
            Explain...
        """

        abs_path = posixpath.join(app_path, self.rel_path)

        arg_list = []

        if abs_path.endswith('.py'):
            arg_list.append('python')

        arg_list.append(u'{}'.format(abs_path))

        for in_arg in self.inputs:
            arg_list.append(u'--{}'.format(in_arg['id']))
            if in_arg['id'] in self.pref.keys():
                arg_list.append(u'{}'.format(self.pref[in_arg['id']]))
            else:
                arg_list.append(u'{}'.format(in_arg['default']))

        for out_arg in self.outputs:
            out_id = u'--{}'.format(out_arg['id'])
            if out_id not in arg_list:
                arg_list.append(out_id)
                if out_arg['id'] in self.pref.keys():
                    arg_list.append(u'{}'.format(self.pref[out_arg['id']]))
                else:
                    arg_list.append(u'{}'.format(out_arg['default']))

        for pref_name, pref_value in self.pref.items():
            pref_id = u'--{}'.format(pref_name)
            if pref_id not in arg_list:
                arg_list.append(pref_id)
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

    def __init__(self, run_type, input_file, app_registry, app_type_list):

        log_msg('Inputs provided:')
        log_msg('\tworkflow input file: {}'.format(input_file))
        log_msg('\tapplication registry file: {}'.format(app_registry))
        log_msg('\trun type: {}'.format(run_type))
        log_msg(log_div)

        self.run_type = run_type
        self.input_file = input_file
        self.app_registry_file = app_registry
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
        log_msg('\tLoading the json file...')
        with open(self.app_registry_file, 'r') as f:
            app_registry_data = json.load(f)
        log_msg('\tOK')

        # initialize the app registry
        self._init_app_registry()

        log_msg('\tCollecting application data...')
        # for each application type
        for app_type in sorted(self.app_registry.keys()):            

            # if the registry contains information about it
            app_type_long = app_type+'Applications'
            if app_type_long in app_registry_data:

                # get the list of available applications
                available_apps = app_registry_data[app_type_long]['Applications']
                api_info = app_registry_data[app_type_long]['API']

                # and store their name and executable location
                for app in available_apps:                    
                    self.app_registry[app_type][app['Name']] = WorkflowApplication(
                         app_type=app_type, app_info=app, api_info=api_info)

        log_msg('\tOK')

        log_msg('\tAvailable applications:')

        for app_type, app_list in self.app_registry.items():
            for app_name, app_object in app_list.items():
                log_msg('\t\t{} : {}'.format(app_type, app_name))

        #pp.pprint(self.app_registry)

        log_msg('Successfully parsed application registry')
        log_msg(log_div)

    def _parse_inputs(self):
        """
        Load the information about the workflow to run

        """

        log_msg('Parsing workflow input file')

        # open input file
        log_msg('\tLoading the json file...')
        with open(self.input_file, 'r') as f:
            input_data = json.load(f)
        log_msg('\tOK')

        # parse the location of the run_dir
        if 'runDir' in input_data:
            self.run_dir = input_data['runDir']
        else:
            raise WorkFlowInputError('Need a runDir entry in the input file')

        # parse the location(s) of the applications directory
        if 'localAppDir' in input_data:
            self.app_dir_local = input_data['localAppDir']
        else:
            raise WorkFlowInputError('Need a localAppDir entry in the input file')

        if 'remoteAppDir' in input_data:
            self.app_dir_remote = input_data['remoteAppDir']
        else:
            self.app_dir_remote = self.app_dir_local
            show_warning('remoteAppDir not specified. Using the value provided '
                'for localAppDir instead. This will lead to problems if you '
                'want to run a simulation remotely.')
            #raise WorkFlowInputError('Need a remoteAppDir entry in the input file')

        for loc_name, loc_val in zip(
            ['Run dir', 'Local applications dir','Remote applications dir'], 
            [self.run_dir, self.app_dir_remote, self.app_dir_local]):
            log_msg('\t{} location: {}'.format(loc_name, loc_val))

        if 'Building' in self.app_type_list:
            if 'buildingFile' in input_data:
                self.building_file_name = input_data['buildingFile']
            else:
                self.building_file_name = "buildings.json"
            log_msg('\tbuilding file name: {}'.format(self.building_file_name))


        # get the list of requested applications
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
                    if eventClassification in ['Earthquake', 'Wind'] :

                        app_object = deepcopy(
                            self.app_registry['Event'].get(event['Application']))

                        if app_object is None:
                            raise WorkFlowInputError(
                                'Application entry missing for {}'.format('Events'))

                        app_object.set_pref(event['ApplicationData'])
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

                    app_object.set_pref(requested_apps[app_type]['ApplicationData'])
                    self.workflow_apps[app_type] = app_object
              
                else:
                    if app_type != "Modeling":
                        raise WorkFlowInputError(
                            'Need {} entry in Applications'.format(app_type))
                    else:                        
                        self.app_registry.pop("Modeling", None)
                        log_msg('\tNo Modeling among requested applications.')

        if "Modeling" not in self.app_registry:
            self.app_type_list.remove("Modeling")

        log_msg('\tRequested workflow:')
        for app_type, app_object in self.workflow_apps.items():
            log_msg('\t\t{} : {}'.format(app_type, app_object.name))

        log_msg('Successfully parsed workflow inputs')
        log_msg(log_div)

    def create_building_files(self):
        """
        Short description

        Longer description

        Parameters
        ----------

        """

        log_msg('Creating files for individual buildings')

        building_file = posixpath.join(self.run_dir, self.building_file_name)

        bldg_app = self.workflow_apps['Building']

        # TODO: not elegant code, fix later
        os.chdir(self.run_dir)

        building_file = building_file.replace('.json', 
            '{}-{}.json'.format(bldg_app.pref['Min'], bldg_app.pref['Max'])) 
        self.building_file_path = building_file

        for output in bldg_app.outputs:
            if output['id'] == 'buildingFile':
                output['default'] = building_file

        bldg_command_list = bldg_app.get_command_list(
            app_path = self.app_dir_local)

        bldg_command_list.append(u'--getRV')

        command = create_command(bldg_command_list, self.run_type)        

        log_msg('Creating initial building files...')
        log_msg('\n{}\n'.format(command), prepend_timestamp=False)
        
        result, returncode = run_command(command)

        log_msg('\tOutput: ')
        log_msg('\n{}\n'.format(result), prepend_timestamp=False)

        log_msg('Building files successfully created.')
        log_msg(log_div)

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
                src = posixpath.join(self.run_dir, BIM_file),
                dst = posixpath.join(
                    self.run_dir, 
                    '{}/templatedir/{}'.format(bldg_id, BIM_file)))

        else:
            os.chdir('templatedir') #TODO: we might want to add a generic id dir to be consistent with the regional workflow here

            # Make a copy of the input file and rename it to BIM.json
            # This is a temporary fix, will be removed eventually.            
            dst = posixpath.join(os.getcwd(),BIM_file)
            if BIM_file != self.input_file:
                shutil.copy(src = self.input_file, dst = dst)

        log_msg('Simulation directory successfully initialized.')
        log_msg(log_div)

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

        log_msg('Simulation directory successfully cleaned up.')
        log_msg(log_div)

    def init_workdir(self):
        """
        Short description

        Longer description

        Parameters
        ----------

        """
        log_msg('Initializing the working directory.')

        shutil.rmtree(self.run_dir)

        os.mkdir(self.run_dir)

        log_msg('Working directory successfully initialized.')
        log_msg(log_div)

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
            if os.path.isdir(os.path.join(self.run_dir, file_or_dir)):
                shutil.rmtree(file_or_dir, ignore_errors=True)

        log_msg('Working directory successfully cleaned up.')
        log_msg(log_div)


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

        if (("Modeling" in app_sequence) and
            ("Modeling" not in self.workflow_apps.keys())):
            app_sequence.remove("Modeling")

        for app_type in app_sequence:

            workflow_app = self.workflow_apps[app_type]

            # TODO: not elegant code, fix later
            if BIM_file is not None:
                for input_var in workflow_app.inputs:
                    if input_var['id'] == 'filenameBIM':
                        input_var['default'] = BIM_file

            command_list = workflow_app.get_command_list(
                app_path = self.app_dir_local)

            command_list.append(u'--getRV')

            command = create_command(command_list, self.run_type)
            
            log_msg('\tRunning {} app for RV...'.format(app_type))
            log_msg('\n{}\n'.format(command), prepend_timestamp=False)
            
            result, returncode = run_command(command)

            log_msg('\tOutput: ')
            log_msg('\n{}\n'.format(result), prepend_timestamp=False)

        log_msg('Files with random variables successfully created.')
        log_msg(log_div)


    def create_driver_file(self, app_sequence, bldg_id=None):
        """
        Short description

        Longer description

        Parameters
        ----------
        """

        log_msg('Creating the workflow driver file')

        os.chdir(self.run_dir)

        if bldg_id is not None:
            os.chdir(bldg_id)       

        os.chdir('templatedir')

        driver_script = u''

        if (("Modeling" in app_sequence) and
            ("Modeling" not in self.workflow_apps.keys())):
            app_sequence.remove("Modeling")

        for app_type in app_sequence:
            command_list = self.workflow_apps[app_type].get_command_list(
                app_path = self.app_dir_remote)

            driver_script += create_command(command_list, self.run_type) + u'\n'

        log_msg('Workflow driver script:')
        log_msg('\n{}\n'.format(driver_script), prepend_timestamp=False)

        with open('driver','w') as f:
            f.write(driver_script)

        log_msg('Workflow driver file successfully created.')
        log_msg(log_div)

    def simulate_response(self, BIM_file = 'BIM.json', bldg_id=None):
        """
        Short description

        Longer description

        Parameters
        ----------
        """

        log_msg('Running response simulation')

        os.chdir(self.run_dir)

        if bldg_id is not None:
            os.chdir(bldg_id)       

        os.chdir('templatedir')

        workflow_app = self.workflow_apps['UQ']

        # TODO: not elegant code, fix later
        if BIM_file is not None:
            for input_var in workflow_app.inputs:
                if input_var['id'] == 'filenameBIM':
                    input_var['default'] = BIM_file

        command_list = workflow_app.get_command_list(
            app_path=self.app_dir_local)

        # add the run type to the uq command list
        command_list.append(u'--runType')
        command_list.append(u'{}'.format(self.run_type))

        command = create_command(command_list, self.run_type)

        log_msg('\tSimulation command:')
        log_msg('\n{}\n'.format(command), prepend_timestamp=False)

        result, returncode = run_command(command)

        if self.run_type == 'run':
            log_msg('Response simulation finished successfully.')
        elif self.run_type == 'set_up':
            log_msg('Response simulation set up successfully')
        log_msg(log_div)

    def estimate_losses(self, BIM_file = 'BIM.json', bldg_id = None, input_file = None):
        """
        Short description

        Longer description

        Parameters
        ----------
        """

        log_msg('Running damage and loss assessment')

        os.chdir(self.run_dir)

        if 'Building' not in self.app_type_list:
            # Copy the dakota.json file from the templatedir to the run_dir so that
            # all the required inputs are in one place.
            input_file = ntpath.basename(input_file)
            shutil.copy(
                src = posixpath.join(self.run_dir,'templatedir/{}'.format(input_file)),
                dst = posixpath.join(self.run_dir,BIM_file))
        else:            
            # copy the BIM file from the main dir to the building dir
            shutil.copy(
                src = posixpath.join(self.run_dir, BIM_file),
                dst = posixpath.join(self.run_dir, 
                                     '{}/{}'.format(bldg_id, BIM_file)))
            os.chdir(str(bldg_id))

        workflow_app = self.workflow_apps['DL']

        # TODO: not elegant code, fix later
        if BIM_file is not None:
            for input_var in workflow_app.inputs:
                if input_var['id'] == 'filenameDL':
                    input_var['default'] = BIM_file     

        command_list = self.workflow_apps['DL'].get_command_list(
            app_path=self.app_dir_local)     

        command = create_command(command_list, self.run_type)

        log_msg('\tDamage and loss assessment command:')
        log_msg('\n{}\n'.format(command), prepend_timestamp=False)

        result, returncode = run_command(command)

        log_msg(result, prepend_timestamp=False)

        log_msg('Damage and loss assessment finished successfully.')
        log_msg(log_div)

    def aggregate_dmg_and_loss(self, bldg_data):
        """
        Short description

        Longer description

        Parameters
        ----------
        """

        log_msg('Collecting damage and loss results')

        os.chdir(self.run_dir)

        # start with the damage data
        DM_agg = pd.DataFrame()

        min_id = int(bldg_data[0]['id'])
        max_id = int(bldg_data[0]['id'])
        for bldg in bldg_data:
            bldg_id = bldg['id']
            min_id = min(int(bldg_id), min_id)
            max_id = max(int(bldg_id), max_id)
            
            try:
            with open(bldg_id+'/DM.json') as f:
                DM = json.load(f)            
                
            for FG in DM.keys():

                if FG == 'aggregate':
                    PG = ''
                    DS_list = list(DM[FG].keys())
                else:
                    PG = next(iter(DM[FG]))
                    DS_list = list(DM[FG][PG].keys())
                
                if ((DM_agg.size == 0) or 
                    (FG not in DM_agg.columns.get_level_values('FG'))):
                    MI = pd.MultiIndex.from_product([[FG,],DS_list],names=['FG','DS'])
                    DM_add = pd.DataFrame(columns=MI, index=[bldg_id])
                    
                    for DS in DS_list:
                        if PG == '':
                            val = DM[FG][DS]
                        else:
                            val = DM[FG][PG][DS]
                        DM_add.loc[bldg_id, (FG, DS)] = val
                        
                    DM_agg = pd.concat([DM_agg, DM_add], axis=1, sort=False)
                
                else:        
                    for DS in DS_list:
                        if PG == '':
                            val = DM[FG][DS]
                        else:
                            val = DM[FG][PG][DS]
                        DM_agg.loc[bldg_id, (FG, DS)] = val
            except:
                log_msg('Error reading DM data for building {}'.format(bldg_id))

        # then collect the decision variables
        DV_agg = pd.DataFrame()

        for bldg in bldg_data:
            bldg_id = bldg['id']
            
            try:
            
            with open(bldg_id+'/DV.json') as f:
                DV = json.load(f)
                
            for DV_type in DV.keys():
                
                stat_list = list(DV[DV_type]['total'].keys())
                
                if ((DV_agg.size == 0) or 
                    (DV_type not in DV_agg.columns.get_level_values('DV'))): 
                
                    MI = pd.MultiIndex.from_product(
                        [[DV_type,],stat_list],names=['DV','stat'])
                
                    DV_add = pd.DataFrame(columns=MI, index=[bldg_id])
                    
                    for stat in stat_list:
                        DV_add.loc[bldg_id, (DV_type, stat)] = DV[DV_type]['total'][stat]
                        
                    DV_agg = pd.concat([DV_agg, DV_add], axis=1, sort=False)
                else:                     
                    for stat in stat_list:
                        DV_agg.loc[bldg_id, (DV_type, stat)] = DV[DV_type]['total'][stat]

            except:
                log_msg('Error reading DM data for building {}'.format(bldg_id))

        # save the collected DataFrames as csv files
        DM_agg.to_csv('DM_{}-{}.csv'.format(min_id, max_id))
        DV_agg.to_csv('DV_{}-{}.csv'.format(min_id, max_id))

        log_msg('Damage and loss results collected successfully.')
        log_msg(log_div)


        











