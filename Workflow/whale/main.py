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
import os
import shutil
from copy import deepcopy
import subprocess

pp = pprint.PrettyPrinter(indent=4)

log_div = '-' * (80-21)  # 21 to have a total length of 80 with the time added

def log_msg(msg):
    """
    Print a message to the screen with the current time as prefix

    The time is in ISO-8601 format, e.g. 2018-06-16T20:24:04Z

    Parameters
    ----------
    msg: string
       Message to print.

    """
 
    print('{} {}'.format(strftime('%Y-%m-%dT%H:%M:%SZ', gmtime()), msg))

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

def create_command(command_list):
    """
    Short description

    Long description

    Parameters
    ----------
    command_list: array of unicode strings
        Explain...
    """
    if command_list[0] == 'python':
        command = 'python "{}" '.format(command_list[1]) + ' '.join(command_list[2:])
    else:
        command = '"{}" '.format(command_list[0]) + ' '.join(command_list[1:])

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

    try:
        result = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
        returncode = 0
    except subprocess.CalledProcessError as e:
        result = e.output
        returncode = e.returncode

    if returncode != 0:
        log_error('return code: {}'.format(returncode))
        
    return result.decode(sys.stdout.encoding), returncode

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
            arg_list.append(u'{}'.format(in_arg['default']))

        for out_arg in self.outputs:
            out_id = u'--{}'.format(out_arg['id'])
            if out_id not in arg_list:
                arg_list.append(out_id)
                arg_list.append(u'{}'.format(out_arg['default']))

        for in_name, in_value in self.pref.items():
            arg_list.append(u'--{}'.format(in_name))
            arg_list.append(u'{}'.format(in_value))

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
        if 'remoteAppDir' in input_data:
            self.app_dir_remote = input_data['remoteAppDir']
        else:
            raise WorkFlowInputError('Need a remoteAppDir entry in the input file')

        if 'localAppDir' in input_data:
            self.app_dir_local = input_data['localAppDir']
        else:
            raise WorkFlowInputError('Need a localAppDir entry in the input file')

        for loc_name, loc_val in zip(
            ['Run dir', 'Local applications dir','Remote applications dir'], 
            [self.run_dir, self.app_dir_remote, self.app_dir_local]):
            log_msg('\t{} location: {}'.format(loc_name, loc_val))

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
                    if eventClassification == 'Earthquake':

                        app_object = deepcopy(
                            self.app_registry['Event'].get(event['Application']))

                        if app_object is None:
                            raise WorkFlowInputError(
                                'Application entry missing for {}'.format('Events'))

                        app_object.set_pref(event['ApplicationData'])
                        self.workflow_apps['Event'] = app_object
                      
                    else: 
                        raise WorkFlowInputError(
                            ('Currently, only earthquake events are supported. '
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
                    raise WorkFlowInputError(
                        'Need {} entry in Applications'.format(app_type))

        log_msg('\tRequested workflow:')
        for app_type, app_object in self.workflow_apps.items():
            log_msg('\t\t{} : {}'.format(app_type, app_object.name))

        log_msg('Successfully parsed workflow inputs')
        log_msg(log_div)

    def create_RV_files(self, app_sequence): # we will probably need to rename this one
        """
        Short description

        Longer description

        Parameters
        ----------

        """

        log_msg('Creating files with random variables')

        os.chdir(self.run_dir)
        os.chdir('templatedir')

        # Make a copy of the input file and rename it to BIM.json
        # This is a temporary fix, will be removed eventually.
        shutil.copy(
            src = self.input_file,
            dst = posixpath.join(self.run_dir,'templatedir/BIM.json')) 

        for app_type in app_sequence:
            command_list = self.workflow_apps[app_type].get_command_list(
                app_path = self.app_dir_local)

            command_list.append(u'--getRV')

            command = create_command(command_list)
            
            log_msg('\tRunning {} app for RV...'.format(app_type))
            print('\n{}\n'.format(command))
            
            result, returncode = run_command(command)

            log_msg('\tOutput: ')
            print('\n{}\n'.format(result))

        log_msg('Files with random variables successfully created.')
        log_msg(log_div)


    def create_driver_file(self, app_sequence):
        """
        Short description

        Longer description

        Parameters
        ----------
        """

        log_msg('Creating the workflow driver file')

        driver_script = u''

        for app_type in app_sequence:
            command_list = self.workflow_apps[app_type].get_command_list(
                app_path = self.app_dir_remote)

            driver_script += create_command(command_list) + u'\n'

        os.chdir(self.run_dir)
        os.chdir('templatedir')

        log_msg('Workflow driver script:')
        print('\n{}\n'.format(driver_script))

        with open('driver','w') as f:
            f.write(driver_script)

        log_msg('Workflow driver file successfully created.')
        log_msg(log_div)

    def simulate_response(self):
        """
        Short description

        Longer description

        Parameters
        ----------
        """

        log_msg('Running response simulation')

        os.chdir(self.run_dir)
        os.chdir('templatedir')

        command_list = self.workflow_apps['UQ'].get_command_list(
            app_path=self.app_dir_local)

        # add the run type to the uq command list
        command_list.append(u'--runType')
        command_list.append(u'{}'.format(self.run_type))

        command = create_command(command_list)

        log_msg('\tSimulation command:')
        print('\n{}\n'.format(command))

        result, returncode = run_command(command)

        if self.run_type == 'run':
            log_msg('Response simulation finished successfully.')
        elif self.run_type == 'set_up':
            log_msg('Response simulation set up successfully')
        log_msg(log_div)

    def estimate_losses(self):
        """
        Short description

        Longer description

        Parameters
        ----------
        """

        log_msg('Running damage and loss assessment')

        os.chdir(self.run_dir)

        # Copy the dakota.json file from the templatedir to the run_dir so that
        # all the required inputs are in one place.
        shutil.copy(
            src = posixpath.join(self.run_dir,'templatedir/dakota.json'),
            dst = posixpath.join(self.run_dir,'dakota.json'))

        command_list = self.workflow_apps['DL'].get_command_list(
            app_path=self.app_dir_local)

        command = create_command(command_list)

        log_msg('\tDamage and loss assessment command:')
        print('\n{}\n'.format(command))

        result, returncode = run_command(command)

        print(result)

        log_msg('Damage and loss assessment finished successfully.')




        











