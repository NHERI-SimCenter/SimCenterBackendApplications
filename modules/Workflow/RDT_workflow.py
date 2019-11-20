# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 The Regents of the University of California
# Copyright (c) 2019 Leland Stanford Junior University
#
# This file is part of the RDT Application.
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
# You should have received a copy of the BSD 3-Clause License along with the
# RDT Application. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Frank McKenna
# Adam Zsarnóczay
# Wael Elhaddad
# Michael Gardner
# Chaofeng Wang

# import functions for Python 2.X support
from __future__ import division, print_function
import sys, os, json
if sys.version.startswith('2'): 
    range=xrange
    string_types = basestring
else:
    string_types = str

import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import whale.main as whale
from whale.main import log_msg, log_div

def main(run_type, input_file, app_registry, 
         force_cleanup=False, bldg_id_range=[None, None]):

    # initialize the log file
    with open(input_file, 'r') as f:
        inputs = json.load(f)
    runDir = inputs['runDir']

    whale.log_file = runDir + '/log.txt'
    with open(whale.log_file, 'w') as f:
        f.write('RDT workflow\n') 

    # echo the inputs
    log_msg(log_div)
    log_msg('Started running the workflow script')
    log_msg(log_div)
    if force_cleanup:
        log_msg('Forced cleanup turned on.')

    WF = whale.Workflow(run_type, input_file, app_registry,
        app_type_list = ['Building', 'Event', 'Modeling', 'EDP', 'Simulation', 
                         'UQ', 'DL'])

    if bldg_id_range[0] is not None:
        print(bldg_id_range)
        log_msg(
            'Overriding simulation limits; running buildings {} - {}'.format(
                bldg_id_range[0], bldg_id_range[1]))

        # If a Min or Max attribute is used when calling the script, we need to 
        # update the min and max values in the input file.
        bldg_min, bldg_max = bldg_id_range
        WF.workflow_apps['Building'].pref['Min'] = bldg_min
        WF.workflow_apps['Building'].pref['Max'] = bldg_max

    # initialize the working directory
    WF.init_workdir()

    # prepare the basic inputs for individual buildings
    WF.create_building_files()

    # TODO: not elegant code, fix later
    with open(WF.building_file_path, 'r') as f:
        bldg_data = json.load(f)

    for bldg in bldg_data: #[:1]:
        log_msg(bldg)

        # initialize the simulation directory
        WF.init_simdir(bldg['id'], bldg['file'])

        # prepare the input files for the simulation
        WF.create_RV_files(
            app_sequence = ['Event', 'Modeling', 'EDP', 'Simulation'],
            BIM_file = bldg['file'], bldg_id=bldg['id'])

        # create the workflow driver file
        WF.create_driver_file(
            app_sequence = ['Building', 'Event', 'Modeling', 'EDP', 'Simulation'],
            bldg_id=bldg['id'])        

        # run uq engine to simulate response
        WF.simulate_response(BIM_file = bldg['file'], bldg_id=bldg['id'])

        # run dl engine to estimate losses
        WF.estimate_losses(BIM_file = bldg['file'], bldg_id = bldg['id'])

    # aggregate damage and loss results
    WF.aggregate_dmg_and_loss(bldg_data = bldg_data)

if __name__ == '__main__':

    #Defining the command line arguments
    workflowArgParser = argparse.ArgumentParser(
        "Run the NHERI SimCenter workflow for a set of buildings")

    workflowArgParser.add_argument(
        "configuration", 
        help="Configuration file specifying the applications and data to be "
             "used")
    workflowArgParser.add_argument(
        "-Min", "--Min", type=int, default=None, 
        help="Override the index of the first building")
    workflowArgParser.add_argument(
        "-Max", "--Max", type=int, default=None, 
        help="Override the index of the last building")
    workflowArgParser.add_argument(
        "-c", "--check", 
        help="Check the configuration file")
    workflowArgParser.add_argument(
        "-r", "--registry", default="WorkflowApplications.json", 
        help="Path to file containing registered workflow applications")
    workflowArgParser.add_argument(
        "-f", "--forceCleanup",  action="store_true", 
        help="Path to file containing registered workflow applications")

    #Parsing the command line arguments
    wfArgs = workflowArgParser.parse_args() 

    if wfArgs.check:
        run_type = 'set_up'
    else:   
        run_type = 'run'

    #Calling the main workflow method and passing the parsed arguments
    main(run_type = run_type, 
         input_file = wfArgs.configuration,
         app_registry = wfArgs.registry,
         force_cleanup = wfArgs.forceCleanup,
         bldg_id_range = [wfArgs.Min, wfArgs.Max])