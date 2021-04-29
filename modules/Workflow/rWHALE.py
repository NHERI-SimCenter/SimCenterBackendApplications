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

import sys, os, json
import argparse
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import whale.main as whale
from whale.main import log_msg, log_div

def main(run_type, input_file, app_registry,
         force_cleanup, bldg_id_filter, reference_dir,
         working_dir, app_dir, log_file):

    # initialize the log file
    with open(input_file, 'r') as f:
        inputs = json.load(f)
    if working_dir is not None:
        runDir = working_dir
    else:
        runDir = inputs['runDir']

    if not os.path.exists(runDir):
        os.mkdir(runDir)
    if log_file == 'log.txt':
        whale.log_file = runDir + '/log.txt'
    else:
        whale.log_file = log_file
    with open(whale.log_file, 'w') as f:
        f.write('RDT workflow\n')

    whale.print_system_info()

    # echo the inputs
    log_msg(log_div)
    log_msg('Started running the workflow script')
    log_msg(log_div)
    if force_cleanup:
        log_msg('Forced cleanup turned on.')

    WF = whale.Workflow(run_type, input_file, app_registry,
        app_type_list = ['Building', 'RegionalEvent', 'RegionalMapping',
                         'Event', 'Modeling', 'EDP', 'Simulation', 'UQ', 'DL'],
        reference_dir = reference_dir,
        working_dir = working_dir,
        app_dir = app_dir,
        units = inputs.get('units', None),
        outputs=inputs.get('outputs', None))

    if bldg_id_filter is not None:
        print(bldg_id_filter)
        log_msg(
            f'Overriding simulation scope; running buildings {bldg_id_filter}')

        # If a Min or Max attribute is used when calling the script, we need to
        # update the min and max values in the input file.
        WF.workflow_apps['Building'].pref["filter"] = bldg_id_filter

    # initialize the working directory
    WF.init_workdir()

    # prepare the basic inputs for individual buildings
    building_file = WF.create_building_files()
    WF.perform_regional_mapping(building_file)

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

        if force_cleanup:
            #clean up intermediate files from the simulation
            WF.cleanup_simdir(bldg['id'])

    # aggregate results
    WF.aggregate_results(bldg_data = bldg_data)

    if force_cleanup:
        # clean up intermediate files from the working directory
        WF.cleanup_workdir()

if __name__ == '__main__':

    #Defining the command line arguments

    workflowArgParser = argparse.ArgumentParser(
        "Run the NHERI SimCenter workflow for a set of assets.",
        allow_abbrev=False)

    workflowArgParser.add_argument("configuration",
        help="Configuration file specifying the applications and data to be "
             "used")
    workflowArgParser.add_argument("-F", "--filter",
        default=None,
        help="Provide a subset of building ids to run")
    workflowArgParser.add_argument("-c", "--check",
        help="Check the configuration file")
    workflowArgParser.add_argument("-r", "--registry",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "WorkflowApplications.json"),
        help="Path to file containing registered workflow applications")
    workflowArgParser.add_argument("-f", "--forceCleanup",
        action="store_true",
        help="Remove working directories after the simulation is completed.")
    workflowArgParser.add_argument("-d", "--referenceDir",
        default=os.path.join(os.getcwd(), 'input_data'),
        help="Relative paths in the config file are referenced to this directory.")
    workflowArgParser.add_argument("-w", "--workDir",
        default=os.path.join(os.getcwd(), 'results'),
        help="Absolute path to the working directory.")
    workflowArgParser.add_argument("-a", "--appDir",
        default=None,
        help="Absolute path to the local application directory.")
    workflowArgParser.add_argument("-l", "--logFile",
        default='log.txt',
        help="Path where the log file will be saved.")

    #Parsing the command line arguments
    wfArgs = workflowArgParser.parse_args()

    # update the local app dir with the default - if needed
    if wfArgs.appDir is None:
        workflow_dir = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
        wfArgs.appDir = workflow_dir.parents[1]

    if wfArgs.check:
        run_type = 'set_up'
    else:
        run_type = 'run'

    #Calling the main workflow method and passing the parsed arguments
    main(run_type = run_type,
         input_file = wfArgs.configuration,
         app_registry = wfArgs.registry,
         force_cleanup = wfArgs.forceCleanup,
         bldg_id_filter = wfArgs.filter,
         reference_dir = wfArgs.referenceDir,
         working_dir = wfArgs.workDir,
         app_dir = wfArgs.appDir,
         log_file = wfArgs.logFile)