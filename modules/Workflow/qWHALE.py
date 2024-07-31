# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 The Regents of the University of California
# Copyright (c) 2019 Leland Stanford Junior University
#
# This file is part of the SimCenter Backend Applications.
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
# SimCenter Backend Applications. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Frank McKenna
# Adam Zsarnóczay
# Wael Elhaddad
# Michael Gardner
# Chaofeng Wang

import sys, os, json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import whale.main as whale
from whale.main import log_msg, log_div


def main(run_type, input_file, app_registry, working_dir, app_dir, log_file):
    # initialize the log file
    with open(input_file, 'r') as f:
        inputs = json.load(f)
    runDir = inputs['runDir']

    if working_dir is not None:
        runDir = working_dir
    else:
        runDir = inputs['runDir']

    whale.log_file = runDir + '/log.txt'
    # initialize log file
    whale.set_options(
        {'LogFile': runDir + '/log.txt', 'LogShowMS': False, 'PrintLog': True}
    )
    log_msg(
        '\nqWHALE workflow\n', prepend_timestamp=False, prepend_blank_space=False
    )

    whale.print_system_info()

    # echo the inputs
    log_div(prepend_blank_space=False)
    log_div(prepend_blank_space=False)
    log_msg('Started running the workflow script')
    log_div()

    WF = whale.Workflow(
        run_type,
        input_file,
        app_registry,
        app_type_list=['FEM', 'UQ'],
        working_dir=working_dir,
        app_dir=app_dir,
    )

    # initialize the working directory
    WF.init_simdir()

    # prepare the input files for the simulation
    WF.preprocess_inputs(app_sequence=['FEM'])

    # run uq engine to simulate response
    WF.simulate_response()


if __name__ == '__main__':
    """
    if len(sys.argv) != 4:
        print('\nNeed three arguments, e.g.:\n')
        print('    python %s action workflowinputfile.json workflowapplications.json' % sys.argv[0])
        print('\nwhere: action is either check or run\n')
        exit(1)

    main(run_type=sys.argv[1], input_file=sys.argv[2], app_registry=sys.argv[3])
    """

    # Defining the command line arguments

    workflowArgParser = argparse.ArgumentParser(
        'Run the NHERI SimCenter sWHALE workflow for a single asset.',
        allow_abbrev=False,
    )

    workflowArgParser.add_argument(
        'runType', help='Specifies the type of run requested.'
    )
    workflowArgParser.add_argument(
        'inputFile', help='Specifies the input file for the workflow.'
    )
    workflowArgParser.add_argument(
        'registry',
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'WorkflowApplications.json'
        ),
        help='Path to file containing registered workflow applications',
    )
    workflowArgParser.add_argument(
        '-w',
        '--workDir',
        default=None,
        help='Absolute path to the working directory.',
    )
    workflowArgParser.add_argument(
        '-a',
        '--appDir',
        default=None,
        help='Absolute path to the local application directory.',
    )
    workflowArgParser.add_argument(
        '-l',
        '--logFile',
        default='log.txt',
        help='Path where the log file will be saved.',
    )

    # Parsing the command line arguments
    wfArgs = workflowArgParser.parse_args()

    # update the local app dir with the default - if needed
    if wfArgs.appDir is None:
        workflow_dir = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
        wfArgs.appDir = workflow_dir.parents[1]

    # Calling the main workflow method and passing the parsed arguments

    main(
        run_type=wfArgs.runType,
        input_file=wfArgs.inputFile,
        app_registry=wfArgs.registry,
        working_dir=wfArgs.workDir,
        app_dir=wfArgs.appDir,
        log_file=wfArgs.logFile,
    )
