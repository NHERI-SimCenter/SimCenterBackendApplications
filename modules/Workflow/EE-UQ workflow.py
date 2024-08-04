#  # noqa: INP001, D100
# Copyright (c) 2019 The Regents of the University of California
# Copyright (c) 2019 Leland Stanford Junior University
#
# This file is part of pelicun.
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
# pelicun. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Frank McKenna
# Adam Zsarnóczay
# Wael Elhaddad
# Michael Gardner
# Chaofeng Wang

# import functions for Python 2.X support
import json
import os
import sys

if sys.version.startswith('2'):
    range = xrange  # noqa: A001, F821
    string_types = basestring  # noqa: F821
else:
    string_types = str

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))  # noqa: PTH120

import whale.main as whale
from whale.main import log_div, log_msg


def main(run_type, input_file, app_registry):  # noqa: ANN001, ANN201, D103
    # initialize the log file
    with open(input_file) as f:  # noqa: PLW1514, PTH123
        inputs = json.load(f)
    runDir = inputs['runDir']  # noqa: N806

    whale.log_file = runDir + '/log.txt'
    with open(whale.log_file, 'w') as f:  # noqa: FURB103, PLW1514, PTH123
        f.write('EE-UQ workflow\n')

    # echo the inputs
    log_msg(log_div)
    log_msg('Started running the workflow script')
    log_msg(log_div)

    WF = whale.Workflow(  # noqa: N806
        run_type,
        input_file,
        app_registry,
        app_type_list=['Event', 'Modeling', 'EDP', 'Simulation', 'UQ'],
    )

    # initialize the working directory
    WF.init_simdir()

    # prepare the input files for the simulation
    WF.create_RV_files(app_sequence=['Event', 'Modeling', 'EDP', 'Simulation'])

    # create the workflow driver file
    WF.create_driver_file(app_sequence=['Event', 'Modeling', 'EDP', 'Simulation'])

    # run uq engine to simulate response
    WF.simulate_response()


if __name__ == '__main__':
    if len(sys.argv) != 4:  # noqa: PLR2004
        print('\nNeed three arguments, e.g.:\n')  # noqa: T201
        print(  # noqa: T201
            '    python %s action workflowinputfile.json workflowapplications.json'  # noqa: UP031
            % sys.argv[0]
        )
        print('\nwhere: action is either check or run\n')  # noqa: T201
        exit(1)  # noqa: PLR1722

    main(run_type=sys.argv[1], input_file=sys.argv[2], app_registry=sys.argv[3])
