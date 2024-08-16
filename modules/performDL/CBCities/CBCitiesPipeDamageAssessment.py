#  # noqa: INP001, D100
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
#
# Modified 'cb-cities' code provided by the Soga Research Group UC Berkeley
# Dr. Stevan Gavrilovic

import argparse
import json
import os
import posixpath

import numpy as np
import pandas as pd
from CBCitiesMethods import *  # noqa: F403


def main(node_info, pipe_info):  # noqa: D103
    # Load Data

    print('Loading the node json file...')  # noqa: T201

    with open(node_info) as f:  # noqa: PTH123
        node_data = json.load(f)  # noqa: F841

    with open(pipe_info) as f:  # noqa: PTH123
        pipe_data = json.load(f)

    min_id = int(pipe_data[0]['id'])
    max_id = int(pipe_data[0]['id'])

    allPipes = []  # noqa: N806

    for pipe in pipe_data:
        AIM_file = pipe['file']  # noqa: N806

        asst_id = pipe['id']

        min_id = min(int(asst_id), min_id)
        max_id = max(int(asst_id), max_id)

        # Open the AIM file
        with open(AIM_file) as f:  # noqa: PTH123
            pipe = AIM_data = json.load(f)  # noqa: N806, F841, PLW2901

        allPipes.append(pipe)

    # read pgv for nodes
    #    pgv_csv_files = glob('../data/rupture/rupture62_im/*.csv')

    # Mapping & Saving
    import multiprocessing as mp

    pool = mp.Pool(mp.cpu_count() - 1)
    results = pool.map(add_failrate2pipe, [pipe for pipe in allPipes])  # noqa: C416, F405
    pool.close()

    df = pd.DataFrame({'DV': {}, 'MeanFailureProbability': {}})  # noqa: PD901

    for pipe in results:
        failureProbArray = pipe['fail_prob']  # noqa: N806
        avgFailureProb = np.average(failureProbArray)  # noqa: N806
        pipe_id = pipe['GeneralInformation']['AIM_id']

        print('pipe_id: ', pipe_id)  # noqa: T201
        #        print("failureProbArray: ",failureProbArray)
        print('avgFailureProb: ', avgFailureProb)  # noqa: T201

        df2 = pd.DataFrame(
            {'DV': pipe_id, 'MeanFailureProbability': avgFailureProb}, index=[0]
        )
        df = pd.concat([df, df2], axis=0)  # noqa: PD901

    # Get the directory for saving the results, assume it is the same one with the AIM file
    aimDir = os.path.dirname(pipe_info)  # noqa: PTH120, N806
    aimFileName = os.path.basename(pipe_info)  # noqa: PTH119, N806, F841

    saveDir = posixpath.join(aimDir, f'DV_{min_id}-{max_id}.csv')  # noqa: N806

    df.to_csv(saveDir, index=False)

    return 0
    # failed_pipes = fail_pipes_number(pipe)


if __name__ == '__main__':
    # Defining the command line arguments
    workflowArgParser = argparse.ArgumentParser(  # noqa: N816
        'Run the CB-cities water distribution damage and loss workflow.',
        allow_abbrev=False,
    )

    workflowArgParser.add_argument(
        '-n', '--nodeInfo', default=None, help='Node information.'
    )
    workflowArgParser.add_argument(
        '-p', '--pipeInfo', default=None, help='Pipe Information.'
    )
    workflowArgParser.add_argument(
        '-s', '--save_dir', default=None, help='Directory where to save the results.'
    )

    # Parsing the command line arguments
    wfArgs = workflowArgParser.parse_args()  # noqa: N816

    # update the local app dir with the default - if needed
    #    if wfArgs.appDir is None:
    #        workflow_dir = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
    #        wfArgs.appDir = workflow_dir.parents[1]

    # Calling the main workflow method and passing the parsed arguments
    main(node_info=wfArgs.nodeInfo, pipe_info=wfArgs.pipeInfo)
