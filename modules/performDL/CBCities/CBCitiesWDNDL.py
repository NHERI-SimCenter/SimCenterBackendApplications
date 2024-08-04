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
import posixpath
import sys
from time import gmtime, strftime

import numpy as np
import pandas as pd


def log_msg(msg):  # noqa: ANN001, ANN201, D103
    formatted_msg = '{} {}'.format(strftime('%Y-%m-%dT%H:%M:%SZ', gmtime()), msg)

    print(formatted_msg)  # noqa: T201


from CBCitiesMethods import *  # noqa: E402, F403


def run_DL_calc(aim_file_path, saveDir, output_name):  # noqa: ANN001, ANN201, N802, N803, D103
    # Load Data

    print('Loading the pipeline json file...')  # noqa: T201

    # Open the AIM file
    with open(aim_file_path) as f:  # noqa: PLW1514, PTH123
        pipe = AIM_data = json.load(f)  # noqa: N806, F841

    add_failrate2pipe(pipe)  # noqa: F405

    failureRateArray = pipe['fail_prob']  # noqa: N806
    avgRr = np.average(failureRateArray)  # noqa: N806

    df = pd.DataFrame({'DV': '0', 'RepairRate': avgRr}, index=[0])  # noqa: PD901

    savePath = posixpath.join(saveDir, output_name)  # noqa: N806

    df.to_csv(savePath, index=False)

    return 0


def main(args):  # noqa: ANN001, ANN201, D103
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--filenameDL')
    parser.add_argument('-p', '--demandFile', default=None)
    parser.add_argument('--outputEDP', default='EDP.csv')
    parser.add_argument('--outputDM', default='DM.csv')
    parser.add_argument('--outputDV', default='DV.csv')
    parser.add_argument('--resource_dir', default=None)
    parser.add_argument('--dirnameOutput', default=None)

    args = parser.parse_args(args)

    log_msg('Initializing CB-Cities calculation...')

    out = run_DL_calc(
        aim_file_path=args.filenameDL,
        saveDir=args.dirnameOutput,
        output_name=args.outputDV,
    )

    if out == -1:
        log_msg('DL calculation failed.')
    else:
        log_msg('DL calculation completed.')


if __name__ == '__main__':
    main(sys.argv[1:])
