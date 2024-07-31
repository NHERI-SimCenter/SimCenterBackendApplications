# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Leland Stanford Junior University
# Copyright (c) 2022 The Regents of the University of California
#
# This file is part of the SimCenter Backend Applications
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
# this file. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarnóczay
# Sangri Kuanshi

# Description:
# Read SAM info saved the surrogate model file (it was saved when training the surrogate model) and write it to new SAM.json
# Input files: AIM.json, surrogate.json (user provided)
# Output files: SAM.json

import sys, argparse, json, os


def create_SAM(AIM_file, SAM_file):
    #
    # Find SAM.json info from surrogate model file
    #

    # load AIM

    with open(AIM_file, 'r') as f:
        root_AIM = json.load(f)

    print('General Information tab is ignored')
    root_SAM = root_AIM['Applications']['Modeling']

    # find and load surrogate json

    # surrogate_path = os.path.join(root_SAM['ApplicationData']['MS_Path'],root_SAM['ApplicationData']['mainScript'])
    surrogate_path = os.path.join(
        os.getcwd(), root_SAM['ApplicationData']['mainScript']
    )
    print(surrogate_path)

    with open(surrogate_path, 'r') as f:
        surrogate_model = json.load(f)

    # find SAM in surrogate json

    root_SAM = surrogate_model['SAM']

    # sanity check

    if root_AIM['Applications']['EDP']['Application'] != 'SurrogateEDP':
        with open('../workflow.err', 'w') as f:
            f.write('Please select [None] in the EDP tab.')
        exit(-1)

    if (
        root_AIM['Applications']['Simulation']['Application']
        != 'SurrogateSimulation'
    ):
        with open('../workflow.err', 'w') as f:
            f.write('Please select [None] in the FEM tab.')
        exit(-1)

    # write SAM.json

    with open(SAM_file, 'w') as f:
        json.dump(root_SAM, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameAIM')
    parser.add_argument('--filenameEVENT')  # not used
    parser.add_argument('--filenameSAM')
    parser.add_argument('--mainScript')
    parser.add_argument('--getRV', nargs='?', const=True, default=False)  # Not used
    args = parser.parse_args()

    sys.exit(create_SAM(args.filenameAIM, args.filenameSAM))
