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

import sys, argparse,json, os

def create_SAM(AIM_file, EVENT_file, SAM_file,
    model_script):

    with open(AIM_file, 'r') as f:
        root_AIM = json.load(f)
    #root_GI = root_AIM['GeneralInformation']

    print("General Information tab is ignored")
    root_SAM = root_AIM['Applications']['Modeling']

    surrogate_path = os.path.join(root_SAM['ApplicationData']['MS_Path'],root_SAM['ApplicationData']['mainScript'])
    print(surrogate_path)

    with open(surrogate_path, 'r') as f:
        surrogate_model = json.load(f) 

    root_SAM = surrogate_model['SAM']



    if root_AIM["Applications"]["EDP"]["Application"] != "SurrogateEDP":
            with open("../workflow.err","w") as f:
                f.write("Please select [None] in the EDP tab.")
            exit(-1)

    if root_AIM["Applications"]["Simulation"]["Application"] != "SurrogateSimulation":
            with open("../workflow.err","w") as f:
                f.write("Please select [None] in the FEM tab.")
            exit(-1)

    #root_GI = root_AIM['GeneralInformation']

    # Read "SAM" from the surrogate model, and dump it into SAM_file
    '''
    try:
        stories = root_GI['NumberOfStories']
    except:
        raise ValueError("number of stories information missing")

    if column_line is None:
        # KZ: looking into SAM
        root_SAM = root_AIM.get('Modeling', {})
        nodes = root_SAM.get('centroidNodes', [])
        if len(nodes) == 0:
            nodes = list(range(stories+1))
    else:
        nodes = [int(node) for node in column_line.split(',')]
        nodes = nodes[:stories+1]

    node_map = []
    for floor, node in enumerate(nodes):
        node_entry = {}
        node_entry['node'] = node
        # KZ: correcting the cline
        node_entry['cline'] = 'response'
        node_entry['floor'] = f'{floor}'
        node_map.append(node_entry)

    root_SAM = {
        'mainScript': model_script,
        'modelPath': model_path,
        'dofMap': dof_map,
        'recorderNodes': nodes,
        'type': 'CustomPyInput',
        'NodeMapping': node_map,
        'numStory': stories,
        # KZ: correcting the ndm format --> this causing standardEarthquakeEDP failure...
        'ndm': int(ndm),
        # TODO: improve this if we want random vars in the structure
        'randomVar': []
    }
    '''

    with open(SAM_file, 'w') as f:
        json.dump(root_SAM, f, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameAIM')
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameSAM')
    parser.add_argument('--mainScript')
    parser.add_argument('--getRV', nargs='?', const=True, default=False) # Not used
    args = parser.parse_args()

    sys.exit(create_SAM(
        args.filenameAIM, args.filenameEVENT, args.filenameSAM,
        args.mainScript))
