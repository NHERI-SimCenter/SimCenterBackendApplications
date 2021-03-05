# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
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
# Adam Zsarn�czay
#

import sys, argparse,json

def create_SAM(BIM_file, EVENT_file, SAM_file,
    model_script, model_path, ndm, dof_map, column_line, getRV):

    with open(BIM_file, 'r') as f:
        root_BIM = json.load(f)['GeneralInformation']

    try:
        stories = root_BIM['NumberOfStories']
    except:
        raise ValueError("OpenSeesPyInput - structural information missing")

    if column_line is None:
        nodes = list(range(stories+1))
    else:
        nodes = [int(node) for node in column_line.split(',')]
        nodes = nodes[:stories+1]

    node_map = []
    for floor, node in enumerate(nodes):
        node_entry = {}
        node_entry['node'] = node
        node_entry['cline'] = '1'
        node_entry['floor'] = f'{floor}'
        node_map.append(node_entry)

    root_SAM = {
        'mainScript': model_script,
        'modelPath': model_path,
        'dofMap': dof_map,
        'recorderNodes': nodes,
        'type': 'OpenSeesPyInput',
        'NodeMapping': node_map,
        'numStory': stories,
        'ndm': ndm,
        # TODO: improve this if we want random vars in the structure
        'randomVar': []
    }

    with open(SAM_file, 'w') as f:
        json.dump(root_SAM, f, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameBIM')
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameSAM')
    parser.add_argument('--mainScript')
    parser.add_argument('--modelPath')
    parser.add_argument('--ndm', default="3")
    parser.add_argument('--dofMap', default="1, 2, 3")
    parser.add_argument('--columnLine', default=None)
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    sys.exit(create_SAM(
        args.filenameBIM, args.filenameEVENT, args.filenameSAM,
        args.mainScript, args.modelPath, args.ndm,
        args.dofMap, args.columnLine, args.getRV))