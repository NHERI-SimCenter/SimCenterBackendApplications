#  # noqa: INP001, D100
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
#

import argparse
import json
import sys


def create_SAM(  # noqa: ANN201, N802, D103, PLR0913
    AIM_file,  # noqa: ANN001, N803
    EVENT_file,  # noqa: ANN001, ARG001, N803
    SAM_file,  # noqa: ANN001, N803
    model_script,  # noqa: ANN001
    model_path,  # noqa: ANN001
    ndm,  # noqa: ANN001
    dof_map,  # noqa: ANN001
    column_line,  # noqa: ANN001
    getRV,  # noqa: ANN001, ARG001, N803
):
    # KZ: modifying BIM to AIM
    with open(AIM_file, encoding='utf-8') as f:  # noqa: PTH123
        root_AIM = json.load(f)  # noqa: N806
    root_GI = root_AIM['GeneralInformation']  # noqa: N806

    try:
        stories = int(root_GI['NumberOfStories'])
    except:  # noqa: E722
        raise ValueError('number of stories information missing')  # noqa: B904, EM101, TRY003

    if column_line is None:
        # KZ: looking into SAM
        root_SAM = root_AIM.get('Modeling', {})  # noqa: N806
        nodes = root_SAM.get('centroidNodes', [])
        if len(nodes) == 0:
            nodes = list(range(stories + 1))
    else:
        nodes = [int(node) for node in column_line.split(',')]
        nodes = nodes[: stories + 1]

    node_map = []
    for floor, node in enumerate(nodes):
        node_entry = {}
        node_entry['node'] = node
        # KZ: correcting the cline
        node_entry['cline'] = 'response'
        node_entry['floor'] = f'{floor}'
        node_map.append(node_entry)

    root_SAM = {  # noqa: N806
        'mainScript': model_script,
        'modelPath': model_path,
        'dofMap': dof_map,
        'recorderNodes': nodes,
        'type': 'CustomPyInput',
        'NodeMapping': node_map,
        'numStory': stories,
        # KZ: correcting the ndm format --> this causing standardEarthquakeEDP failure...
        'ndm': int(ndm),
        # TODO: improve this if we want random vars in the structure  # noqa: FIX002, TD002, TD003
        'randomVar': [],
    }

    # pass all other attributes in the AIM GI to SAM
    for cur_key in root_GI.keys():  # noqa: SIM118
        cur_item = root_GI.get(cur_key, None)
        if cur_key in root_SAM.keys():  # noqa: SIM118
            pass
        else:
            root_SAM[cur_key] = cur_item

    with open(SAM_file, 'w', encoding='utf-8') as f:  # noqa: PTH123
        json.dump(root_SAM, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameAIM')
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameSAM')
    parser.add_argument('--mainScript')
    parser.add_argument('--modelPath', default='')
    parser.add_argument('--ndm', default='3')
    parser.add_argument('--dofMap', default='1, 2, 3')
    parser.add_argument('--columnLine', default=None)
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    sys.exit(
        create_SAM(
            args.filenameAIM,
            args.filenameEVENT,
            args.filenameSAM,
            args.mainScript,
            args.modelPath,
            args.ndm,
            args.dofMap,
            args.columnLine,
            args.getRV,
        )
    )
