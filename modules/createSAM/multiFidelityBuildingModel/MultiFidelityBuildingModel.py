#  # noqa: INP001, D100
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
#
# You should have received a copy of the BSD 3-Clause License along with
# this file. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarn�czay
#

import argparse
import json
import os
import shutil
import subprocess
import sys


def create_SAM(  # noqa: N802, D103, C901
    AIM_file,  # noqa: N803
    EVENT_file,  # noqa: N803
    SAM_file,  # noqa: N803
    getRV,  # noqa: N803
):
    with open(AIM_file, encoding='utf-8') as f:  # noqa: PTH123
        AIM = json.load(f)  # noqa: N806

    current_building_id = AIM['GeneralInformation']['AIM_id']
    database_path = AIM['Modeling']['buildingDatabase']
    with open(  # noqa: PTH123, UP015
        os.path.join('..', '..', '..', '..', 'input_data', database_path),  # noqa: PTH118
        'r',  # noqa: PTH118, RUF100
    ) as f:
        json_string = f.read()
        database = json.loads(json_string)

    ### check if the id exists in json file
    matching_count = 0
    for item in database:
        if 'id' in item:
            if isinstance(item['id'], list):
                id_list = [int(myid) for myid in item['id']]
                if int(current_building_id) in id_list:
                    print(f'building {current_building_id} found in the database')  # noqa: T201
                    matching_model = (
                        item  # If user_id is in the list, return this item
                    )
                    # break
                    matching_count += 1
            elif int(item['id']) == int(current_building_id):
                matching_model = item  # If the id matches directly, return this item
                matching_count += 1

    if matching_count > 2:  # noqa: PLR2004
        msg = f'Error in multifidelity preprocessor: Multiple model files are mapped to the same asset id {current_building_id}.'
        print(msg)  # noqa: T201
        raise ValueError(msg)

    if matching_count == 1:
        print('Model found, running custom OpenSees model')  # noqa: T201
        with open('SAM.json', 'w') as f:  # noqa: PTH123
            json.dump(matching_model, f, indent=2)  # Write with pretty formatting

        if getRV:
            #
            # Copy all the important files
            #
            src = os.path.join(  # noqa: PTH118
                '..', '..', '..', '..', 'input_data', matching_model['mainFolder']
            )
            dst = os.getcwd()  # noqa: PTH109

            for item in os.listdir(src):
                source_item = os.path.join(src, item)  # noqa: PTH118
                dest_item = os.path.join(dst, item)  # noqa: PTH118

                # If the item is a directory, recursively copy it
                if os.path.isdir(source_item):  # noqa: PTH112
                    shutil.copytree(source_item, dest_item, dirs_exist_ok=True)
                else:
                    # If it's a file, copy it to the destination
                    shutil.copy2(source_item, dest_item)

    else:
        # call MDOF-Lu
        MDOF_exe = os.path.join(  # noqa: PTH118, N806
            os.path.dirname(  # noqa: PTH120
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # noqa: PTH100, PTH120
            ),
            'createSAM',
            'MDOF-LU',
            'MDOF-LU',
        )

        if getRV:
            command = [
                MDOF_exe,
                '--filenameAIM',
                AIM_file,
                '--filenameEVENT',
                EVENT_file,
                '--filenameSAM',
                SAM_file,
                '--getRV',
            ]
        else:
            command = [
                MDOF_exe,
                '--filenameAIM',
                AIM_file,
                '--filenameEVENT',
                EVENT_file,
                '--filenameSAM',
                SAM_file,
            ]

        subprocess.run(command, check=True)  # noqa: S603


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameAIM')
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameSAM')
    parser.add_argument('--mainScript')
    parser.add_argument('--modelPath')
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
            args.getRV,
        )
    )
