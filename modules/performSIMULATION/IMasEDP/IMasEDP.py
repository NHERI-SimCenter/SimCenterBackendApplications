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
from pathlib import Path, PurePath

import numpy as np


def write_RV(AIM_input_path, EVENT_input_path):  # noqa: C901, N802, N803, D103
    # open the event file and get the list of events
    with open(EVENT_input_path, encoding='utf-8') as f:  # noqa: PTH123
        EVENT_in = json.load(f)  # noqa: N806
    # open the event file and get the list of events
    with open(AIM_input_path, encoding='utf-8') as f:  # noqa: PTH123
        AIM_in = json.load(f)  # noqa: N806
    # This is for geojson type of event input, which should be the future standard
    if 'SimCenterEvent' in AIM_in['Events'][0]:
        value_list = AIM_in['Events'][0]['SimCenterEvent']['Values']
        label_list = AIM_in['Events'][0]['SimCenterEvent']['Labels']
        evt = EVENT_in['Events'][0]
        data_dir = Path(evt['data_dir'])
        f_scale = evt['unitScaleFactor']
        header = label_list
        EDP_output = np.array(value_list)  # noqa: N806

    # Below is for backward compatibility with the old format
    else:
        # if there is a list of possible events, load all of them
        if len(EVENT_in['randomVariables']) > 0:
            event_list = EVENT_in['randomVariables'][0]['elements']
        else:
            event_list = [
                EVENT_in['Events'][0]['event_id'],
            ]

        evt = EVENT_in['Events'][0]
        data_dir = Path(evt['data_dir'])
        f_scale = evt['unitScaleFactor']

        file_sample_dict = {}

        for e_i, event in enumerate(event_list):
            filename, sample_id, __ = event.split('x')

            if filename not in file_sample_dict:
                file_sample_dict.update({filename: [[], []]})

            file_sample_dict[filename][0].append(e_i)
            file_sample_dict[filename][1].append(int(sample_id))

        EDP_output = None  # noqa: N806

        for filename in file_sample_dict:
            # get the header
            header_data = np.genfromtxt(
                data_dir / filename,
                delimiter=',',
                names=None,
                max_rows=1,
                dtype=str,
                ndmin=1,
            )
            header = header_data  # .dtype.

            data = np.genfromtxt(data_dir / filename, delimiter=',', skip_header=1)

            # get the number of columns and reshape the data
            col_count = len(header)
            if col_count > 1:
                data = data.reshape((data.size // col_count, col_count))
            else:
                data = np.atleast_1d(data)

            # choose the right samples
            samples = data[file_sample_dict[filename][1]]

            if EDP_output is None:
                if len(samples.shape) > 1:
                    EDP_output = np.zeros((len(event_list), samples.shape[1]))  # noqa: N806
                else:
                    EDP_output = np.zeros(len(event_list))  # noqa: N806

            EDP_output[file_sample_dict[filename][0]] = samples

    # Below is used for both geojson and old hazard difinition format
    if len(EDP_output.shape) == 1:
        EDP_output = np.reshape(EDP_output, (EDP_output.shape[0], 1))  # noqa: N806

    EDP_output = EDP_output.T  # noqa: N806

    for c_i, col in enumerate(header):
        f_i = f_scale.get(col.strip(), f_scale.get('ALL', None))
        if f_i is None:
            raise ValueError(f'No units defined for {col}')  # noqa: EM102, TRY003

        EDP_output[c_i] *= f_i

    EDP_output = EDP_output.T  # noqa: N806

    index = np.reshape(np.arange(EDP_output.shape[0]), (EDP_output.shape[0], 1))

    EDP_output = np.concatenate([index, EDP_output], axis=1)  # noqa: N806

    working_dir = Path(PurePath(EVENT_input_path).parent)
    # working_dir = posixpath.dirname(EVENT_input_path)

    # prepare the header
    header_out = []
    for h_label in header:
        # remove leading and trailing whitespace
        h_label = h_label.strip()  # noqa: PLW2901

        # convert suffixes to the loc-dir format used by the SimCenter
        if h_label.endswith('_h'):  # horizontal
            header_out.append(f'1-{h_label[:-2]}-1-1')

        elif h_label.endswith('_v'):  # vertical
            header_out.append(f'1-{h_label[:-2]}-1-3')

        elif h_label.endswith('_x'):  # x direction
            header_out.append(f'1-{h_label[:-2]}-1-1')

        elif h_label.endswith('_y'):  # y direction
            header_out.append(f'1-{h_label[:-2]}-1-2')

        else:  # if none of the above is given, default to 1-1
            header_out.append(f'1-{h_label.strip()}-1-1')

    np.savetxt(
        working_dir / 'response.csv',
        EDP_output,
        delimiter=',',
        header=',' + ', '.join(header_out),
        comments='',
    )


# TODO: consider removing this function  # noqa: TD002
# It is not used currently
def create_EDP(EVENT_input_path, EDP_input_path):  # noqa: N802, N803, D103
    # load the EDP file
    with open(EDP_input_path, encoding='utf-8') as f:  # noqa: PTH123
        EDP_in = json.load(f)  # noqa: N806

    # load the EVENT file
    with open(EVENT_input_path, encoding='utf-8') as f:  # noqa: PTH123
        EVENT_in = json.load(f)  # noqa: N806

    # store the IM(s) in the EDP file
    for edp in EDP_in['EngineeringDemandParameters'][0]['responses']:
        for im in EVENT_in['Events']:
            if edp['type'] in im.keys():  # noqa: SIM118
                edp['scalar_data'] = [im[edp['type']]]

    with open(EDP_input_path, 'w', encoding='utf-8') as f:  # noqa: PTH123
        json.dump(EDP_in, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameAIM', default=None)
    parser.add_argument('--filenameSAM', default=None)
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameEDP')
    parser.add_argument('--filenameSIM', default=None)
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    if args.getRV:
        sys.exit(write_RV(args.filenameAIM, args.filenameEVENT))
    else:
        sys.exit(create_EDP(args.filenameEVENT, args.filenameEDP))
