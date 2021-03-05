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
# Adam Zsarnóczay
#

import os, sys, posixpath
import argparse, json
import string
import numpy as np

def write_RV(EVENT_input_path):

    # open the event file and get the list of events
    with open(EVENT_input_path, 'r') as f:
        EVENT_in = json.load(f)
    event_list = EVENT_in['randomVariables'][0]['elements']

    evt = EVENT_in['Events'][0]
    data_dir = evt['data_dir']
    f_scale = evt['unitScaleFactor']

    file_sample_dict = {}

    for e_i, event in enumerate(event_list):
        filename, sample_id, __ = event.split('x')

        if filename not in file_sample_dict.keys():
            file_sample_dict.update({filename: [[], []]})

        file_sample_dict[filename][0].append(e_i)
        file_sample_dict[filename][1].append(int(sample_id))

    EDP_output = None

    for filename in file_sample_dict.keys():

        # get the header
        header_data = np.genfromtxt(posixpath.join(data_dir, filename),
                                    delimiter=',', names=True, max_rows=1)
        header = header_data.dtype.names

        data = np.genfromtxt(posixpath.join(data_dir, filename),
                             delimiter=',', skip_header=1)

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
                EDP_output = np.zeros((len(event_list), samples.shape[1]))
            else:
                EDP_output = np.zeros(len(event_list))

        EDP_output[file_sample_dict[filename][0]] = samples

    if len(EDP_output.shape) == 1:
        EDP_output = np.reshape(EDP_output, (EDP_output.shape[0], 1))

    EDP_output = EDP_output * f_scale

    index = np.reshape(np.arange(EDP_output.shape[0]), (EDP_output.shape[0],1))

    EDP_output = np.concatenate([index, EDP_output], axis=1)

    working_dir = posixpath.dirname(EVENT_input_path)

    # prepare the header
    header_out = []
    for h_label in header:
        h_label = h_label.strip()
        if h_label.endswith('_h'):
            header_out.append(f'1-{h_label[:-2]}-1-1')
        elif h_label.endswith('_v'):
            header_out.append(f'1-{h_label[:-2]}-1-3')
        elif h_label.endswith('_x'):
            header_out.append(f'1-{h_label[:-2]}-1-1')
        elif h_label.endswith('_y'):
            header_out.append(f'1-{h_label[:-2]}-1-2')
        else:
            header_out.append(f'1-{h_label.strip()}-1-1')

    np.savetxt(working_dir+'response.csv', EDP_output, delimiter=',',
        header=','+', '.join(header_out), comments='')

# TODO: consider removing this function
def create_EDP(EVENT_input_path, EDP_input_path):

    # load the EDP file
    with open(EDP_input_path, 'r') as f:
        EDP_in = json.load(f)

    # load the EVENT file
    with open(EVENT_input_path, 'r') as f:
        EVENT_in = json.load(f)

    # store the IM(s) in the EDP file
    for edp in EDP_in["EngineeringDemandParameters"][0]["responses"]:
        for im in EVENT_in["Events"]:
            if edp["type"] in im.keys():
                edp["scalar_data"] = [im[edp["type"]]]

    with open(EDP_input_path, 'w') as f:
        json.dump(EDP_in, f, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameBIM', default=None)
    parser.add_argument('--filenameSAM', default=None)
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameEDP')
    parser.add_argument('--filenameSIM', default=None)
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    if args.getRV:
        sys.exit(write_RV(args.filenameEVENT))
    else:
        sys.exit(create_EDP(args.filenameEVENT, args.filenameEDP))