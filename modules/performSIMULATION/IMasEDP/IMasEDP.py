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
    string_types = str

import argparse, posixpath, ntpath, json

def write_RV():
    pass 

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
        sys.exit(write_RV())
    else:
        sys.exit(create_EDP(args.filenameEVENT, args.filenameEDP))