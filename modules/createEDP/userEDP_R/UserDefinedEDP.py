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

import sys
import argparse, json

def write_RV(BIM_file, EVENT_file, EDP_file, EDP_specs):

    # We do this to provide an option for different behavior under setup,
    # even though it is unlikely to have random variables for EDPs.
    write_EDP(BIM_file, EVENT_file, EDP_file, EDP_specs)

def write_EDP(BIM_file, EVENT_file, EDP_file, EDP_specs):

    with open(BIM_file, 'r') as f:
        bim_file = json.load(f)

    with open(EVENT_file, 'r') as f:
        event_file = json.load(f)

    stories = bim_file['GI']['stories']

    with open(EDP_specs, 'r') as f:
        edp_specs = json.load(f)

    EDP_locs = edp_specs['locations']
    EDP_types = edp_specs['EDP_types']

    EDP_list = []
    total_EDP_num = 0

    for edp_name, edp_data in EDP_types.items():
        for loc_id, loc_data in edp_data.items():
            for story in range(stories+1):


                if edp_name == 'PID':
                    if story > 0:
                        EDP_list.append({
                            'type'       : edp_name,
                            'id'         : int(loc_id) + story,
                            'cline'      : loc_id,
                            'floor1'      : story-1,
                            'floor2'     : story,
                            'node'       : [EDP_locs[loc_id][s]
                                            for s in [story-1, story]],
                            'dofs'       : loc_data,
                            'scalar_data': []
                        })
                        total_EDP_num += len(loc_data)
                else:
                    EDP_list.append({
                        'type'       : edp_name,
                        'id'         : int(loc_id) + story,
                        'cline'      : loc_id,
                        'floor'      : story,
                        'node'       : EDP_locs[loc_id][story],
                        'dofs'       : loc_data,
                        'scalar_data': []
                    })
                    total_EDP_num += len(loc_data)

    edp_file = {
        "RandomVariables": [],
        "total_number_edp": total_EDP_num,
        "EngineeringDemandParameters": [{
            'name': '...',
            'responses': EDP_list
        }]
    }

    with open(EDP_file, 'w') as f:
        json.dump(edp_file, f, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameBIM')
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameSAM')
    parser.add_argument('--filenameEDP')
    parser.add_argument('--EDPspecs')
    parser.add_argument('--getRV', nargs='?', const=True, default=False)

    args = parser.parse_args()

    if args.getRV:
        sys.exit(write_RV(args.filenameBIM, args.filenameEVENT,
                          args.filenameEDP, args.EDPspecs))
    else:
        sys.exit(write_EDP(args.filenameBIM, args.filenameEVENT,
                           args.filenameEDP, args.EDPspecs))