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
# Joanna J. Zou
#

import os, sys
import argparse, json
import importlib

from pathlib import Path

# import the common constants and methods
this_dir = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
main_dir = this_dir.parents[1]

sys.path.insert(0, str(main_dir / 'common'))

from simcenter_common import *

convert_EDP = {
    'max_abs_acceleration' : 'PFA',
    'max_rel_disp' : 'PFD',
    'max_drift' : 'PID',
    'max_roof_drift': 'PRD',
    'residual_drift': 'RID',
    'residual_disp': 'RFD'
}

def write_RV():

    pass

    # TODO: check simulation data exists and contains all important fields
    # TODO: get simulation data & write to SIM file

def run_openseesPy(EVENT_input_path, SAM_input_path, BIM_input_path,
                   EDP_input_path):

    # these imports are here to save time when the app is called without
    # the -getRV flag
    import sys
    import numpy as np
    import openseespy.opensees as ops

    log_msg('Startring simulation script...')

    sys.path.insert(0, os.getcwd())

    # load the model builder script
    with open(BIM_input_path, 'r') as f:
        BIM_in = json.load(f)

    model_params = BIM_in['GeneralInformation']

    with open(SAM_input_path, 'r') as f:
        SAM_in = json.load(f)

    sys.path.insert(0, SAM_in['modelPath'])

    model_script_path = SAM_in['mainScript']

    dof_map = [int(dof) for dof in SAM_in['dofMap'].split(',')]

    node_map = dict([(int(entry['floor']), int(entry['node']))
                     for entry in SAM_in['NodeMapping']])

    model_script = importlib.__import__(
        model_script_path[:-3], globals(), locals(), ['build_model',], 0)
    build_model = model_script.build_model

    ops.wipe()

    # build the model
    build_model(model_params=model_params)

    # load the event file
    with open(EVENT_input_path, 'r') as f:
        EVENT_in = json.load(f)['Events'][0]

    event_list = EVENT_in['timeSeries']
    pattern_list = EVENT_in['pattern']
    # TODO: use dictionary
    pattern_ts_link = [p['timeSeries'] for p in pattern_list]

    TS_list = []

    # define the time series
    for evt_i, event in enumerate(event_list):

        ops.timeSeries('Path', evt_i+2, '-dt', event['dT'], '-factor', 1.0,
                   '-values', *event['data'], '-prependZero')

        pat = pattern_list[pattern_ts_link.index(event['name'])]

        ops.pattern('UniformExcitation', evt_i+2, dof_map[pat['dof']-1], '-accel', evt_i+2)

        TS_list.append(list(np.array([0.,] + event['data'])))

    # load the analysis script
    analysis_script = importlib.__import__(
        model_script_path[:-3], globals(), locals(), ['run_analysis',], 0)
    run_analysis = analysis_script.run_analysis

    recorder_nodes = SAM_in['recorderNodes']

    # create the EDP specification

    # load the EDP file
    with open(EDP_input_path, 'r') as f:
        EDP_in = json.load(f)

    EDP_list = EDP_in['EngineeringDemandParameters'][0]['responses']

    edp_specs = {}
    for response in EDP_list:

        if response['type'] in list(convert_EDP.keys()):
            response['type'] = convert_EDP[response['type']]

        if response['type'] not in edp_specs.keys():
            edp_specs.update({response['type']: {}})

        if 'node' in list(response.keys()):

            if response.get('id', None) is None:
                response.update({'id': 0})

            edp_specs[response['type']].update({
                response['id']: dict([(dof, list(np.atleast_1d(response['node'])))
                                      for dof in response['dofs']])})
        else:

            if response.get('floor', None) is not None:
                floor = int(response['floor'])
                node_list = [node_map[floor],]
            else:
                floor = int(response['floor2'])
                floor1 = int(response['floor1'])
                node_list = [node_map[floor1], node_map[floor]]

            if response.get('id', None) is None:
                response.update({'id': floor})
            if floor is not None:
                edp_specs[response['type']].update({
                    response['id']: dict([(dof, node_list)
                                          for dof in response['dofs']])})

    #for edp_name, edp_data in edp_specs.items():
    #    print(edp_name, edp_data)

    # run the analysis
    # TODO: default analysis script
    EDP_res = run_analysis(GM_dt = EVENT_in['dT'],
        GM_npts=EVENT_in['numSteps'],
        TS_List = TS_list, EDP_specs = edp_specs,
        model_params = model_params)

    # save the EDP results

    #print(EDP_res)

    for response in EDP_list:
        edp = EDP_res[response['type']][response['id']]
        #print(edp)

        response['scalar_data'] = edp # [val for dof, val in edp.items()]
        #print(response)

    with open(EDP_input_path, 'w') as f:
        json.dump(EDP_in, f, indent=2)

    log_msg('Simulation script finished.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--filenameBIM',
        default=None)
    parser.add_argument('--filenameSAM')
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameEDP',
        default=None)
    parser.add_argument('--filenameSIM',
        default=None)
    parser.add_argument('--getRV',
        default=False,
        nargs='?', const=True)

    args = parser.parse_args()

    if args.getRV:
        sys.exit(write_RV())
    else:
        sys.exit(run_openseesPy(
            args.filenameEVENT, args.filenameSAM, args.filenameBIM,
            args.filenameEDP))
