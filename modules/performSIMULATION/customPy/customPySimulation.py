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
#

import os, sys
import argparse, json
import importlib, shutil

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

    # create an empty SIM file

    SIM = {}

    with open('SIM.json', 'w', encoding="utf-8") as f:
        json.dump(SIM, f, indent=2)

    # TODO: check simulation data exists and contains all important fields
    # TODO: get simulation data & write to SIM file

def run_simulation(EVENT_input_path, SAM_input_path, BIM_input_path,
                   EDP_input_path):

    # these imports are here to save time when the app is called without
    # the -getRV flag
    import sys

    log_msg('Startring simulation script...')

    working_dir = os.getcwd()

    sys.path.insert(0, os.getcwd())

    # load the BIM file
    with open(BIM_input_path, 'r', encoding="utf-8") as f:
        BIM_in = json.load(f)

    # load the SAM file
    with open(SAM_input_path, 'r', encoding="utf-8") as f:
        SAM_in = json.load(f)

    # load the event file
    with open(EVENT_input_path, 'r', encoding="utf-8") as f:
        EVENT_in = json.load(f)['Events'][0]

    # load the EDP file
    with open(EDP_input_path, 'r', encoding="utf-8") as f:
        EDP_in = json.load(f)

    # KZ: commented out --> we're running at the current workdir
    #sys.path.insert(0, SAM_in['modelPath'])
    #os.chdir(SAM_in['modelPath'])
    #print(os.listdir(os.getcwd()))
    #print(os.getcwd())

    custom_script_path = SAM_in['mainScript']

    # copy the custom scripts to the current directory if not yet
    if os.path.exists(custom_script_path):
        pass
    else:
        custom_script_dir = SAM_in.get('modelPath',None)
        if custom_script_dir is None:
            log_msg('No modelPath found in the SAM file.')
        else:
            shutil.copytree(custom_script_dir,os.getcwd(),dirs_exist_ok=True)
            log_msg('Custom scripts copied from {} to {}'.format(custom_script_dir,os.getcwd()))

    custom_script = importlib.__import__(
        custom_script_path[:-3], globals(), locals(), ['custom_analysis',], 0)

    custom_analysis = custom_script.custom_analysis

    # run the analysis
    EDP_res = custom_analysis(BIM=BIM_in, EVENT=EVENT_in, SAM=SAM_in, EDP=EDP_in)

    os.chdir(working_dir)
    results_txt = ""

    EDP_list = EDP_in['EngineeringDemandParameters'][0]['responses']
    # KZ: rewriting the parsing step of EDP_res to EDP_list
    for response in EDP_list:
        print('response = ', response)
        response['scalar_data'] = []
        try:
            val = EDP_res.get(response['type'], None)
            print('val = ', val)
            if val is None:
                # try conversion
                edp_name = convert_EDP.get(response['type'], None)
                print('edp_name = ', edp_name)
                if edp_name is not None:
                    if 'PID' in edp_name:
                        cur_floor = response['floor2']
                        dofs = response.get('dofs',[])
                    elif 'PRD' in edp_name:
                        cur_floor = response['floor2']
                        dofs = response['dofs']
                    else:
                        cur_floor = response['floor']
                        dofs = response['dofs']
                    if len(dofs) == 0:
                        dofs = [1, 2] #default is bidirection
                        response['dofs'] = dofs
                    print('dofs = ', dofs)
                    for cur_dof in dofs:
                        key_name = '1-'+edp_name+'-{}-{}'.format(int(cur_floor), int(cur_dof))
                        print('key_name = ', key_name)
                        res = EDP_res.get(key_name, None)
                        if res is None:
                            response['scalar_data'].append('NaN')
                            results_txt += 'NaN '
                        else:
                            response['scalar_data'].append(float(EDP_res[key_name]))
                            results_txt += str(float(EDP_res[key_name])) + ' '
                            print('response = ', response)
                else:
                    response['scalar_data'] = ['NaN']
                    results_txt += 'NaN '
            else:
                response['scalar_data'] = [float(val)]
                results_txt += str(float(EDP_res[response['type']])) + ' '
        except:
            response['scalar_data'] = ['NaN']
            results_txt += 'NaN '
        #edp = EDP_res[response['type']][response['id']]
        #print(edp)

        #response['scalar_data'] = edp # [val for dof, val in edp.items()]
        #print(response)
    results_txt = results_txt[:-1]

    with open(EDP_input_path, 'w', encoding="utf-8") as f:
        json.dump(EDP_in, f, indent=2)

    with open('results.out', 'w', encoding="utf-8") as f:
        f.write(results_txt)

    """
    model_params = BIM_in['GeneralInformation']

    dof_map = [int(dof) for dof in SAM_in['dofMap'].split(',')]

    node_map = dict([(int(entry['floor']), int(entry['node']))
                     for entry in SAM_in['NodeMapping']])

    ops.wipe()

    # build the model
    build_model(model_params=model_params)

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

    with open(EDP_input_path, 'w', encoding="utf-8") as f:
        json.dump(EDP_in, f, indent=2)
    """

    log_msg('Simulation script finished.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--filenameAIM',
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
        sys.exit(run_simulation(
            args.filenameEVENT, args.filenameSAM, args.filenameAIM,
            args.filenameEDP))
