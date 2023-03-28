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
import sys
import numpy as np

from pathlib import Path

#from simcenter_common import *

convert_EDP = {
    'max_abs_acceleration' : 'PFA',
    'max_rel_disp' : 'PFD',
    'max_drift' : 'PID',
    'max_roof_drift': 'PRD',
    'residual_drift': 'RID',
    'residual_disp': 'RFD'
}

def run_surrogateGP(EVENT_input_path, SAM_input_path, AIM_input_path,
                   EDP_input_path):

    # these imports are here to save time when the app is called without
    # the -getRV flag
    #import openseespy.opensees as ops

    with open(AIM_input_path, 'r') as f:
        root_AIM = json.load(f)
    #root_GI = root_AIM['GeneralInformation']

    print("General Information tab is ignored")
    root_SAM = root_AIM['Applications']['Modeling']

    surrogate_path = os.path.join(root_SAM['ApplicationData']['MS_Path'],root_SAM['ApplicationData']['mainScript'])
    print(surrogate_path)

    # with open(surrogate_path, 'r') as f:
    #     surrogate_model = json.load(f)


    #
    # Let's call GPdriver creater
    #
    pythonEXE = sys.executable

    surrogatePredictionPath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 'performFEM', 'surrogateGP', 'gpPredict.py')


    curpath = os.getcwd()
    params_name = os.path.join(curpath,"params.in")
    surrogate_name = os.path.join(curpath,root_SAM['ApplicationData']['postprocessScript']) # pickl
    surrogate_meta_name = os.path.join(curpath,root_SAM['ApplicationData']['mainScript'])   # json

    # compute IMs
    print(f"{pythonEXE} {surrogatePredictionPath} {params_name} {surrogate_meta_name} {surrogate_name}")
    os.system(f"{pythonEXE} {surrogatePredictionPath} {params_name} {surrogate_meta_name} {surrogate_name}")

    #
    # check if the correct workflow applications are selected
    #

    if root_AIM["Applications"]["Modeling"]["Application"] != "SurrogateGPBuildingModel":
            with open("../workflow.err","w") as f:
                f.write("Do not select [None] in the FEM tab. [None] is used only when using pre-trained surrogate, i.e. when [Surrogate] is selected in the SIM Tab.")
            exit(-1)


    '''
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
    '''


def write_EDP(EVENT_input_path, SAM_input_path, AIM_input_path,
                   EDP_input_path):

    with open(AIM_input_path, 'r') as f:
        root_AIM = json.load(f)

    root_SAM = root_AIM['Applications']['Modeling']
    surrogate_path = os.path.join(root_SAM['ApplicationData']['MS_Path'],root_SAM['ApplicationData']['mainScript'])
    print(surrogate_path)

    with open(surrogate_path, 'r') as f:
        surrogate_model = json.load(f)

    #
    # EDP names and values to be mapped
    #

    edp_names = surrogate_model["ylabels"]
    edp_vals = np.loadtxt('results.out').tolist()


    #
    # Read EDP file, mapping between EDPnames and EDP.json and write scalar_data 
    #

    with open(EDP_input_path, 'r') as f:
        rootEDP = json.load(f)


    numEvents = len(rootEDP['EngineeringDemandParameters'])
    numResponses = rootEDP["total_number_edp"];


    i = 0 # current event id
    event=rootEDP['EngineeringDemandParameters'][i]
    eventEDPs = event['responses'];

    for j in range(len(eventEDPs)):
        eventEDP = eventEDPs[j]
        eventType = eventEDP["type"];

        known = False;
        if (eventType == "max_abs_acceleration"):
          edpAcronym = "PFA";
          floor = eventEDP["floor"];
          known = True;
        elif   (eventType == "max_drift"):
          edpAcronym = "PID";
          floor = eventEDP["floor2"];
          known = True;
        elif   (eventType == "max_roof_drift"):
          edpAcronym = "PRD";
          floor = eventEDP["floor2"];
          known = True;
        elif   (eventType == "residual_disp"):
          edpAcronym = "RD";
          floor = eventEDP["floor"];
          known = True;
        elif (eventType == "max_pressure"):
          edpAcronym = "PSP";
          floor = eventEDP["floor2"];              
          known = True;
        elif (eventType == "max_rel_disp"):
          edpAcronym = "PFD";
          floor = eventEDP["floor"];
          known = True;
        elif (eventType == "peak_wind_gust_speed"):
          edpAcronym = "PWS";
          floor = eventEDP["floor"];
          known = True;
        else :
          edpList = [eventType];

        if known:
            dofs = eventEDP["dofs"];

            scalar_data = []
            for dof in dofs:
                my_edp_name = '1-' + edpAcronym + '-' + floor + '-' + str(dof);

                idscalar = edp_names.index(my_edp_name)
                scalar_data += [edp_vals[idscalar]]
                edpList = [my_edp_name];

            eventEDPs[j]["scalar_data"] = scalar_data

    rootEDP['EngineeringDemandParameters'][0].pop('name','') # Remove EQ name if exists because it is confusing
    rootEDP['EngineeringDemandParameters'][0]["responses"] = eventEDPs


    with open(EDP_input_path, 'w') as f:
        json.dump(rootEDP, f, indent=2)


'''
  json_t *EDPs = json_object_get(rootEDP,"EngineeringDemandParameters");

  if (EDPs != NULL) {

    numResponses = int(json_integer_value(json_object_get(rootEDP,"total_number_edp")));
    SimCenterUQFile << " response_functions = " << numResponses << "\n response_descriptors = ";

    // for each event write the edps
    int numEvents = int(json_array_size(EDPs));
    
    // loop over all events
    for (int i=0; i<numEvents; i++) {
      
      json_t *event = json_array_get(EDPs,i);
      json_t *eventEDPs = json_object_get(event,"responses");
      int numResponses = int(json_array_size(eventEDPs));  
      
      // loop over all edp for the event
      for (int j=0; j<numResponses; j++) {
    
    json_t *eventEDP = json_array_get(eventEDPs,j);
    const char *eventType = json_string_value(json_object_get(eventEDP,"type"));
    bool known = false;
    std::string edpAcronym("");
    const char *floor = NULL;
    std::cerr << "writeResponse: type: " << eventType;
    // based on edp do something 
    if (strcmp(eventType,"max_abs_acceleration") == 0) {
      edpAcronym = "PFA";
      floor = json_string_value(json_object_get(eventEDP,"floor"));
      known = true;
    } else if   (strcmp(eventType,"max_drift") == 0) {
      edpAcronym = "PID";
      floor = json_string_value(json_object_get(eventEDP,"floor2"));
      known = true;
    } else if   (strcmp(eventType,"residual_disp") == 0) {
      edpAcronym = "RD";
      floor = json_string_value(json_object_get(eventEDP,"floor"));
      known = true;
    } else if (strcmp(eventType,"max_pressure") == 0) {
      edpAcronym = "PSP";
      floor = json_string_value(json_object_get(eventEDP,"floor2"));
      known = true;
    } else if (strcmp(eventType,"max_rel_disp") == 0) {
      edpAcronym = "PFD";
      floor = json_string_value(json_object_get(eventEDP,"floor"));
      known = true;
    } else if (strcmp(eventType,"peak_wind_gust_speed") == 0) {
      edpAcronym = "PWS";
      floor = json_string_value(json_object_get(eventEDP,"floor"));
      known = true;
    } else {
      SimCenterUQFile << "'" << eventType << "' ";
      std::string newEDP(eventType);
      edpList.push_back(newEDP);
    }
    
    if (known == true) {
      json_t *dofs = json_object_get(eventEDP,"dofs");
      int numDOF = int(json_array_size(dofs));
      
      // loop over all edp for the event
      for (int k=0; k<numDOF; k++) {
        int dof = int(json_integer_value(json_array_get(dofs,k)));
        SimCenterUQFile << "'" << i+1 << "-" << edpAcronym << "-" << floor << "-" << dof << "' ";
        std::string newEDP = std::string(std::to_string(i+1)) + std::string("-")
          + edpAcronym 
          + std::string("-") 
          + std::string(floor) +
          std::string("-") + std::string(std::to_string(dof));
        edpList.push_back(newEDP);
      }
    }
      }
    }
  } 
'''
















if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--filenameAIM',
        default=None)
    parser.add_argument('--filenameSAM')
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameEDP',default=None)
    parser.add_argument('--filenameSIM',default=None)
    parser.add_argument('--getRV',default=False,nargs='?', const=True)

    args = parser.parse_args()

    run_surrogateGP(
        args.filenameEVENT, args.filenameSAM, args.filenameAIM,
        args.filenameEDP)
    write_EDP(
        args.filenameEVENT, args.filenameSAM, args.filenameAIM,
        args.filenameEDP)