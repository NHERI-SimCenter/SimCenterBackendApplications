from __future__ import division, print_function
import os, sys
if sys.version.startswith('2'):
    range=xrange
    string_types = basestring
else:
    string_types = str

import argparse, posixpath, ntpath, json
import importlib
import numpy as np
from openseespy.opensees import *

convert_EDP = {
    'max_abs_acceleration' : 'PFA',
    'max_rel_disp' : 'PFD',
    'max_drift' : 'PID'
}

def write_RV():

    pass

    # TODO: check simulation data exists and contains all important fields
    # TODO: get simulation data & write to SIM file

def run_openseesPy(EVENT_input_path, SAM_input_path, BIM_input_path, 
                   EDP_input_path):

    sys.path.insert(0, os.getcwd())

    # load the model builder script
    with open(SAM_input_path, 'r') as f:
        SAM_in = json.load(f)

    model_script = importlib.__import__(
        SAM_in['mainScript'][:-3], globals(), locals(), ['build_model',], 0)
    build_model = model_script.build_model

    # build the model
    build_model()

    # load the event file
    with open(EVENT_input_path, 'r') as f:
        EVENT_in = json.load(f)['Events'][0]

    event_list = EVENT_in['timeSeries']
    pattern_list = EVENT_in['pattern']
    pattern_ts_link = [p['timeSeries'] for p in pattern_list]

    TS_list = []

    f_G = 386.089

    # define the time series
    for evt_i, event in enumerate(event_list):
        
        timeSeries('Path', evt_i+2, '-dt', event['dT'], '-factor', event['factor']*f_G,
                   '-values', *event['data'], '-prependZero')

        pat = pattern_list[pattern_ts_link.index(event['name'])]

        pattern('UniformExcitation', evt_i+2, pat['dof'], '-accel', evt_i+2)

        #TS_list.append(list(np.insert(np.array(event['data'])*event['factor']*f_G, 0, 0.)))
        TS_list.append(list(np.array([0.,] + event['data'])*event['factor']*f_G))

    # TODO: recorders

    # load the analysis script
    with open(BIM_input_path, 'r') as f:
        BIM_in = json.load(f)
        analysis_script_path = BIM_in['Simulation']['fileName']
        recorder_nodes = BIM_in['StructuralInformation']['nodes']
    analysis_script = importlib.__import__(
        analysis_script_path[:-3], globals(), locals(), ['run_analysis',], 0)
    run_analysis = analysis_script.run_analysis

    # run the analysis
    EDP_res = run_analysis(GM_dt = EVENT_in['dT'], GM_npts=EVENT_in['numSteps'], 
        TS_List = TS_list, nodes_COD = recorder_nodes)

    # TODO: default analysis script

    # save the EDP results
    with open(EDP_input_path, 'r') as f:
        EDP_in = json.load(f)

    EDP_list = EDP_in['EngineeringDemandParameters'][0]['responses']

    for response in EDP_list:
        EDP_kind = convert_EDP[response['type']]
        if EDP_kind != 'PID':
            loc = response['floor']
        else:
            loc = response['floor2']
        response['scalar_data'] = EDP_res[EDP_kind][loc]

    with open(EDP_input_path, 'w') as f:
        json.dump(EDP_in, f, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameBIM', default=None)
    parser.add_argument('--filenameSAM')
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameEDP', default=None)
    parser.add_argument('--filenameSIM', default=None)
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    if args.getRV:
        sys.exit(write_RV())
    else:
        sys.exit(run_openseesPy(
            args.filenameEVENT, args.filenameSAM, args.filenameBIM,
            args.filenameEDP))



