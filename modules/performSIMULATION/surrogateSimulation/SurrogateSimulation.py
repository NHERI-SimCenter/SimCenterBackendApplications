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

def run_surrogateGP(AIM_input_path, EDP_input_path):

    # these imports are here to save time when the app is called without
    # the -getRV flag
    #import openseespy.opensees as ops

    with open(AIM_input_path, 'r') as f:
        root_AIM = json.load(f)
    #root_GI = root_AIM['GeneralInformation']

    root_SAM = root_AIM['Applications']['Modeling']

    surrogate_path = os.path.join(root_SAM['ApplicationData']['MS_Path'],root_SAM['ApplicationData']['mainScript'])

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
    # print(f"{pythonEXE} {surrogatePredictionPath} {params_name} {surrogate_meta_name} {surrogate_name}")
    os.system(f"{pythonEXE} {surrogatePredictionPath} {params_name} {surrogate_meta_name} {surrogate_name}")

    #
    # check if the correct workflow applications are selected
    #

    if (root_AIM["Applications"]["Modeling"]["Application"] != "SurrogateGPBuildingModel") and (root_AIM["Applications"]["Simulation"]["Application"] != "SurrogateRegionalPy"):
            with open("../workflow.err","w") as f:
                f.write("Do not select [None] in the FEM tab. [None] is used only when using pre-trained surrogate, i.e. when [Surrogate] is selected in the SIM Tab.")
            exit(-1)


def write_EDP(AIM_input_path,EDP_input_path, newEDP_input_path=None):

    with open(AIM_input_path, 'r') as f:
        root_AIM = json.load(f)

    if newEDP_input_path ==None:
        newEDP_input_path = EDP_input_path

    root_SAM = root_AIM['Applications']['Modeling']
    curpath = os.getcwd()
    #surrogate_path = os.path.join(root_SAM['ApplicationData']['MS_Path'],root_SAM['ApplicationData']['mainScript'])
    surrogate_path = os.path.join(curpath,root_SAM['ApplicationData']['mainScript'])

    with open(surrogate_path, 'r') as f:
        surrogate_model = json.load(f)

    #
    # EDP names and values to be mapped
    #

    edp_names = surrogate_model["ylabels"]
    
    if not os.path.isfile('results.out'):
        # not found
        print("Skiping surrogateEDP - results.out does not exists in " + os.getcwd())
        exit(-1)
    elif os.stat('results.out').st_size == 0:
        # found but empty
        print("Skiping surrogateEDP - results.out is empty in " + os.getcwd())
        exit(-1)


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
          floor = "1";
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


    with open(newEDP_input_path, 'w') as f:
        json.dump(rootEDP, f, indent=2)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--filenameAIM',
        default=None)
    parser.add_argument('--filenameSAM')
    parser.add_argument('--filenameEVENT') # not used
    parser.add_argument('--filenameEDP',default=None)
    parser.add_argument('--filenameSIM',default=None) # not used
    parser.add_argument('--getRV',default=False,nargs='?', const=True)

    args = parser.parse_args()

    if not args.getRV:
        run_surrogateGP(args.filenameAIM,args.filenameEDP)
        write_EDP(args.filenameAIM, args.filenameEDP)