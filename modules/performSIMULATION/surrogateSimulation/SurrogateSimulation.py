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
# Joanna J. Zou
#

import argparse  # noqa: I001
import json
import os
import sys
import subprocess

import numpy as np

errFileName = os.path.join(os.getcwd(), 'workflow.err')  # noqa: N816, PTH109, PTH118
sys.stderr = open(errFileName, 'a')  # noqa: SIM115, PTH123

# from simcenter_common import *

convert_EDP = {  # noqa: N816
    'max_abs_acceleration': 'PFA',
    'max_rel_disp': 'PFD',
    'max_drift': 'PID',
    'max_roof_drift': 'PRD',
    'residual_drift': 'RID',
    'residual_disp': 'RFD',
}


def run_surrogateGP(AIM_input_path, EDP_input_path):  # noqa: ARG001, N802, N803, D103
    # these imports are here to save time when the app is called without
    # the -getRV flag
    # import openseespy.opensees as ops

    with open(AIM_input_path, encoding='utf-8') as f:  # noqa: PTH123
        root_AIM = json.load(f)  # noqa: N806
    # root_GI = root_AIM['GeneralInformation']

    root_SAM = root_AIM['Applications']['Modeling']  # noqa: N806

    surrogate_path = os.path.join(  # noqa: PTH118, F841
        root_SAM['ApplicationData']['MS_Path'],
        root_SAM['ApplicationData']['mainScript'],
    )

    # with open(surrogate_path, 'r') as f:
    #     surrogate_model = json.load(f)

    #
    # Let's call GPdriver creator?
    #
    pythonEXE = sys.executable  # noqa: N806

    surrogatePredictionPath = os.path.join(  # noqa: PTH118, N806
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # noqa: PTH100, PTH120
        'performFEM',
        'surrogateGP',
        'gpPredict.py',
    )

    curpath = os.getcwd()  # noqa: PTH109
    params_name = os.path.join(curpath, 'params.in')  # noqa: PTH118
    surrogate_name = os.path.join(  # noqa: PTH118
        curpath, root_SAM['ApplicationData']['postprocessScript']
    )  # pickl
    surrogate_meta_name = os.path.join(  # noqa: PTH118
        curpath, root_SAM['ApplicationData']['mainScript']
    )  # json

    # compute IMs
    # print(f"{pythonEXE} {surrogatePredictionPath} {params_name} {surrogate_meta_name} {surrogate_name}")

    command = [
        pythonEXE,
        surrogatePredictionPath,
        params_name,
        surrogate_meta_name,
        surrogate_name,
    ]
    # subprocess.run(command, check=True)  # noqa: RUF100, S603

    try:
        result = subprocess.check_output(  # noqa: S603
            command, stderr=subprocess.STDOUT, text=True
        )
        returncode = 0
    except subprocess.CalledProcessError as e:
        result = e.output
        returncode = e.returncode

    if not returncode == 0:  # noqa: SIM201
        print(  # noqa: T201
            result,
            file=sys.stderr,
        )  # noqa: RUF100, T201

    # os.system(  # noqa: RUF100, S605
    #    f'{pythonEXE} {surrogatePredictionPath} {params_name} {surrogate_meta_name} {surrogate_name}'
    # )

    #
    # check if the correct workflow applications are selected
    #

    if (
        root_AIM['Applications']['Modeling']['Application']
        != 'SurrogateGPBuildingModel'
    ) and (
        root_AIM['Applications']['Simulation']['Application']
        != 'SurrogateRegionalPy'
    ):
        # with open('./workflow.err', 'w') as f:  # noqa: PTH123, RUF100
        #     f.write(
        #         'Do not select [None] in the FEM tab. [None] is used only when using pre-trained surrogate, i.e. when [Surrogate] is selected in the SIM Tab.'
        #     )
        # exit(-1)  # noqa: PLR1722, RUF100
        print(  # noqa: T201
            'Do not select [None] in the FEM tab. [None] is used only when using pre-trained surrogate, i.e. when [Surrogate] is selected in the SIM Tab.',
            file=sys.stderr,
        )  # noqa: RUF100, T201
        exit(-1)  # noqa: PLR1722


def write_EDP(AIM_input_path, EDP_input_path, newEDP_input_path=None):  # noqa: C901, N802, N803, D103
    with open(AIM_input_path, encoding='utf-8') as f:  # noqa: PTH123
        root_AIM = json.load(f)  # noqa: N806

    if newEDP_input_path == None:  # noqa: E711
        newEDP_input_path = EDP_input_path  # noqa: N806

    root_SAM = root_AIM['Applications']['Modeling']  # noqa: N806
    curpath = os.getcwd()  # noqa: PTH109
    # surrogate_path = os.path.join(root_SAM['ApplicationData']['MS_Path'],root_SAM['ApplicationData']['mainScript'])
    surrogate_path = os.path.join(curpath, root_SAM['ApplicationData']['mainScript'])  # noqa: PTH118

    with open(surrogate_path, encoding='utf-8') as f:  # noqa: PTH123
        surrogate_model = json.load(f)

    #
    # EDP names and values to be mapped
    #

    edp_names = surrogate_model['ylabels']

    if not os.path.isfile('results.out'):  # noqa: PTH113
        # not found
        print('Skiping surrogateEDP - results.out does not exist in ' + os.getcwd())  # noqa: T201, PTH109
        exit(-1)  # noqa: PLR1722
    elif os.stat('results.out').st_size == 0:  # noqa: PTH116
        # found but empty
        print('Skiping surrogateEDP - results.out is empty in ' + os.getcwd())  # noqa: T201, PTH109
        exit(-1)  # noqa: PLR1722

    edp_vals = np.loadtxt('results.out').tolist()
    #
    # Read EDP file, mapping between EDPnames and EDP.json and write scalar_data
    #

    with open(EDP_input_path, encoding='utf-8') as f:  # noqa: PTH123
        rootEDP = json.load(f)  # noqa: N806

    numEvents = len(rootEDP['EngineeringDemandParameters'])  # noqa: N806, F841
    numResponses = rootEDP['total_number_edp']  # noqa: N806, F841
    i = 0  # current event id
    event = rootEDP['EngineeringDemandParameters'][i]
    eventEDPs = event['responses']  # noqa: N806
    for j in range(len(eventEDPs)):
        eventEDP = eventEDPs[j]  # noqa: N806
        eventType = eventEDP['type']  # noqa: N806
        known = False
        if eventType == 'max_abs_acceleration':
            edpAcronym = 'PFA'  # noqa: N806
            floor = eventEDP['floor']
            known = True
        elif eventType == 'max_drift':
            edpAcronym = 'PID'  # noqa: N806
            floor = eventEDP['floor2']
            known = True
        elif eventType == 'rms_acceleration':
            edpAcronym = 'RMSA'  # noqa: N806
            floor = eventEDP['floor']
            known = True
        elif eventType == 'max_roof_drift':
            edpAcronym = 'PRD'  # noqa: N806
            floor = '1'
            known = True
        elif eventType == 'residual_disp':
            edpAcronym = 'RD'  # noqa: N806
            floor = eventEDP['floor']
            known = True
        elif eventType == 'max_pressure':
            edpAcronym = 'PSP'  # noqa: N806
            floor = eventEDP['floor2']
            known = True
        elif eventType == 'max_rel_disp':
            edpAcronym = 'PFD'  # noqa: N806
            floor = eventEDP['floor']
            known = True
        elif eventType == 'peak_wind_gust_speed':
            edpAcronym = 'PWS'  # noqa: N806
            floor = eventEDP['floor']
            known = True
        else:
            edpList = [eventType]  # noqa: N806

        if known:
            dofs = eventEDP['dofs']
            scalar_data = []
            for dof in dofs:
                my_edp_name = '1-' + edpAcronym + '-' + floor + '-' + str(dof)
                idscalar = edp_names.index(my_edp_name)
                scalar_data += [edp_vals[idscalar]]
                edpList = [my_edp_name]  # noqa: N806, F841

            eventEDPs[j]['scalar_data'] = scalar_data

    rootEDP['EngineeringDemandParameters'][0].pop(
        'name', ''
    )  # Remove EQ name if exists because it is confusing
    rootEDP['EngineeringDemandParameters'][0]['responses'] = eventEDPs

    with open(newEDP_input_path, 'w', encoding='utf-8') as f:  # noqa: PTH123
        json.dump(rootEDP, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filenameAIM', default=None)
    parser.add_argument('--filenameSAM')
    parser.add_argument('--filenameEVENT')  # not used
    parser.add_argument('--filenameEDP', default=None)
    parser.add_argument('--filenameSIM', default=None)  # not used
    parser.add_argument('--getRV', default=False, nargs='?', const=True)

    args = parser.parse_args()

    if not args.getRV:
        run_surrogateGP(args.filenameAIM, args.filenameEDP)
        write_EDP(args.filenameAIM, args.filenameEDP)
