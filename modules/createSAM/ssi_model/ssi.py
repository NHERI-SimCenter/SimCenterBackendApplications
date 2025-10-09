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
#           Amin Pakzad
# 
# Description:
# implementing SSI simulations to the SIM tab


import argparse
import json
import os
import sys

errFileName = os.path.join(os.getcwd(), 'workflow.err') 


def create_SAM(AIM_file, EVENT_file ,SAM_file):
    print("create_SAM is called")
    print("AIM_file: ", AIM_file)
    print("SAM_file: ", SAM_file)
    SAM_json = {}
    AIM_json = {}
    EVENT_json = {}
    with open(EVENT_file, 'r') as f:
        EVENT_json = json.load(f)
    with open(AIM_file, 'r') as f:
        AIM_json = json.load(f)

    SAM_json = AIM_json['Modeling']
    SAM_json['GeneralInformation'] = AIM_json['GeneralInformation']
    SAM_json['subtype'] = "SSISimulation"


    # {"mainScript": "femoramodel_example.tcl", "type": "OpenSeesInput", "NodeMapping": [], "numStory": -1, "ndm": 3, "ndf": 3, "subType": "FemoraInput", "useDamping": false, "coresPerModel": 10, "randomVar": [{"name": "softMat_vs", "value": 276.9518021028489}, {"name": "softMat_vp_vs_ratio", "value": 1.753312207525559}, {"name": "softMat_rho", "value": 2.049575892485012}], "dampingRatio": 0}
    SAM_json['type'] = "OpenSeesInput"
    SAM_json['mainScript'] = ""  # will be set in the workflow
    SAM_json['NodeMapping'] = AIM_json["Modeling"]["structure_info"].get("NodeMapping", [])
    SAM_json['numStory'] = -1
    SAM_json['ndm'] = AIM_json["Modeling"]["structure_info"].get("ndm", 3)
    SAM_json['ndf'] = AIM_json["Modeling"]["structure_info"].get("ndf", 6)
    SAM_json['subType'] = "SSISimulation"
    SAM_json['useDamping'] = False
    SAM_json['coresPerModel'] = -1 # will be set in the workflow
    SAM_json['randomVar'] = AIM_json["Modeling"].get("randomVar", [])
    SAM_json['dampingRatio'] = 0.0

    with open(SAM_file, 'w') as f:
        json.dump(SAM_json, f, indent=4)
    print("SAM file is created")
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameAIM')
    parser.add_argument('--filenameEVENT') 
    parser.add_argument('--filenameSAM')
    parser.add_argument('--mainScript')
    parser.add_argument('--getRV', nargs='?', const=True, default=False) 
    args = parser.parse_args()
    print(args)
    sys.exit(create_SAM(args.filenameAIM, args.filenameEVENT , args.filenameSAM))




