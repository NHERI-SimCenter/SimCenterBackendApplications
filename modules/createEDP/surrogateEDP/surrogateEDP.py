from __future__ import division, print_function
import os, sys
if sys.version.startswith('2'):
    range=xrange
    string_types = basestring
else:
    string_types = str

import argparse, posixpath, ntpath, json

def write_RV(AIM_input_path, EDP_input_path, EDP_type):
    
    # load the AIM file
    with open(AIM_input_path, 'r') as f:
        root_AIM = json.load(f)

    #
    # Is this the correct application
    #

    if root_AIM["Applications"]["Modeling"]["Application"] != "SurrogateGPBuildingModel":
            with open("../workflow.err","w") as f:
                f.write("Do not select [None] in the EDP tab. [None] is used only when using pre-trained surrogate, i.e. when [Surrogate] is selected in the SIM Tab.")
            exit(-1)

    #
    # Get EDP info from surrogate model file
    #

    print("General Information tab is ignored")
    root_SAM = root_AIM['Applications']['Modeling']

    surrogate_path = os.path.join(root_SAM['ApplicationData']['MS_Path'], root_SAM['ApplicationData']['mainScript'])
    print(surrogate_path)

    with open(surrogate_path, 'r') as f:
        surrogate_model = json.load(f)

    root_EDP = surrogate_model['EDP']



    # if it is surrogate,
    # Save Load EDP.json from standard surrogate models and write it to EDP
    '''
    EDP_list = []
    if "EDP" in AIM_in.keys():
        for edp in AIM_in["EDP"]:
            EDP_list.append({
                "type": edp["type"],
                "cline": edp.get("cline", "1"),
                "floor": edp.get("floor", "1"),
                "dofs": edp.get("dofs", [1,]),  
                "scalar_data": [],         
            })
    else:
        EDP_list.append({
            "type": EDP_type,
            "cline": "1",
            "floor": "1",
            "dofs": [1,],
            "scalar_data": [],         
        })

    EDP_json = {
        "RandomVariables": [],
        "total_number_edp": len(EDP_list),
        "EngineeringDemandParameters": [{
            "responses": EDP_list
        },]
    }
    '''
    with open(EDP_input_path, 'w') as f:
        json.dump(root_EDP, f, indent=2)

def create_EDP(AIM_input_path, EDP_input_path, EDP_type):
    pass   

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameAIM', default=None)
    parser.add_argument('--filenameSAM', default=None)
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--filenameEDP')
    parser.add_argument('--type')
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    if args.getRV:
        sys.exit(write_RV(args.filenameAIM, args.filenameEDP, args.type))
    else:
        sys.exit(create_EDP(args.filenameAIM, args.filenameEDP, args.type))
