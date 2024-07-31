import os  # noqa: INP001, D100
import sys

if sys.version.startswith('2'):
    range = xrange  # noqa: A001, F821
    string_types = basestring  # noqa: F821
else:
    string_types = str

import argparse
import json


def write_RV(AIM_input_path, EDP_input_path, EDP_type):  # noqa: ANN001, ANN201, ARG001, N802, N803, D103
    # load the AIM file
    with open(AIM_input_path, encoding='utf-8') as f:  # noqa: PTH123
        root_AIM = json.load(f)  # noqa: N806

    #
    # Is this the correct application
    #

    if (
        root_AIM['Applications']['Modeling']['Application']
        != 'SurrogateGPBuildingModel'
    ):
        with open('../workflow.err', 'w') as f:  # noqa: PTH123
            f.write(
                'Do not select [None] in the EDP tab. [None] is used only when using pre-trained surrogate, i.e. when [Surrogate] is selected in the SIM Tab.'  # noqa: E501
            )
        exit(-1)  # noqa: PLR1722

    #
    # Get EDP info from surrogate model file
    #

    print('General Information tab is ignored')  # noqa: T201
    root_SAM = root_AIM['Applications']['Modeling']  # noqa: N806

    surrogate_path = os.path.join(  # noqa: PTH118
        root_SAM['ApplicationData']['MS_Path'],
        root_SAM['ApplicationData']['mainScript'],
    )
    print(surrogate_path)  # noqa: T201

    with open(surrogate_path, encoding='utf-8') as f:  # noqa: PTH123
        surrogate_model = json.load(f)

    root_EDP = surrogate_model['EDP']  # noqa: N806

    # if it is surrogate,
    # Save Load EDP.json from standard surrogate models and write it to EDP
    """
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
    """  # noqa: W291
    with open(EDP_input_path, 'w', encoding='utf-8') as f:  # noqa: PTH123
        json.dump(root_EDP, f, indent=2)


def create_EDP(AIM_input_path, EDP_input_path, EDP_type):  # noqa: ANN001, ANN201, ARG001, N802, N803, D103
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
