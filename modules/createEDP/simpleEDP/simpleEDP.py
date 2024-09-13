import sys  # noqa: INP001, D100

if sys.version.startswith('2'):
    range = xrange  # noqa: A001, F821
    string_types = basestring  # noqa: F821
else:
    string_types = str

import argparse
import json


def write_RV(AIM_input_path, EDP_input_path, EDP_type):  # noqa: N802, N803, D103
    # load the AIM file
    with open(AIM_input_path, encoding='utf-8') as f:  # noqa: PTH123
        AIM_in = json.load(f)  # noqa: N806

    EDP_list = []  # noqa: N806
    if 'EDP' in AIM_in.keys():  # noqa: SIM118
        for edp in AIM_in['EDP']:
            EDP_list.append(  # noqa: PERF401
                {
                    'type': edp['type'],
                    'cline': edp.get('cline', '1'),
                    'floor': edp.get('floor', '1'),
                    'dofs': edp.get(
                        'dofs',
                        [
                            1,
                        ],
                    ),
                    'scalar_data': [],
                }
            )
    else:
        EDP_list.append(
            {
                'type': EDP_type,
                'cline': '1',
                'floor': '1',
                'dofs': [
                    1,
                ],
                'scalar_data': [],
            }
        )

    EDP_json = {  # noqa: N806
        'RandomVariables': [],
        'total_number_edp': len(EDP_list),
        'EngineeringDemandParameters': [
            {'responses': EDP_list},
        ],
    }

    with open(EDP_input_path, 'w') as f:  # noqa: PTH123
        json.dump(EDP_json, f, indent=2)


def create_EDP(AIM_input_path, EDP_input_path, EDP_type):  # noqa: ARG001, N802, N803, D103
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
