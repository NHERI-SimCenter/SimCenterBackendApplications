import argparse  # noqa: CPY001, D100, INP001
import json
import posixpath
import sys

import numpy as np


def write_RV(BIM_file, EVENT_file, data_dir):  # noqa: N802, N803, D103
    with open(BIM_file) as f:  # noqa: PLW1514, PTH123
        bim_data = json.load(f)

    event_file = {'randomVariables': [], 'Events': []}

    events = bim_data['Events']['Events']

    if len(events) > 1:
        event_file['randomVariables'].append(
            {
                'distribution': 'discrete_design_set_string',
                'name': 'eventID',
                'value': 'RV.eventID',
                'elements': [],
            }
        )
        event_file['Events'].append(
            {'type': 'Seismic', 'subtype': 'SW4_Event', 'event_id': 'RV.eventID'}
        )
    else:
        event_file['Events'].append(
            {
                'type': 'Seismic',
                'subtype': 'SW4_Event',
                'event_id': 0,
            }
        )

    RV_elements = []  # noqa: N806
    for event in events:
        if event['EventClassification'] == 'Earthquake':
            RV_elements.append(event['fileName'])  # noqa: PERF401

    event_file['randomVariables'][0]['elements'] = RV_elements

    # load the first event
    event_file['Events'][0].update(
        load_record(events[0]['fileName'], data_dir, empty=True)
    )

    with open(EVENT_file, 'w') as f:  # noqa: PLW1514, PTH123
        json.dump(event_file, f, indent=2)


def load_record(fileName, data_dir, scale_factor=1.0, empty=False):  # noqa: FBT002, N803, D103
    fileName = fileName.split('x')[0]  # noqa: N806

    with open(posixpath.join(data_dir, f'{fileName}.json')) as f:  # noqa: PLW1514, PTH123
        event_data = json.load(f)

    event_dic = {
        'name': fileName,
        'dT': event_data['dT'],
        'numSteps': len(event_data['data_x']),
        'timeSeries': [],
        'pattern': [],
    }

    if not empty:
        for i, (src_label, tar_label) in enumerate(
            zip(['data_x', 'data_y'], ['accel_X', 'accel_Y'])
        ):
            if src_label in event_data.keys():  # noqa: SIM118
                event_dic['timeSeries'].append(
                    {
                        'name': tar_label,
                        'type': 'Value',
                        'dT': event_data['dT'],
                        'data': list(np.array(event_data[src_label]) * scale_factor),
                    }
                )
                event_dic['pattern'].append(
                    {
                        'type': 'UniformAcceleration',
                        'timeSeries': tar_label,
                        'dof': i + 1,
                    }
                )

    return event_dic


def get_records(BIM_file, EVENT_file, data_dir):  # noqa: N803, D103
    with open(BIM_file) as f:  # noqa: PLW1514, PTH123
        bim_file = json.load(f)

    with open(EVENT_file) as f:  # noqa: PLW1514, PTH123
        event_file = json.load(f)

    event_id = event_file['Events'][0]['event_id']

    scale_factor = dict(  # noqa: C404
        [
            (evt['fileName'], evt.get('factor', 1.0))
            for evt in bim_file['Events']['Events']
        ]
    )[event_id]

    event_file['Events'][0].update(load_record(event_id, data_dir, scale_factor))

    with open(EVENT_file, 'w') as f:  # noqa: PLW1514, PTH123
        json.dump(event_file, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameAIM')
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--pathSW4results')
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    if args.getRV:
        sys.exit(write_RV(args.filenameAIM, args.filenameEVENT, args.pathSW4results))
    else:
        sys.exit(
            get_records(args.filenameAIM, args.filenameEVENT, args.pathSW4results)
        )
