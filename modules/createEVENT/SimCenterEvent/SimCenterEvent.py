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
#

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# import the common constants and methods
this_dir = Path(os.path.dirname(os.path.abspath(__file__))).resolve()  # noqa: PTH100, PTH120
main_dir = this_dir.parents[1]

sys.path.insert(0, str(main_dir / 'common'))

from simcenter_common import get_scale_factors, get_unit_bases  # noqa: E402


def write_RV(AIM_file, EVENT_file):  # noqa: N802, N803, D103
    # load the AIM file to get information about the assigned events
    with open(AIM_file, encoding='utf-8') as f:  # noqa: PTH123
        aim_file = json.load(f)

    input_units = None
    if 'RegionalEvent' in aim_file.keys():  # noqa: SIM118
        input_units = aim_file['RegionalEvent'].get('units', None)

    output_units = aim_file.get('units', None)

    # scale the input data to the event unit used internally
    f_scale_units = get_scale_factors(input_units, output_units)

    # get input unit bases
    input_unit_bases = get_unit_bases(input_units)

    # get the location of the event input files
    # TODO: assuming a single event for now  # noqa: TD002
    aim_event_input = aim_file['Events'][0]

    if 'SimCenterEvent' in aim_event_input:
        simcenter_event = aim_event_input['SimCenterEvent']
        data_dir = Path(simcenter_event['EventFolderPath'])
        # Get the number of realizations
        if 'intensityMeasure' in simcenter_event:
            num_of_realizations = len(simcenter_event['intensityMeasure']['Events'][0])
            if 'timeHistory' in simcenter_event:
                if num_of_realizations != len(simcenter_event['timeHistory']['Events'][0]):
                    msg = 'Number of realizations in intensityMeasure and timeHistory do not match'
                    raise ValueError(msg)
        elif 'timeHistory' in simcenter_event:
            num_of_realizations = len(simcenter_event['timeHistory']['Events'][0])
        else:
            msg = 'No intensityMeasure or timeHistory in SimCenterEvent'
            raise ValueError(msg)

        # currently assume only intensityMeasure or timeHistory
        if 'intensityMeasure' in simcenter_event:
            event_type = 'intensityMeasure'
        elif 'timeHistory' in simcenter_event:
            event_type = 'timeHistory'
        else:
            msg = 'No intensityMeasure or timeHistory in SimCenterEvent'
            raise ValueError(msg)

        event_file = {'randomVariables': [], 'Events': []}
        if num_of_realizations > 1:
            # if there is more than one event then we need random variables

            # initialize the randomVariables part of the EVENT file
            if event_type == 'intensityMeasure':
                event_file['randomVariables'].append(
                    {
                        'distribution': 'discrete_design_set_string',
                        'name': 'eventID',
                        'value': 'RV.eventID',
                        'elements': ['event_' + str(i) for i in range(num_of_realizations)],
                    }
                )

                # initialize the Events part of the EVENT file
                event_file['Events'].append(
                    {
                    'type': event_type,
                    'event_id': 'RV.eventID',
                    'unitScaleFactor': f_scale_units,
                    'units': input_unit_bases,
                    'values': simcenter_event[event_type]['Events'],
                    'labels': simcenter_event[event_type]['Labels'],
                    }
                )
            elif event_type == 'timeHistory':
                event_file['randomVariables'].append(
                    {
                        'distribution': 'discrete_design_set_string',
                        'name': 'eventID',
                        'value': 'RV.eventID',
                        'elements': [],
                    }
                )

                # initialize the Events part of the EVENT file
                event_file['Events'].append(
                    {
                        # 'type': 'Seismic', I am pretty sure we are not using this now
                        # or we are using it incorrectly, so I removed it for the time being
                        # and replaced it with the information that is actually used
                        'type': aim_event_input['type'],
                        'event_id': 'RV.eventID',
                        'unitScaleFactor': f_scale_units,
                        'units': input_unit_bases,
                        'data_dir': str(data_dir),
                    }
                )

                # collect the filenames
                RV_elements = simcenter_event[event_type]['Events'][0]  # noqa: N806
                # for event in events:
                #    #if event['EventClassification'] in ['Earthquake', 'Hurricane',
                #    #                                    'Flood']:
                #    #RV_elements.append(event['fileName'])
                #    RV_elements.append(event[0])

                # and add them to the list of randomVariables
                event_file['randomVariables'][0]['elements'] = RV_elements

                # if time histories are used, then load the first event
                # TODO: this is needed by some other code that should be fixed and this  # noqa: TD002
                #  part should be removed.
                event_file['Events'][0].update({'scale_factors': simcenter_event[event_type]['ScaleFactors']})
                event_file['Events'][0].update(
                    load_record(simcenter_event[event_type]['Events'][0][0], data_dir, empty=num_of_realizations > 1)
                )
                # , event_class = event_class))

        else:
            # if there is only one event, then we do not need random variables

            # initialize the Events part of the EVENT file
            # The events are now two dimensiontal list. The first dimension is sequence of events
            # The second dimension is different grid in the same event

            event_file['Events'].append(
                {
                    # 'type': 'Seismic',
                    'type': event_type,
                    'event_id': simcenter_event[event_type]['Events'][0][0],
                    'unitScaleFactor': f_scale_units,
                    'units': input_unit_bases,
                    'data_dir': str(data_dir),
                }
            )


        # save the EVENT dictionary to a json file
        with open(EVENT_file, 'w', encoding='utf-8') as f:  # noqa: PTH123
            json.dump(event_file, f, indent=2)

        return
    # Below are for backward compatibility
    data_dir = Path(aim_event_input['EventFolderPath'])
    # get the list of events assigned to this asset
    events = aim_event_input['Events']

    # initialize the dictionary that will become EVENT.json
    event_file = {'randomVariables': [], 'Events': []}

    if len(events) > 1:
        # if there is more than one event then we need random variables

        # initialize the randomVariables part of the EVENT file
        event_file['randomVariables'].append(
            {
                'distribution': 'discrete_design_set_string',
                'name': 'eventID',
                'value': 'RV.eventID',
                'elements': [],
            }
        )

        # initialize the Events part of the EVENT file
        event_file['Events'].append(
            {
                # 'type': 'Seismic', I am pretty sure we are not using this now
                # or we are using it incorrectly, so I removed it for the time being
                # and replaced it with the information that is actually used
                'type': aim_event_input['type'],
                'event_id': 'RV.eventID',
                'unitScaleFactor': f_scale_units,
                'units': input_unit_bases,
                'data_dir': str(data_dir),
            }
        )

        # collect the filenames
        RV_elements = np.array(events).T[0].tolist()  # noqa: N806
        # for event in events:
        #    #if event['EventClassification'] in ['Earthquake', 'Hurricane',
        #    #                                    'Flood']:
        #    #RV_elements.append(event['fileName'])
        #    RV_elements.append(event[0])

        # and add them to the list of randomVariables
        event_file['randomVariables'][0]['elements'] = RV_elements

    else:
        # if there is only one event, then we do not need random variables

        # initialize the Events part of the EVENT file
        event_file['Events'].append(
            {
                # 'type': 'Seismic',
                'type': aim_event_input['type'],
                'event_id': events[0][0],
                'unitScaleFactor': f_scale_units,
                'units': input_unit_bases,
                'data_dir': str(data_dir),
            }
        )

    # if time histories are used, then load the first event
    # TODO: this is needed by some other code that should be fixed and this  # noqa: TD002
    #  part should be removed.

    if aim_event_input['type'] == 'timeHistory':
        event_file['Events'][0].update(
            load_record(events[0][0], data_dir, empty=len(events) > 1)
        )
        # , event_class = event_class))

    # save the EVENT dictionary to a json file
    with open(EVENT_file, 'w', encoding='utf-8') as f:  # noqa: PTH123
        json.dump(event_file, f, indent=2)


def load_record(  # noqa: D103
    file_name,
    data_dir,
    f_scale_user=1.0,
    f_scale_units={'ALL': 1.0},  # noqa: B006
    empty=False,  # noqa: FBT002
):
    # event_class=None):

    # just in case
    data_dir = Path(data_dir)

    # extract the file name (the part after "x" is only for bookkeeping)
    file_name = file_name.split('x')[0]

    # open the input event data file
    # (SimCenter json format is assumed here)
    with open(data_dir / f'{file_name}.json', encoding='utf-8') as f:  # noqa: PTH123
        event_data = json.load(f)

    # check if Event File is already in EVENT format
    isEventFile = False  # noqa: N806
    if event_data.__contains__('Events'):
        event_dic = event_data['Events'][0]
        # event_dic['dT'] = event_data['Events'][0]['dT']
        # event_dic['numSteps'] = event_data['Events'][0]['numSteps']
        # event_dic['timeSeries'] = event_data['Events'][0]['timeSeries']
        # event_dic['pattern'] = event_data['Events'][0]['pattern']
        return event_dic  # noqa: RET504

        isEventFile = True  # noqa: N806

    else:  # noqa: RET505
        # initialize the internal EVENT file structure
        event_dic = {
            'name': file_name,
            'dT': event_data['dT'],
            'numSteps': len(event_data['data_x']),
            'timeSeries': [],
            'pattern': [],
        }

    if not isEventFile:
        f_scale_units = f_scale_units.get('TH_file', f_scale_units.get('ALL', None))
        if f_scale_units is None:
            raise ValueError('No unit scaling is defined for time history data.')  # noqa: EM101, TRY003

        f_scale = float(f_scale_units) * float(f_scale_user)

        # generate the event files
        # TODO: add 'z' later  # noqa: TD002
        for i, dir_ in enumerate(['x', 'y']):
            src_label = 'data_' + dir_
            tar_label = src_label

            # if there is data in the given direction in the input file
            if src_label in event_data.keys():  # noqa: SIM118
                # then load that data into the output EVENT file and scale it
                event_dic['timeSeries'].append(
                    {
                        'name': tar_label,
                        'type': 'Value',
                        'dT': event_data['dT'],
                        'data': list(np.array(event_data[src_label]) * f_scale),
                    }
                )

                # (empty is used when generating only random variables in write_RV)
                if empty:
                    event_dic['timeSeries'][-1]['data'] = []

                # TODO: We will need to generalize this as soon as we add  # noqa: TD002
                # different types of time histories
                # Assuming acceleration time history for now.
                event_dic['pattern'].append(
                    {
                        'type': 'UniformAcceleration',
                        'timeSeries': tar_label,
                        'dof': i + 1,
                    }
                )

    return event_dic


def get_records(AIM_file, EVENT_file):  # noqa: N803
    """This function is only called if UQ is part of the workflow. That is, it is
    not called if we are using IMasEDP and skipping the response simulation.

    """  # noqa: D205, D401, D404
    # load the AIM file
    with open(AIM_file, encoding='utf-8') as f:  # noqa: PTH123
        AIM_file = json.load(f)  # noqa: N806

    # load the EVENT file
    with open(EVENT_file, encoding='utf-8') as f:  # noqa: PTH123
        event_file = json.load(f)

    # event_class = AIM_file['Events']['Events'][0]['EventClassification']

    # get the event_id to identify which event to load
    # (the event id might have been randomly generated earlier)
    event_id = event_file['Events'][0]['event_id']

    # get the scale factors to convert input data to the internal event unit
    f_scale_units = event_file['Events'][0]['unitScaleFactor']

    # get the scale factor if a user specified it

    event_data = np.array(AIM_file['Events'][0]['Events']).T
    event_loc = np.where(event_data == event_id)[1][0]
    f_scale_user = event_data.T[event_loc][1]

    # f_scale_user = dict([(evt['fileName'], evt.get('factor', 1.0))
    #                     for evt in AIM_file["Events"]["Events"]])[event_id]

    # get the location of the event data
    data_dir = Path(AIM_file['Events'][0]['EventFolderPath'])

    # load the event data and scale it
    event_file['Events'][0].update(
        load_record(event_id, data_dir, f_scale_user, f_scale_units)
    )  # , event_class = event_class))

    # save the updated EVENT file
    with open(EVENT_file, 'w', encoding='utf-8') as f:  # noqa: PTH123
        json.dump(event_file, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Read event input files (e.g. time history data, intensity measure '
        'fields and convert them to standard SimCenter EVENT.json files',
        allow_abbrev=False,
    )

    parser.add_argument('--filenameAIM', help='Name of the AIM file')
    parser.add_argument('--filenameEVENT', help='Name of the EVENT file')
    parser.add_argument(
        '--inputUnit', help='Units of the data in the input file', default=None
    )
    parser.add_argument(
        '--getRV',
        help='If True, the application prepares on the RandomVariables in '
        'the EVENT file; otherwise it loads the appropriate EVENT data.',
        default=False,
        nargs='?',
        const=True,
    )

    args = parser.parse_args()

    if args.getRV:
        sys.exit(write_RV(args.filenameAIM, args.filenameEVENT))
    else:
        sys.exit(get_records(args.filenameAIM, args.filenameEVENT))
