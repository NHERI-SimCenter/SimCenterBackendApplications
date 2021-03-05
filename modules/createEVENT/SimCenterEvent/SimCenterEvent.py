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
#

import argparse, json, sys, os
import numpy as np
from pathlib import Path

# import the common constants and methods
this_dir = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
main_dir = this_dir.parents[1]

sys.path.insert(0, str(main_dir / 'common'))

from simcenter_common import *


def get_scale_factors(input_unit, output_units, event_class):
    """
    Determine the scale factor to convert input event to internal event data

    """

    # special case: if the input unit is "as-is" then do not do any scaling
    if input_unit != 'as-is':

        if input_unit is not None:
            # if the inputs are not in the SimCenter's standard units

            # get the scale factor to standard units
            f_in = globals().get(input_unit, None)
            if f_in is None:
                raise ValueError(
                    f"Input unit for event files not recognized: {input_unit}")

        else:
            # if the inputs are in the SimCenter's standard units then convert
            # from those to the SI units
            # SimCenter standard units:
            #
            #  ground motion acceleration: g
            #  wind speed: mph
            #  inundation height: ft

            if event_class == 'Earthquake':
                f_in = globals()['g']
            elif event_class == 'Hurricane':
                f_in = globals()['mph']
            elif event_class == 'Flood':
                f_in = globals()['ft']
            else:
                raise ValueError(f"Event class not recognized: {event_class}")

        # if no length unit is specified, 'inch' is assumed
        unit_length = output_units.get('length', 'inch')
        f_length = globals().get(unit_length, None)
        if f_length is None:
            raise ValueError(
                f"Specified length unit not recognized: {unit_length}")

        # if no time unit is specified, 'sec' is assumed
        unit_time = output_units.get('time', 'sec')
        f_time = globals().get(unit_time, None)
        if f_time is None:
            raise ValueError(
                f"Specified time unit not recognized: {unit_time}")

        # the output unit depends on the event class
        if event_class == 'Earthquake':
            # acceleration
            f_out = f_time ** 2.0 / f_length

        elif event_class == 'Hurricane':
            # velocity
            f_out = f_time / f_length

        elif event_class == 'Flood':
            # depth
            f_out = 1.0 / f_length

        else:
            raise ValueError(f"Event class not recognized: {event_class}")

        # the scale factor is the product of input and output scaling
        f_scale = f_in * f_out

    else:
        f_scale = 1.0

    return f_scale

def write_RV(BIM_file, EVENT_file, input_unit):

    # load the BIM file to get information about the assigned events
    with open(BIM_file, 'r') as f:
        bim_data = json.load(f)

    event_class = bim_data['Events']['Events'][0]['EventClassification']

    # scale the input data to the event unit used internally
    f_scale_units = get_scale_factors(input_unit,
        output_units=bim_data['GeneralInformation'].get('units',None),
        event_class=event_class)

    # get the location of the event input files
    data_dir = Path(bim_data['Events']['EventFolderPath'])

    # get the list of events assigned to this asset
    events = bim_data['Events']['Events']

    # initialize the dictionary that will become EVENT.json
    event_file = {
        'randomVariables': [],
        'Events': []
    }

    if len(events) > 1:
        # if there is more than one event then we need random variables

        # initialize the randomVariables part of the EVENT file
        event_file['randomVariables'].append({
            'distribution': 'discrete_design_set_string',
            'name': 'eventID',
            'value': 'RV.eventID',
            'elements': []
        })

        # initialize the Events part of the EVENT file
        event_file['Events'].append({
            'type': 'Seismic',
            'subtype': bim_data['Events']['Events'][0]['type'],
            'event_id': 'RV.eventID',
            'unitScaleFactor': f_scale_units,
            'data_dir': str(data_dir)
            })

        # collect the filenames
        RV_elements = []
        for event in events:
            if event['EventClassification'] in ['Earthquake', 'Hurricane',
                                                'Flood']:
                RV_elements.append(event['fileName'])

        # and add them to the list of randomVariables
        event_file['randomVariables'][0]['elements'] = RV_elements

    else:
        # if there is only one event, then we do not need random variables

        # initialize the Events part of the EVENT file
        event_file['Events'].append({
            'type': 'Seismic',
            'subtype': bim_data['Events']['Events'][0]['type'],
            'event_id': events[0]['fileName'],
            'unitScaleFactor': f_scale_units,
            'data_dir': str(data_dir)
            })

    # if time histories are used, then load the first event
    # TODO: this is needed by some other code that should be fixed and this
    #  part should be removed.
    if events[0]['type'] == 'timeHistory':
        event_file['Events'][0].update(
            load_record(events[0]['fileName'], data_dir,
                        empty=len(events) > 1, event_class = event_class))

    # save the EVENT dictionary to a json file
    with open(EVENT_file, 'w') as f:
        json.dump(event_file, f, indent=2)

def load_record(file_name, data_dir, f_scale=1.0, empty=False,
                event_class=None):

    #just in case
    data_dir = Path(data_dir)

    # extract the file name (the part after "x" is only for bookkeeping)
    file_name = file_name.split('x')[0]

    # open the input event data file
    # (the SimCenter json format is assumed here)
    with open(data_dir / '{}.json'.format(file_name), 'r') as f:
        event_data = json.load(f)

    # initialize the internal EVENT file structure
    event_dic = {
        'name': file_name,
        'dT' : event_data['dT'],
        'numSteps': len(event_data['data_x']),
        'timeSeries': [],
        'pattern': []
    }

    # (empty is used when generating only random variables in write_RV)
    if not empty:

        # generate the event files
        # TODO: add 'z' later
        for i, dir_ in enumerate(['x', 'y']):

            src_label = 'data_'+dir_

            # the target label depends on the event class
            # TODO: it seems much easier to use the data_ labels and keep the
            #  event_class as an attribute in the EVENT file
            # TODO: it also does not make any sense to use capital letters
            #  for the dir in internal labels and small ones for the inputs
            if event_class == 'Earthquake':
                tar_label = 'accel_'+dir_.capitalize()

            elif event_class == 'Hurricane':
                tar_label = 'speed_'+dir_.capitalize()

            elif event_class == 'Flood':
                tar_label = 'height'
            else:
                raise ValueError(f"Event class not recognized: {event_class}")

            # if there is data in the given direction in the input file
            if src_label in event_data.keys():

                # then load that data into the output EVENT file and scale it
                event_dic['timeSeries'].append({
                    'name': tar_label,
                    'type': 'Value',
                    'dT': event_data['dT'],
                    'data': list(np.array(event_data[src_label]) * f_scale)
                })

                event_dic['pattern'].append({
                    'type': 'UniformAcceleration',
                    'timeSeries': tar_label,
                    'dof': i + 1
                })

    return event_dic

def get_records(BIM_file, EVENT_file, input_unit):

    # load the BIM file
    with open(BIM_file, 'r') as f:
        bim_file = json.load(f)

    # load the EVENT file
    with open(EVENT_file, 'r') as f:
        event_file = json.load(f)

    event_class = bim_file['Events']['Events'][0]['EventClassification']

    # get the event_id to identify which event to load
    # (the event id might have been randomly generated earlier)
    event_id = event_file['Events'][0]['event_id']

    # get the scale factor to convert input data to the internal even unit
    f_scale_units = event_file['Events'][0]['unitScaleFactor']

    # get the scale factor if a user specified it
    f_scale_user = dict([(evt['fileName'], evt.get('factor', 1.0))
                         for evt in bim_file["Events"]["Events"]])[event_id]

    # get the location of the event data
    data_dir = Path(bim_file['Events']['EventFolderPath'])

    # load the event data and scale it
    event_file['Events'][0].update(
        load_record(event_id, data_dir, f_scale_user * f_scale_units,
                    event_class = event_class))

    # save the updated EVENT file
    with open(EVENT_file, 'w') as f:
        json.dump(event_file, f, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        "Read event input files (e.g. time history data, intensity measure "
        "fields and convert them to standard SimCenter EVENT.json files",
        allow_abbrev=False
    )

    parser.add_argument('--filenameBIM',
        help = "Name of the BIM file")
    parser.add_argument('--filenameEVENT',
        help = "Name of the EVENT file")
    parser.add_argument('--inputUnit',
        help = "Units of the data in the input file",
        default = None)
    parser.add_argument('--getRV',
        help = "If True, the application prepares on the RandomVariables in "
               "the EVENT file; otherwise it loads the appropriate EVENT data.",
        default=False,
        nargs='?', const=True)

    args = parser.parse_args()

    if args.getRV:
        sys.exit(write_RV(args.filenameBIM, args.filenameEVENT, args.inputUnit))
    else:
        sys.exit(get_records(args.filenameBIM, args.filenameEVENT,
                             args.inputUnit))
