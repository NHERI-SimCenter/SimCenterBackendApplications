from __future__ import division, print_function
import os, sys

if sys.version.startswith('2'):
    range = xrange
    string_types = basestring
else:
    string_types = str

import argparse, posixpath, ntpath, json


def write_RV(BIM_input_path, EVENT_input_path):
    # create the empty EVENT.json file
    EVENT_in = {'Events': []}

    with open(EVENT_input_path, 'w') as f:
        json.dump(EVENT_in, f, indent=2)

    # TODO: if there are multiple events, we need to create a random variable for them


def get_windspeed(BIM_input_path, EVENT_input_path, wind_database_path, severity):
    sys.path.insert(0, os.getcwd())

    # load the BIM file
    with open(BIM_input_path, 'r') as f:
        BIM_in = json.load(f)

    # load the EVENT file
    with open(EVENT_input_path, 'r') as f:
        EVENT_in = json.load(f)

    # if there is a wind database path provided
    if wind_database_path is not None:
        # then we need to load the wind data from there
        with open(wind_database_path, 'r') as f:
            wind_db = json.load(f)

        # the location id is stored in the BIM file
        location_id = BIM_in['GeneralInformation']['id']

        for event in wind_db:
            if event['id'] == location_id:
                wind_speed_in = event
                break

    # if there is no wind database, then a single wind input file is expected
    else:
        # load the file with the wind speeds
        for event in BIM_in['Events']:
            if (event['EventClassification'] == 'Wind') and (
                event['Events'][0]['type'] == 'ASCE7_WindSpeed'
            ):
                event_info = event['Events'][0]
                with open(event_info['fileName'], 'r') as f:
                    wind_speed_in = json.load(f)

    event_id = wind_speed_in['id']

    if severity is None:
        severity = event_info['severity']

    for wind_data in wind_speed_in['atcHazardData']['WindData']['datasets']:
        if wind_data['name'] == severity:
            event_data = wind_data
            break

    event_json = {
        'type': 'Wind',
        'subtype': 'ASCE7_WindSpeed',
        'index': event_id,
        'peak_wind_gust_speed': event_data['data']['value'],
        'unit': event_data['unit'],
    }
    EVENT_in['Events'].append(event_json)

    with open(EVENT_input_path, 'w') as f:
        json.dump(EVENT_in, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenameAIM')
    parser.add_argument('--filenameEVENT')
    parser.add_argument('--windDatabase', default=None)
    parser.add_argument('--severity', default=None)
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    if args.getRV:
        sys.exit(write_RV(args.filenameAIM, args.filenameEVENT))
    else:
        sys.exit(
            get_windspeed(
                args.filenameAIM,
                args.filenameEVENT,
                args.windDatabase,
                args.severity,
            )
        )
