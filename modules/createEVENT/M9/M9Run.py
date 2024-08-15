import argparse
import os

import M9API
import M9App2
import M9Stations

if __name__ == '__main__':
    information = {
        'LocationFlag': True,
        'RegionFlag': False,
        'APIFLAG': False,
        'lat': 47.65290010591034,
        'lon': -122.30531923052669,
        'RegionShape': 'Circle',
        'min_lat': 47.58,
        'max_lat': 47.62,
        'min_lon': -122.38,
        'max_lon': -122.34,
        'radius': 10,
        'grid_type': 'A',
        'directory': 'Events',
        'number_of_realizations': 1,
    }

    #
    # create a parser to parse input args & update default information struct
    #

    parser = argparse.ArgumentParser()
    parser.add_argument('--lat', help='Latitude', required=False)
    parser.add_argument('--lng', help='Longitude', required=False)
    parser.add_argument('-g', '--gridType', help='grid Type', required=False)
    parser.add_argument(
        '-n', '--number', help='number of realizations', required=False
    )
    parser.add_argument(
        '-o', '--output', help='number of realizations', required=False
    )
    parser.add_argument('-a', '--API', help='API FLAG', required=False)
    args = parser.parse_args()

    if args.lat:
        information['lat'] = float(args.lat)
    if args.lng:
        information['lon'] = float(args.lng)
    if args.output:
        information['directory'] = args.output
    if args.number:
        information['number_of_realizations'] = int(args.number)
    if args.gridType:
        information['grid_type'] = args.gridType
    if args.API == 'true':
        information['APIFLAG'] = True

    #
    # go get the motions
    #
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    if information['APIFLAG']:
        print(
            'Using API for extracting motions:\n This may take a while. Please be patient.'
        )
        M9API.M9(information)
    else:
        M9Stations.getStations(information, plot=False, show=False)
        M9App2.Submit_tapis_job()
    exit()
