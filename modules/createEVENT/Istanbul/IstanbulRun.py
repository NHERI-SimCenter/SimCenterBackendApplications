import argparse  # noqa: INP001, D100
import os

import IstanbulApp2
import IstanbulStations

if __name__ == '__main__':
    information = {
        'RegionFlag': False,
        'LocationFlag': True,
        'TopoFlag': True,
        'BedrockFlag': True,
        'RegionShape': 'Rectangle',
        'min_lat': 40.9940,
        'max_lat': 40.9945,
        'min_lon': 28.8985,
        'max_lon': 28.8995,
        'lat': 40.9940,
        'lon': 28.8990,
        'directory': 'Events',
        'number_of_realizations': 2,
    }

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
    args = parser.parse_args()

    if args.lat:
        information['lat'] = float(args.lat)
    if args.lng:
        information['lon'] = float(args.lng)
    if args.output:
        information['directory'] = args.output
    if args.number:
        information['number_of_realizations'] = int(args.number)

    # change the directory to the file location
    os.chdir(os.path.dirname(os.path.realpath(__file__)))  # noqa: PTH120
    IstanbulStations.getStations(information, plot=False, show=False)
    IstanbulApp2.Submit_tapis_job()
    exit()  # noqa: PLR1722
