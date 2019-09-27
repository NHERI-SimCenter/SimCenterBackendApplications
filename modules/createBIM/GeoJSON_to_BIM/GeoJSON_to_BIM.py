from __future__ import division, print_function
import os, sys
if sys.version.startswith('2'):
    range=xrange
    string_types = basestring
else:
    string_types = str

import argparse, posixpath, ntpath, json

def create_building_files(output_file, building_source_file, config_file, 
    min_id, max_id):

    # check if the min and max values are provided in the right order
    if (min_id is not None) and (max_id is not None):
        if min_id > max_id:
            tmp = min_id
            min_id = max_id
            max_id = tmp

    with open(config_file, 'r') as f:
        units_data = json.load(f)['units']

    with open(building_source_file, 'r') as f:
        building_source_list = json.load(f)["features"]

    buildings_array = []

    for bldg_src in building_source_list:
        bldg_id = int(bldg_src["id"])

        if (((min_id is not None) and (bldg_id < min_id)) or 
            ((max_id is not None) and (bldg_id > max_id))):
            continue

        BIM_i = {
            "RandomVariables": [],
            "GeneralInformation": dict(
                BIM_id = str(bldg_id),
                geometry = bldg_src["geometry"],
                units = units_data,
                **bldg_src["properties"]
            )
        }

        BIM_file_name = "{}-BIM.json".format(bldg_id)

        with open(BIM_file_name, 'w') as f:
            json.dump(BIM_i, f, indent=2)

        buildings_array.append(dict(id=str(bldg_id), file=BIM_file_name))

    with open(output_file, 'w') as f:
        json.dump(buildings_array, f, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--buildingFile')
    parser.add_argument('--buildingSourceFile')
    parser.add_argument('--configFile', default=None)
    parser.add_argument('--Min', default=None)
    parser.add_argument('--Max', default=None)
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    if args.getRV:
        sys.exit(create_building_files(args.buildingFile, args.buildingSourceFile,
            args.configFile, int(args.Min), int(args.Max)))
    else:
        pass # not used