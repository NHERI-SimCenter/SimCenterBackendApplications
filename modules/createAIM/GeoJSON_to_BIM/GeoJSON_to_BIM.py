import argparse
import json
import sys


def create_building_files(output_file, building_source_file, min_id, max_id):
    # check if the min and max values are provided in the right order
    if (min_id is not None) and (max_id is not None):
        if min_id > max_id:
            tmp = min_id
            min_id = max_id
            max_id = tmp

    with open(building_source_file, encoding='utf-8') as f:
        building_source_list = json.load(f)['features']

    buildings_array = []

    for bldg_src in building_source_list:
        bldg_id = int(bldg_src['id'])

        if ((min_id is not None) and (bldg_id < min_id)) or (
            (max_id is not None) and (bldg_id > max_id)
        ):
            continue

        bldg_loc = bldg_src['geometry']['coordinates']

        BIM_i = {
            'RandomVariables': [],
            'GI': dict(
                BIM_id=str(bldg_id),
                location={'latitude': bldg_loc[1], 'longitude': bldg_loc[0]},
                **bldg_src['properties'],
            ),
        }

        BIM_file_name = f'{bldg_id}-BIM.json'

        with open(BIM_file_name, 'w', encoding='utf-8') as f:
            json.dump(BIM_i, f, indent=2)

        buildings_array.append(dict(id=str(bldg_id), file=BIM_file_name))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(buildings_array, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--buildingFile')
    parser.add_argument('--buildingSourceFile')
    parser.add_argument('--Min', default=None)
    parser.add_argument('--Max', default=None)
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    if args.getRV:
        sys.exit(
            create_building_files(
                args.buildingFile,
                args.buildingSourceFile,
                int(args.Min),
                int(args.Max),
            )
        )
    else:
        pass  # not used
