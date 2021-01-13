import sys, argparse
import posixpath, json

def create_building_files(output_file, building_source_file, bldg_filter):

    # these imports are here to save time when the app is called without
    # the -getRV flag
    import numpy as np
    import pandas as pd

    # get the units
    main_dir = posixpath.dirname(output_file)
    with open(posixpath.join(main_dir, 'units.json'), 'r') as f:
        units = json.load(f)

    # check if a filter is provided
    if bldg_filter is not None:
        bldgs_requested = []
        for bldgs in bldg_filter.split(','):
            if "-" in bldgs:
                bldg_low, bldg_high = bldgs.split("-")
                bldgs_requested += list(range(int(bldg_low), int(bldg_high)+1))
            else:
                bldgs_requested.append(int(bldgs))
        bldgs_requested = np.array(bldgs_requested)

    # check if the min and max values are provided in the right order
    # if (min_id is not None) and (max_id is not None):
    #     if min_id > max_id:
    #         tmp = min_id
    #         min_id = max_id
    #         max_id = tmp

    # load the CSV file with the building information
    bldgs_df = pd.read_csv(building_source_file, header=0, index_col=0)

    # if there is a filter, then pull out only the required buildings
    if bldg_filter is not None:
        bldgs_available = bldgs_df.index.values
        bldgs_to_run = bldgs_requested[
            np.where(np.in1d(bldgs_requested, bldgs_available))[0]]
        selected_bldgs = bldgs_df.loc[bldgs_to_run]
    else:
        selected_bldgs = bldgs_df

    # # get the min and max ids in the input data
    # bldg_ids_min = np.min(bldgs_df.index.values)
    # bldg_ids_max = np.max(bldgs_df.index.values)
    #
    # if min_id is None: min_id = bldg_ids_min
    # if max_id is None: max_id = bldg_ids_max
    #
    # min_id = np.max([bldg_ids_min, min_id])
    # max_id = np.min([bldg_ids_max, max_id])
    #
    # # select the slice defined by the min max constraints
    # selected_bldgs = bldgs_df.loc[min_id:max_id, :]

    # identify the labels
    labels = selected_bldgs.columns.values

    buildings_array = []

    # for each building...
    for bldg_id, bldg in selected_bldgs.iterrows():

        # initialize the BIM file
        BIM_i = {
            "RandomVariables": [],
            "GeneralInformation": dict(
                BIM_id = str(int(bldg_id)),
                location = {
                    'latitude': bldg["Latitude"],
                    'longitude': bldg["Longitude"]
                },
                units           = units
            )
        }

        # save every label as-is
        for label in labels:
            BIM_i["GeneralInformation"].update({label: bldg[label]})

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
    parser.add_argument('--filter', default=None)
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    if args.getRV:
        sys.exit(create_building_files(args.buildingFile,
                                       args.buildingSourceFile, args.filter))
    else:
        pass # not used
