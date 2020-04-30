import os, sys, argparse, posixpath, ntpath, json

type_id_to_occupancy = {
	0: 'Other/Unknown',
    1: 'Residential - Single-Family',
    2: 'Residential - Town-Home',
    3: 'Residential - Multi-Family',
    4: 'Office',
    5: 'Hotel',
    6: 'School',
    7: 'Industrial - Light',
    8: 'Industrial - Warehouse',
    9: 'Industrial - Heavy',
    10: 'Retail',
    11: 'Retail',
    12: 'Residential - Mixed Use',
    13: 'Retail',
    14: 'Office',
    15: 'Parking',
    16: 'Parking'
}

def create_building_files(output_file, building_source_file, min_id, max_id):
	
	import numpy as np
	import pandas as pd

	# check if the min and max values are provided in the right order
	if (min_id is not None) and (max_id is not None):
	    if min_id > max_id:
	        tmp = min_id
	        min_id = max_id
	        max_id = tmp	
	        
	buildings_array = []

	bldgs_df = pd.read_csv(building_source_file, header=0, index_col=0)
	bldg_ids_min = np.min(bldgs_df.index.values)
	bldg_ids_max = np.max(bldgs_df.index.values)

	if min_id is None: min_id = bldg_ids_min
	if max_id is None: max_id = bldg_ids_max

	min_id = np.max([bldg_ids_min, min_id])
	max_id = np.min([bldg_ids_max, max_id])

	selected_bldgs = bldgs_df.loc[min_id:max_id, :]

	for bldg_id, bldg in selected_bldgs.iterrows():

		if bldg['Stories'] == 1:
			height = 4.66 # m
		elif bldg['Structure Type'][0] in ['C', 'P', 'R', 'U', 'M']:
			height = bldg['Stories'] * 3.33 # m
		else:
			height = bldg['Stories'] * 4.0 # m

		BIM_i = {
		    "RandomVariables": [],
		    "GI": dict(
		        BIM_id = str(bldg_id),
		        location = {
		            'latitude': bldg.Latitude,
		            'longitude': bldg.Longitude
		        },
		        assetType       = 'Building',
		        yearBuilt       = bldg['Year Built'],
		        structType      = bldg['Structure Type'],
		        name            = bldg_id,
		        area            = bldg['Area'] * 0.09290304,
		        numStory        = bldg['Stories'],
		        occupancy       = type_id_to_occupancy[bldg['Type ID']],
		        height          = height,
		        replacementCost = bldg['Replacement Cost'],
		        replacementTime = 1.0 
		    )
		}

		if 'Damping' in bldg.keys():
			BIM_i["GI"].update({'dampingRatio': bldg['Damping']})

		if 'PGA target' in bldg.keys():
			BIM_i["GI"].update({'PGA_target': bldg['PGA target']})

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
    parser.add_argument('--Min', default=None)
    parser.add_argument('--Max', default=None)
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    if args.getRV:
        sys.exit(create_building_files(args.buildingFile, args.buildingSourceFile,
            int(args.Min), int(args.Max)))
    else:
        pass # not used