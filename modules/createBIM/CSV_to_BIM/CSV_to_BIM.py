import os, sys, argparse, posixpath, ntpath, json

# conversion from UrbanSim Type ID to Occupancy type
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

# units
# distance, area, volume
m = 1.

mm = 0.001 * m
cm = 0.01 * m
km = 1000. * m

inch = 0.0254
ft = 12. * inch
mile = 5280. * ft

# area
m2 = m**2.

mm2 = mm**2.
cm2 = cm**2.
km2 = km**2.

inch2 = inch**2.
ft2 = ft**2.
mile2 = mile**2.

# volume
m3 = m**3.

inch3 = inch**3.
ft3 = ft**3.


def get_label(options, labels, label_name):

	for option in options:
		if option in labels:
			return option

	print(f'ERROR: Could not identify the label for the {label_name}')

def create_building_files(output_file, building_source_file, min_id, max_id):
	
	import numpy as np
	import pandas as pd

	# get the units
	main_dir = posixpath.dirname(output_file)
	with open(posixpath.join(main_dir, 'units.json'), 'r') as f:
		units = json.load(f)

	if units['length'] == 'in':
		units['length'] = 'inch'
	length_scale = globals()[units['length']]
	area_scale = globals()[units['length']+'2']

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

	# identify the labels
	labels = selected_bldgs.columns.values

	lon_label = get_label(['Longitude', 'longitude', 'lon', 'Lon'], labels, 'longitude')
	lat_label = get_label(['Latitude', 'latitude', 'lat', 'Lat'], labels, 'latitude')
	story_label = get_label(['Stories', 'stories'], labels, 'number of stories')
	year_label = get_label(['Year Built', 'yearbuilt', 'yearBuilt'], labels, 'year of construction')
	structure_label = get_label(['Structure Type', 'structure', 'structureType'], labels, 'structure type')
	area_label = get_label(['Area', 'areafootprint', 'areaFootprint'], labels, 'footprint area')
	occupancy_label = get_label(['Type ID', 'occupancy', 'occupancyType'], labels, 'occupancy type')
	cost_label = get_label(['Replacement Cost', 'replacementCost'], labels, 'replacement cost')

	for bldg_id, bldg in selected_bldgs.iterrows():

		if bldg[story_label] == 1:
			height = 4.66 # m
		elif bldg[structure_label][0] in ['C', 'P', 'R', 'U', 'M']:
			height = bldg[story_label] * 3.33 # m
		else:
			height = bldg[story_label] * 4.0 # m

		if occupancy_label == 'Type ID':
			occupancy = type_id_to_occupancy[bldg[occupancy_label]]
		else:
			occupancy = bldg[occupancy_label]

		population = bldg.get('population', 1.0)

		BIM_i = {
		    "RandomVariables": [],
		    "GI": dict(
		        BIM_id = str(bldg_id),
		        location = {
		            'latitude': bldg[lat_label],
		            'longitude': bldg[lon_label]
		        },
		        assetType       = 'Building',
		        yearBuilt       = bldg[year_label],
		        structType      = bldg[structure_label],
		        name            = bldg_id,
		        area            = float(bldg[area_label]) * area_scale,
		        numStory        = bldg[story_label],
		        occupancy       = occupancy,
		        height          = height, #the height is always in [m] already
		        replacementCost = bldg[cost_label],
		        replacementTime = 1.0,
		        population      = population,
		        units           = units
		    )
		}

		if 'soil_type' in bldg.keys():
			BIM_i["GI"].update({'soil_type': bldg['soil_type']})

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