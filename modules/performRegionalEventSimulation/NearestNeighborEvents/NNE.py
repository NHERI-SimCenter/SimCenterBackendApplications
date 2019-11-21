import argparse, posixpath, json
import numpy as np
import pandas as pd
from numpy.random import multinomial

from sklearn.neighbors import NearestNeighbors

def find_neighbors(building_file, event_metadata, samples, neighbors):
    
    meta_df = pd.read_csv(event_metadata, sep='\s+',header=0)
    
    lat_E = meta_df['lat']
    lon_E = meta_df['lon']
    X = np.array([[lo, la] for lo, la in zip(lon_E, lat_E)])

    nbrs = NearestNeighbors(n_neighbors = neighbors, algorithm='ball_tree').fit(X)

    with open(building_file, 'r') as f:
        bldg_dict = json.load(f)

    bim_df = pd.DataFrame(columns=['lat', 'lon', 'file'], index=np.arange(len(bldg_dict)))

    for i, bldg in enumerate(bldg_dict):
        with open(bldg['file'], 'r') as f:
            bldg_data = json.load(f)

        bldg_loc = bldg_data['GI']['location']
        bim_df.iloc[i]['lon'] = bldg_loc['longitude']
        bim_df.iloc[i]['lat'] = bldg_loc['latitude']
        bim_df.iloc[i]['file'] = bldg['file']

    Y = np.array([[lo, la] for lo, la in zip(bim_df['lon'], bim_df['lat'])])

    distances, indices = nbrs.kneighbors(Y)

    for i, (bim_id, dist_list, ind_list) in enumerate(zip(bim_df.index, 
                                                          distances, 
                                                          indices)):
        dist_list = 1./(dist_list**2.0)
        weights = np.array(dist_list)/np.sum(dist_list)
        evt_count = multinomial(samples, weights)

        event_list = []
        for e, i in zip(evt_count, ind_list):
            event_list += [meta_df.iloc[i]['sta'],]*e

        event_list_json = []
        for e_i, event in enumerate(event_list):
            event_list_json.append({
                "EventClassification": "Earthquake",
                "fileName": '{}x{}'.format(event,e_i),
                "type": "SW4"
                })

        bldg_file = bim_df.iloc[bim_id]['file']
        with open(bldg_file, 'r') as f:
            bldg_data = json.load(f)

        bldg_data['Events'] = {
            "EventClassification": "Earthquake",
            "Events": event_list_json,
            "type": "SW4_Events"
        }

        with open(bldg_file, 'w') as f:
            json.dump(bldg_data, f, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--buildingFile')
    parser.add_argument('--filenameEVENTmeta')
    parser.add_argument('--samples', type=int)
    parser.add_argument('--neighbors', type=int)
    args = parser.parse_args()

    find_neighbors(args.buildingFile, args.filenameEVENTmeta,
                   args.samples,args.neighbors)