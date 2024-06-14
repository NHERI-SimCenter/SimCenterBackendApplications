# %%
import os
directory = "./Events"
# check if the directory exists
if not os.path.exists(directory):
    os.makedirs(directory)
import numpy as np 
import pandas as pd
import json



def Istanbul(information):

    TopoFlag = information['TopoFlag']
    LocationFlag = information['LocationFlag']
    numSiteGM = information['number_of_realizations']


    randomFLag    = True ;# if True, the realizations are selected randomly, otherwise, the first numSiteGM sites are selected
    maxnumSiteGM  = 57;
    numSiteGM     = min(numSiteGM, maxnumSiteGM) ;# number of realizations

    directory = "./Events"
    # check if the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # changing realizations order
    indicies = list(range(1,maxnumSiteGM+1));
    if randomFLag:
        np.random.shuffle(indicies)
    indicies = indicies[:numSiteGM]
    

    gdf = pd.read_csv('selectedSites.csv', index_col=0)



    if "TopoFlag":
        # IstanbulDirectory = '/corral-repl/projects/NHERI/published/PRJ-3712/GM_data/GM_topo/'
        IstanbulDirectory = '/home/jovyan/work/projects/PRJ-3712/GM_data/GM_topo/'
    else :
        # IstanbulDirectory = '/corral-repl/projects/NHERI/published/PRJ-3712/GM_data/GM_flat/'
        IstanbulDirectory = '/home/jovyan/work/projects/PRJ-3712/GM_data/GM_flat/'



    # print number of cites 
    print(f'Number of sites: {len(gdf)}')
    for realization in indicies:
        # load the data frame from the hdf file
        if TopoFlag:
            df = pd.HDFStore(f'{IstanbulDirectory}/Istanbul_sim{realization}.hdf5', 'r')
        else :
            df = pd.HDFStore(f'{IstanbulDirectory}/Istanbul_sim{realization}_flat.hdf5', 'r')
            
        # return df
        for site in gdf.index:
            time = df["/Ax_data"][0]
            motiondict = {
                        "Data": "Time history generated using Istanbul simulations",
                        "dT"  : time[1] - time[0],
                        "name": f"site{site}_{realization}",
                        "numSteps" : len(time),
                        "accel_x" : df["Ax_data"][site+1].tolist(),
                        "accel_y" : df["Ay_data"][site+1].tolist(),
                        "accel_z" : df["Az_data"][site+1].tolist(),
                        }
            write_motion(site, "./Events", realization, motiondict)
            gdf['filename'] = f"site_{site}_{realization}"
            if LocationFlag:
                break;
            
    if LocationFlag:
        gdf = gdf.loc[[0]]
    # save the gdf to a csv file in the directory just "Station Name", "Latitude", "Longitude"
    gdf["Bedrock_Vs"] = 750
    gdf[['filename', 'Latitude', 'Longitude', 'Bedrock_Vs']].to_csv(f'{directory}/sites.csv', index=False)
        
        
def write_motion(site_name, directory, i, motiondict):
    filename = f"{directory}/site_{site_name}_{i}.json"
    with open(filename, 'w') as f:
        json.dump(motiondict, f, indent=2)
 



# get the location flag 
with open("information.json", "r") as file:
    information = json.load(file)
Istanbul(information)
