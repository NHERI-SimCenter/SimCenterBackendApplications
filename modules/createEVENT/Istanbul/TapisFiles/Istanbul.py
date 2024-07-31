# %%  # noqa: INP001, D100
import os

directory = './Events'
# check if the directory exists
if not os.path.exists(directory):  # noqa: PTH110
    os.makedirs(directory)  # noqa: PTH103
import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def Istanbul(information):  # noqa: ANN001, ANN201, N802, D103
    TopoFlag = information['TopoFlag']  # noqa: N806
    LocationFlag = information['LocationFlag']  # noqa: N806
    numSiteGM = information['number_of_realizations']  # noqa: N806

    randomFLag = True  # if True, the realizations are selected randomly, otherwise, the first numSiteGM sites are selected  # noqa: N806
    maxnumSiteGM = 57  # noqa: N806
    numSiteGM = min(numSiteGM, maxnumSiteGM)  # number of realizations  # noqa: N806

    directory = './Events'
    # check if the directory exists
    if not os.path.exists(directory):  # noqa: PTH110
        os.makedirs(directory)  # noqa: PTH103

    # changing realizations order
    indices = list(range(1, maxnumSiteGM + 1))
    if randomFLag:
        np.random.shuffle(indices)  # noqa: NPY002
    indices = indices[:numSiteGM]

    gdf = pd.read_csv('selectedSites.csv', index_col=0)

    if 'TopoFlag':
        # IstanbulDirectory = '/corral-repl/projects/NHERI/published/PRJ-3712/GM_data/GM_topo/'
        IstanbulDirectory = '/home/jovyan/work/projects/PRJ-3712/GM_data/GM_topo/'  # noqa: N806
    else:
        # IstanbulDirectory = '/corral-repl/projects/NHERI/published/PRJ-3712/GM_data/GM_flat/'
        IstanbulDirectory = '/home/jovyan/work/projects/PRJ-3712/GM_data/GM_flat/'  # noqa: N806

    # print number of cites
    print(f'Number of sites: {len(gdf)}')  # noqa: T201
    for realization in indices:
        # load the data frame from the hdf file
        if TopoFlag:
            df = pd.HDFStore(  # noqa: PD901
                f'{IstanbulDirectory}/Istanbul_sim{realization}.hdf5', 'r'
            )
        else:
            df = pd.HDFStore(  # noqa: PD901
                f'{IstanbulDirectory}/Istanbul_sim{realization}_flat.hdf5', 'r'
            )

        # return df
        for site in gdf.index:
            time = df['/Ax_data'][0]
            motiondict = {
                'Data': 'Time history generated using Istanbul simulations',
                'dT': time[1] - time[0],
                'name': f'site{site}_{realization}',
                'numSteps': len(time),
                'accel_x': df['Ax_data'][site + 1].tolist(),
                'accel_y': df['Ay_data'][site + 1].tolist(),
                'accel_z': df['Az_data'][site + 1].tolist(),
            }
            write_motion(site, './Events', realization, motiondict)
            gdf['filename'] = f'site_{site}_{realization}'
            if LocationFlag:
                break

    if LocationFlag:
        gdf = gdf.loc[[0]]
    # save the gdf to a csv file in the directory just "Station Name", "Latitude", "Longitude"
    gdf['Bedrock_Vs'] = 750
    gdf[['filename', 'Latitude', 'Longitude', 'Bedrock_Vs']].to_csv(
        f'{directory}/sites.csv', index=False
    )


def write_motion(site_name, directory, i, motiondict):  # noqa: ANN001, ANN201, D103
    filename = f'{directory}/site_{site_name}_{i}.json'
    with open(filename, 'w') as f:  # noqa: PTH123
        json.dump(motiondict, f, indent=2)


# get the location flag
with open('information.json') as file:  # noqa: PTH123
    information = json.load(file)
Istanbul(information)
