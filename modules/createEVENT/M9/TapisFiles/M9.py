# %%  # noqa: INP001, D100
import json
import os

import numpy as np
import pandas as pd
import xarray as xr


# 'netcdf4', 'h5netcdf', 'scipy'
# %%
def M9(information):  # noqa: ANN001, ANN201, N802
    """The default is to select sites from all M9 sites, but
    grid type (options: A, B, C, D, E, Y, and Z, can be empty)
    (ref: https://sites.uw.edu/pnet/m9-simulations/about-m9-simulations/extent-of-model/)
    """  # noqa: D205, D400, D401, D415
    LocationFlag = information['LocationFlag']  # noqa: N806
    numSiteGM = information['number_of_realizations']  # noqa: N806
    grid_type = information[  # noqa: F841
        'grid_type'
    ]  # grid type (options: A, B, C, D, E, Y, and Z, can be "all")

    randomFLag = True  # if True, the realizations are selected randomly, otherwise, the first numSiteGM sites are selected  # noqa: N806, E501
    maxnumSiteGM = 30  # noqa: N806
    numSiteGM = min(numSiteGM, maxnumSiteGM)  # number of realizations  # noqa: N806

    # changing realizations order
    # indicies = list(range(maxnumSiteGM));  # noqa: ERA001
    Realizations = [f'{i:03}' for i in range(1, 33)]  # noqa: N806
    indicies = np.arange(32)
    if randomFLag:
        np.random.shuffle(indicies)  # noqa: NPY002
    indicies = indicies[:numSiteGM]

    M9Path = '/home/jovyan/work/projects/PRJ-4603'  # noqa: N806

    directory = './Events'  # directory to save the data
    # create the directory if it does not exist
    if not os.path.exists(directory):  # noqa: PTH110
        os.makedirs(directory)  # noqa: PTH103

    gdf = pd.read_csv('selectedSites.csv', index_col=0)
    APIFLAG = information[  # noqa: N806
        'APIFLAG'
    ]  # if the APIFLAG is True, we use M9 API to get the motion data

    if not (APIFLAG):
        for i in indicies:
            for _, site in gdf.iterrows():
                # find the first Letter of the site name
                site_name = site['Station Name']
                lat = site['Latitude']
                lon = site['Longitude']
                firstLetter = site_name[0]  # noqa: N806
                filename = f'{M9Path}/csz{Realizations[i]}/{firstLetter}/Xarray.nc'

                # reading the nc file
                data = xr.open_dataset(filename)
                subset = data.sel(lat=lat, lon=lon, method='nearest')
                dt = data.coords['time'].values  # noqa: PD011
                dt = dt[1] - dt[0]
                sitedata = {
                    'dT': dt,
                    'accel_x': subset['acc_x'].values.tolist(),  # noqa: PD011
                    'accel_y': subset['acc_y'].values.tolist(),  # noqa: PD011
                    'accel_z': subset['acc_z'].values.tolist(),  # noqa: PD011
                }
                write_motion(site_name, directory, i, sitedata, APIFLAG)
                gdf['filename'] = f'{site_name}_{i}'
                if LocationFlag:
                    break

    if LocationFlag:
        gdf = gdf.loc[[0]]

    # save the gdf to a csv file in the directory just "Station Name", "Latitude", "Longitude"  # noqa: E501
    gdf[['filename', 'Latitude', 'Longitude']].to_csv(
        f'{directory}/sites.csv', index=False
    )


def write_motion(site_name, directory, i, motiondict, APIFLAG):  # noqa: ANN001, ANN201, N803, D103
    filename = f'{directory}/{site_name}_{i}.json'

    if APIFLAG:
        accel_x = 'AccelerationHistory-EW'
        accel_y = 'AccelerationHistory-NS'
        accel_z = 'AccelerationHistory-Vert'
        dt = 'TimeStep'
        datatowrite = {
            'Data': 'Time history generated using M9 simulations',
            'dT': motiondict[dt],
            'name': f'{site_name}_{i}',
            'numSteps': len(motiondict[accel_x]),
            'accel_x': motiondict[accel_x],
            'accel_y': motiondict[accel_y],
            'accel_z': motiondict[accel_z],
        }
    else:
        datatowrite = motiondict
        datatowrite['Data'] = 'Time history generated using M9 simulations'
        datatowrite['name'] = f'{site_name}_{i}'

    with open(filename, 'w') as f:  # noqa: PTH123
        json.dump(datatowrite, f, indent=2)


if __name__ == '__main__':
    # change the directory to the directory of the current file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # noqa: PTH100, PTH120

    with open('information.json') as file:  # noqa: PTH123
        information = json.load(file)
    M9(information)
