# %%  # noqa: INP001, D100

import subprocess
import sys
from importlib import metadata as importlib_metadata

#
# need to make sure we have some python modules .. identify missing and install with python -m pip
#

modules_reqd = {'numpy', 'pandas', 'geopandas', 'shapely', 'requests', 'argparse'}
modules_installed = set()
for x in importlib_metadata.distributions():
    try:  # noqa: SIM105
        modules_installed.add(x.name)
    except:  # noqa: S110, PERF203, E722
        pass

# If installed packages could not be detected, use importlib_metadata backport:
if not modules_installed:
    import importlib_metadata

    for x in importlib_metadata.distributions():
        try:  # noqa: SIM105
            modules_installed.add(x.name)
        except:  # noqa: S110, PERF203, E722
            pass

missing = modules_reqd - modules_installed

if missing:
    python = sys.executable
    print('\nInstalling packages required for running this widget...')  # noqa: T201
    subprocess.check_call(  # noqa: S603
        [python, '-m', 'pip', 'install', '--user', *missing],
        stdout=subprocess.DEVNULL,
    )
    print('Successfully installed the required packages')  # noqa: T201

#
# now import our packages
#

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from shapely.geometry import Point, Polygon  # noqa: E402


# %%
def M9(information):  # noqa: ANN001, ANN201, C901, N802, PLR0912, PLR0915
    """The default is to select sites from all M9 sites, but
    grid type (options: A, B, C, D, E, Y, and Z, can be empty)
    (ref: https://sites.uw.edu/pnet/m9-simulations/about-m9-simulations/extent-of-model/)
    """  # noqa: D205, D400, D401, D415
    site_location = information['LocationFlag']

    if site_location:
        lat = information['lat']
        lon = information['lon']

    else:
        # its a regional location specified

        if information['RegionShape'] == 'Rectangle':
            # get the region of the desirable sites
            min_lat = information['min_lat']
            max_lat = information['max_lat']
            min_lon = information['min_lon']
            max_lon = information['max_lon']

        if information['RegionShape'] == 'Circle':
            # get the region of the desirable sites
            lat = information['lat']
            lon = information['lon']
            radius = information['radius']

    grid_type = information[
        'grid_type'
    ]  # grid type (options: A, B, C, D, E, Y, and Z, can be "all")

    randomFLag = True  # if True, the realizations are selected randomly, otherwise, the first numSiteGM sites are selected  # noqa: N806
    numSiteGM = information[  # noqa: N806
        'number_of_realizations'
    ]  # number of realizations
    maxnumSiteGM = 30  # noqa: N806
    numSiteGM = min(numSiteGM, maxnumSiteGM)  # number of realizations  # noqa: N806

    # changing realizations order
    indicies = list(range(maxnumSiteGM))
    if randomFLag:
        np.random.shuffle(indicies)  # noqa: NPY002
    indicies = indicies[:numSiteGM]

    directory = information['directory']  # directory to save the data
    # create the directory if it does not exist
    if not os.path.exists(directory):  # noqa: PTH110
        os.makedirs(directory)  # noqa: PTH103

    ## remove the files in the directory
    # os.system(f'rm -r {directory}/*')

    # load the sites information
    path_script = os.path.dirname(os.path.abspath(__file__))  # noqa: PTH100, PTH120
    path_site_file = path_script + '/M9_sites.csv'

    print(path_site_file)  # noqa: T201
    df_allSites = pd.read_csv(path_site_file, index_col=False)  # noqa: N806

    # create a geopandas dataframe
    gdf = gpd.GeoDataFrame(
        df_allSites,
        geometry=gpd.points_from_xy(df_allSites.Longitude, df_allSites.Latitude),
    )

    # deelte the df_allSites to save memory
    del df_allSites

    # limitation of each grid type (minx, miny, maxx, maxy)
    Gridboxes = {  # noqa: N806
        'A': (-123.2147269, 46.90566609, -121.1246222, 48.31489086),
        'B': (-128.4741831, 40.26059707, -121.0785236, 49.1785082),
        'C': (-123.2568915, 45.19862425, -122.2252305, 45.92126901),
        'D': (-123.3293999, 48.9970249, -122.3929914, 49.35841212),
        'E': (-123.8686827, 48.31165993, -123.1877513, 48.70158023),
        'Y': (-127.7497215, 40.41719958, -120.6351016, 50.13127206),
        'Z': (-127.7578767, 40.41524519, -121.2331997, 49.27983578),
        'All': (-128.4741831, 40.26059707, -121.0785236, 49.35841212),
    }

    # create a polygon for the  allowable region
    region = Polygon(
        [
            (Gridboxes[grid_type][0], Gridboxes[grid_type][1]),
            (Gridboxes[grid_type][0], Gridboxes[grid_type][3]),
            (Gridboxes[grid_type][2], Gridboxes[grid_type][3]),
            (Gridboxes[grid_type][2], Gridboxes[grid_type][1]),
        ]
    )

    if grid_type != 'All':
        # filter the site that the Station Name is start with the grid type
        gdf = gdf[gdf['Station Name'].str.startswith(grid_type)]
    else:
        gdf = gdf[gdf['Station Name'].str.startswith(('A', 'B', 'C', 'D', 'E'))]

    if site_location:
        # first check if the location is inner the regoin
        if not region.contains(Point(lon, lat)):
            print('The location is not in the selected grid region')  # noqa: T201
            print(  # noqa: T201
                'Please select a location in the region or change the grid type to "All"'
            )
            return
        else:  # noqa: RET505
            # find the nearest site to the location
            gdf['distance'] = gdf.distance(Point(lon, lat))
            gdf = gdf.sort_values('distance')
            gdf = gdf.iloc[0:4]

    else:
        # its regional

        if information['RegionShape'] == 'Rectangle':
            # Create a polygton using min_lat, max_lat, min_lon, max_lon
            RegionofInterset = Polygon(  # noqa: N806
                [
                    (min_lon, min_lat),
                    (min_lon, max_lat),
                    (max_lon, max_lat),
                    (max_lon, min_lat),
                ]
            )

            # Check that if the RegionofInterset and the region has intersection
            if not region.intersects(RegionofInterset):
                print('The selected region is not in the selected grid region')  # noqa: T201
                print(  # noqa: T201
                    'Please select a region in in the or change the grid type to "All"'
                )
                return
            else:  # noqa: RET505
                # Check if the RegionofInterset is in the region
                if not region.contains(RegionofInterset):
                    print(  # noqa: T201
                        'The selected region is not entirely in the selected grid region'
                    )
                    print(  # noqa: T201
                        'The selected region will be changed to the intersection of the selected region and the grid region'
                    )
                    RegionofInterset = region.intersection(RegionofInterset)  # noqa: N806
                else:
                    print(  # noqa: T201
                        'The selected region is entirely in the selected grid region'
                    )
                # now filter the sites that are in the regionofInterset
                gdf['Color'] = [
                    'red' if RegionofInterset.contains(gdf.geometry[i]) else 'blue'
                    for i in range(len(gdf))
                ]
                gdf = gdf[gdf.within(RegionofInterset)]

        if information['RegionShape'] == 'Circle':
            # chage the gdf to calculte the distance from the center of the circle in km
            gdf['distance'] = gdf.apply(
                lambda row: haversine(lat, lon, row['Latitude'], row['Longitude']),
                axis=1,
            )
            gdf['Color'] = [
                'red' if row['distance'] < radius else 'blue'
                for _, row in gdf.iterrows()
            ]
            gdf = gdf[gdf['distance'] < radius]

    APIFLAG = information[  # noqa: N806
        'APIFLAG'
    ]  # if the APIFLAG is True, we use M9 API to get the motion data

    if APIFLAG:
        # query flags
        ResponseSpectra = True  # noqa: N806

        # get the motion data from the API
        for _, site in gdf.iterrows():
            # get the motion data from the API
            site_name = site['Station Name']
            jobURL = f'https://m9-broadband-download-rwqks6gbba-uc.a.run.app/getMotionFromStationName?StationName={site_name}&ResponseSpectra={ResponseSpectra}'  # noqa: N806
            res_success = False
            iter_num = 0
            max_iter = 5
            print(f'Getting the motion data for {site_name}')  # noqa: T201

            while not (res_success) and (iter_num < max_iter):
                res = requests.get(jobURL)  # noqa: S113
                res_success = res.status_code == 200  # noqa: PLR2004
                iter_num = iter_num + 1

            if res_success:
                gmData = res.json()  # noqa: N806
                for i in indicies:
                    write_motion(site_name, directory, i, gmData[i], APIFLAG)
                    gdf['filename'] = f'{site_name}_{i}'

                if site_location:
                    break
            else:
                print(f'URL not replied for {site_name}, skipping for now')  # noqa: T201
                if site_location:
                    print('trying the next nearest site')  # noqa: T201

            if site_location and not (res_success):
                print('None of the nearest sites have motion data')  # noqa: T201
                print('Please check your internet connection or try again later')  # noqa: T201

    if not (APIFLAG):
        indicies = ['030']
        for i in indicies:
            for _, site in gdf.iterrows():
                # find the first Letter of the site name
                site_name = site['Station Name']
                lat = site['Latitude']
                lon = site['Longitude']
                firstLetter = site_name[0]  # noqa: N806
                filename = f'./csz{indicies[0]}/{firstLetter}/Xarray.nc'

                # reading the nc file
                data = xr.open_dataset(filename)  # noqa: F821
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

    # save the gdf to a csv file in the directory just "Station Name", "Latitude", "Longitude"
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
            'name': f'{site_name}_{i}',
            'dT': motiondict[dt],
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


def haversine(lat1, lon1, lat2, lon2):  # noqa: ANN001, ANN201
    """Calculate the great circle distance between two points
    on the earth specified in decimal degrees.
    """  # noqa: D205
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    r = 6371  # Radius of the Earth in kilometers
    distance = r * c

    return distance  # noqa: RET504
