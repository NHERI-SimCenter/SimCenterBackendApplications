import json  # noqa: INP001, D100
import math
import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon


def getStations(information, plot=False, show=False):  # noqa: ANN001, ANN201, FBT002, C901, N802, PLR0912, PLR0915
    """This function is used to retrieve the information of the Istanbul physics-based simulations"""  # noqa: D400, D401, D404, D415
    RegionFlag = information['RegionFlag']  # noqa: N806
    LocationFlag = information['LocationFlag']  # noqa: N806

    if LocationFlag:
        # get the location of the site
        lat = information['lat']
        lon = information['lon']

    if RegionFlag:
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

    # Read the data from the csv file ignore indexing
    df_allSites = pd.read_csv(  # noqa: N806
        'All_Stations_Lat_Lon_Vs30_BedrockDepth.csv', index_col=False
    )
    df_allSites = df_allSites[['Longitude', 'Latitude', 'Depth (m)']]  # noqa: N806
    # add geometry using Lonnitude and Latitude
    gdf = gpd.GeoDataFrame(
        df_allSites,
        geometry=gpd.points_from_xy(df_allSites.Longitude, df_allSites.Latitude),
    )

    # filter all the sites on the surface
    if information['BedrockFlag']:
        gdf = gdf[gdf['Depth (m)'] < 0 + 1e-5]
    else:
        gdf = gdf[gdf['Depth (m)'] > 0 + 1e-5]

    # delete the df_allSites to save memory
    del df_allSites
    directory = information['directory']  # directory to save the data
    # create the directory if it does not exist
    if not os.path.exists(directory):  # noqa: PTH110
        os.makedirs(directory)  # noqa: PTH103
    # empty the directory
    files = os.listdir(directory)
    for file in files:
        os.remove(directory + '/' + file)  # noqa: PTH107

    if LocationFlag:
        # find the nearest site to the location
        gdf['distance'] = gdf.distance(Point(lon, lat))
        gdf = gdf.sort_values('distance')

        # create a column of the distance color and make the first 4 nearest sites red
        gdf['Color'] = 'blue'
        gdf.loc[gdf.index[:4], 'Color'] = 'red'

    if RegionFlag:
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

            # filter the sites that are within the polygon
            withinindicies = gdf.within(RegionofInterset)
            gdf['Color'] = 'blue'
            gdf.loc[withinindicies, 'Color'] = 'red'
            # gdf = gdf[gdf.within(RegionofInterset)]

            # check if the gdf is empty
            if withinindicies.sum() == 0:
                print(  # noqa: T201
                    'No sites are found in the selected region please change the region of interest'
                )
                return

        if information['RegionShape'] == 'Circle':
            # change the gdf to calculate the distance from the center of the circle in km
            gdf['distance'] = gdf.apply(
                lambda row: haversine(lat, lon, row['Latitude'], row['Longitude']),
                axis=1,
            )
            gdf['Color'] = [
                'red' if row['distance'] < radius else 'blue'
                for _, row in gdf.iterrows()
            ]
            gdf = gdf[gdf['distance'] < radius]

    if RegionFlag:
        gdf['Selected Site'] = gdf['Color'].apply(
            lambda x: 'Yes' if x == 'red' else 'No'
        )
    if LocationFlag:
        gdf['Selected Site'] = 'No'
        gdf.iloc[0, gdf.columns.get_loc('Selected Site')] = (
            'The closest site to the location'
        )
        gdf.iloc[1, gdf.columns.get_loc('Selected Site')] = (
            'The second closest site to the location'
        )
        gdf.iloc[2, gdf.columns.get_loc('Selected Site')] = (
            'The third closest site to the location'
        )
        gdf.iloc[3, gdf.columns.get_loc('Selected Site')] = (
            'The fourth closest site to the location'
        )

    if plot:
        import plotly.express as px

        # plot the sites
        if LocationFlag:
            centerlat = lat
            centerlon = lon
        if RegionFlag:
            if information['RegionShape'] == 'Circle':
                centerlat = lat
                centerlon = lon
            if information['RegionShape'] == 'Rectangle':
                centerlat = (min_lat + max_lat) / 2
                centerlon = (min_lon + max_lon) / 2

        gdf['Color'] = gdf['Color'].replace(
            {'blue': 'All sites', 'red': 'Selected sites'}
        )
        fig = px.scatter_mapbox(
            gdf,
            lat='Latitude',
            lon='Longitude',
            color='Color',
            hover_name=gdf.index,
            hover_data={'Selected Site': True},
            color_discrete_map={'All sites': '#1f77b4', 'Selected sites': '#ff7f0e'},
            center={'lat': centerlat, 'lon': centerlon},
            zoom=15,
            mapbox_style='open-street-map',
        )

        # save the html file
        # fig.write_html("Istanbul.html")
        if show:
            fig.show()

    if RegionFlag:
        gdf = gdf[gdf['Selected Site'] == 'Yes']

    if LocationFlag:
        gdf = gdf[gdf['Selected Site'] != 'No']

    gdf.drop(columns=['geometry', 'Color', 'Selected Site']).to_csv(
        'TapisFiles/selectedSites.csv', index=True
    )
    json.dump(information, open('TapisFiles/information.json', 'w'), indent=2)  # noqa: SIM115, PTH123


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


if __name__ == '__main__':
    information = {
        'RegionFlag': True,
        'LocationFlag': False,
        'RegionShape': 'Rectangle',
        'min_lat': 40.9938,
        'max_lat': 40.9945,
        'min_lon': 28.8978,
        'max_lon': 28.8995,
        'BedrockFlag': True,
        'directory': 'Events',
        'number_of_realizations': 1,
        'TopoFlag': True,
    }

    # change the directory to the file location
    os.chdir(os.path.dirname(os.path.realpath(__file__)))  # noqa: PTH120
    getStations(information, plot=False, show=False)
