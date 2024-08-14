# %%  # noqa: CPY001, D100, INP001
# required libraries numpy, geoandas,pandas,plotly
import json
import math

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon


def getStations(information, plot=False, show=False):  # noqa: FBT002, C901, N802, D103
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

    grid_type = information[
        'grid_type'
    ]  # grid type (options: A, B, C, D, E, Y, and Z, can be "all")

    # load the sites information
    df_allSites = pd.read_csv('M9_sites.csv', index_col=False)  # noqa: N806

    # create a geopandas dataframe
    gdf = gpd.GeoDataFrame(
        df_allSites,
        geometry=gpd.points_from_xy(df_allSites.Longitude, df_allSites.Latitude),
    )

    # delete the df_allSites to save memory
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

    if LocationFlag:
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
            gdf['Color'] = 'blue'
            for i in range(4):
                gdf.iloc[i, gdf.columns.get_loc('Color')] = 'red'

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

    if RegionFlag:
        gdf['Selected Site'] = [
            'Yes' if gdf['Color'][i] == 'red' else 'No' for i in range(len(gdf))
        ]
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

    if plot:
        import plotly.express as px  # noqa: PLC0415

        gdf['Color'] = gdf['Color'].replace(
            {'blue': 'All sites', 'red': 'Selected sites'}
        )
        fig = px.scatter_mapbox(
            gdf,
            lat='Latitude',
            lon='Longitude',
            color='Color',
            hover_name=gdf.index,
            hover_data={'Station Name': True, 'Selected Site': True},
            color_discrete_map={'All sites': '#1f77b4', 'Selected sites': '#ff7f0e'},
            # dont show the selected site in the legend
            center={'lat': centerlat, 'lon': centerlon},
            zoom=10,
            mapbox_style='open-street-map',
        )
        # fig.show()
        # save the html file
        # fig.write_html("M9_sites.html")
        # fig.write_image("M9_sites.png")
        # fig.
        if show:
            fig.show()

    if RegionFlag:
        gdf = gdf[gdf['Selected Site'] == 'Yes']

    if LocationFlag:
        gdf = gdf[gdf['Selected Site'] != 'No']
    gdf.drop(columns=['geometry', 'Color', 'Selected Site']).to_csv(
        'TapisFiles/selectedSites.csv', index=True
    )
    json.dump(information, open('TapisFiles/information.json', 'w'), indent=2)  # noqa: PLW1514, PTH123, SIM115
    # fig.show()


def haversine(lat1, lon1, lat2, lon2):
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

    return distance  # noqa: DOC201, RET504
