# %%
import numpy as np
import matplotlib.pyplot as plt
import json 
import os
from scipy.integrate import trapezoid
import pyvista as pv
from geopy.distance import geodesic
import argparse
import importlib.util
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import plotly.express as px

def load_function_from_file(filepath, function_name):
    # Load the module from the given file path
    spec = importlib.util.spec_from_file_location("module.name", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Get the function from the module
    return getattr(module, function_name)


def calculate_distances_with_direction(lat1, lon1, lat2, lon2):
    # Original points
    point1 = (lat1, lon1)
    point2 = (lat2, lon2)
    
    # Calculate north-south distance (latitude changes, longitude constant)
    north_point = (lat2, lon1)
    north_south_distance = geodesic(point1, north_point).kilometers
    north_south_direction = "north" if lat2 > lat1 else "south"
    
    # Calculate east-west distance (longitude changes, latitude constant)
    west_point = (lat1, lon2)
    west_east_distance = geodesic(point1, west_point).kilometers
    west_east_direction = "east" if lon2 > lon1 else "west"
    
    # south is negative
    north_south_distance = -north_south_distance if north_south_direction == "south" else north_south_distance
    # west is negative
    west_east_distance = -west_east_distance if west_east_direction == "west" else west_east_distance

    return north_south_distance, west_east_distance


def PlotSources(info):

    tmpLocation = info["tmpLocation"]
    faultinfopath = tmpLocation + "/fault/faultInfo.json"
    faultinfo = json.load(open(faultinfopath,"r"))

    # // load the source time fuction in the faultinfo path
    sourcetimeinfopath = tmpLocation + "/fault/SourceTimeFunction.py"
    # // load the source time function
    source_time_function = load_function_from_file(sourcetimeinfopath, "source_time_function")


    numFaults = len(faultinfo["Faultfilenames"])
    faultFiles = faultinfo["Faultfilenames"]

    xfault = faultinfo["xmean"]
    yfault = faultinfo["ymean"]

    # filteringData = info["filteringData"]
    # dataType = info["dataType"]
    # filteringRange = info["filteringRange"]
    
    if numFaults < 1:
        print("No faults to plot")
        return
    
    if numFaults != len(faultFiles):
        print("Number of files does not match number of faults")
        return
    
    meshlist = []

    for i in range(numFaults):
        faultFile = tmpLocation + "/fault/" + faultFiles[i]
        sources = json.load(open(faultFile,"r"))
        x = np.zeros(len(sources))
        y = np.zeros(len(sources))
        z = np.zeros(len(sources))
        c1 = np.zeros(len(sources))
        c2 = np.zeros(len(sources))
        c3 = np.zeros(len(sources))
        c4 = np.zeros(len(sources))
        c5 = np.zeros(len(sources))
        for j,source in enumerate(sources):
            
            sourcetype = source["stf"]["type"]
            x[j] = source["x"] 
            y[j] = source["y"]
            z[j] = source["z"]
            c1[j] = source["strike"]
            c2[j] = source["dip"]
            c3[j] = source["rake"]
            c4[j] = source["t0"]
            parameters = source["stf"]["parameters"]
            c5[j] = source_time_function(*parameters, get_slip=True)




        mesh = pv.PolyData(np.c_[x,y,z])
        mesh["strike"] = c1
        mesh["dip"]    = c2
        mesh["rake"]   = c3
        mesh["t0"]     = c4
        mesh["slip"]   = c5
        # strikes_rad = np.radians(c1)
        # dips_rad = np.radians(c2)
        # rakes_rad = np.radians(c3)
        # # compute the vectors
        # s = np.array([-np.sin(strikes_rad), np.cos(strikes_rad), np.zeros_like(strikes_rad)])
        # d = np.array([
        #     np.cos(strikes_rad) * np.sin(dips_rad), 
        #     np.sin(strikes_rad) * np.sin(dips_rad), 
        #     -np.cos(dips_rad)
        # ])
        # # Compute the rake vectors
        # r = np.cos(rakes_rad) * s + np.sin(rakes_rad) * d
        # # Normalize the rake vectors
        # r_magnitudes = np.linalg.norm(r, axis=0)
        # r_normalized = r / r_magnitudes
        # mesh["rake_vector"] = r_normalized.T
    
        meshlist.append(mesh)


          


    Mesh = meshlist[0]
    for i in range(1,numFaults):
        Mesh = Mesh.merge(meshlist[i])
    
    xy = Mesh.points[:,:2]
    Mesh.points = Mesh.points- np.array([xfault,yfault,0])


    # finding hypocenter
    hypocenterid = np.argmin(c4)
    hypocenter = Mesh.points[hypocenterid].copy()
    # off screen rendering
    # remote_rendering
    pl = pv.Plotter(shape=(3,2),groups=[((0, np.s_[:]))])
    pl.subplot(0,0)
    # without scalars
    Mesh.set_active_scalars(None)
    pl.add_mesh(Mesh,color="blue")
    

    # plotting the hypocenter
    zhead = 0
    ztail = hypocenter[2]
    # create an arrow
    arrow = pv.Arrow(start=(hypocenter[0],hypocenter[1],ztail), direction=[0,0,1],scale = (zhead-ztail))
    pl.add_mesh(arrow, color="black",label="Hypocenter")
    # pl.add_mesh(pv.PolyData(hypocenter),color="black",point_size=20,render_points_as_spheres=True,label="Hypocenter")

    # pl.subplot(1)
    # # strike
    # pl.add_mesh(Mesh.copy(), scalars="strike", cmap="viridis",label=f"Fault {info['faultname']}")

    pl.subplot(1,0)
    # dip
    pl.add_mesh(Mesh.copy(), scalars="dip", cmap="viridis")

    pl.subplot(1,1)
    # rake
    pl.add_mesh(Mesh.copy(), scalars="rake", cmap="viridis")
    # Mesh2 = Mesh.copy()
    # arrows = Mesh2.glyph(orient="rake_vector",scale="slip")
    # arrows = Mesh2.glyph(orient="rake_vector",scale="slip")

    # pl.add_mesh(arrows, color="black")
    # computinf the vectors of the rake using slip and strike

    pl.subplot(2,0)
    # t0
    Mesh.set_active_scalars("t0")
    pl.add_mesh(Mesh.copy(), scalars="t0", cmap="viridis")

    pl.subplot(2,1)
    # slip
    Mesh.set_active_scalars("slip")
    pl.add_mesh(Mesh.copy(), scalars="slip", cmap="viridis")


    # get the bounds
    xmin, xmax, ymin, ymax, zmin, zmax  = Mesh.bounds



    if info["plottingStation"].lower() in ["yes","true"]:
        stationCoordinates = info["stationCoordinates"]
        coords = []
        xy2 = []
        for coord in stationCoordinates:
            north, east = calculate_distances_with_direction(faultinfo["latitude"],faultinfo["longitude"],coord[0],coord[1])
            coords.append([north,east,coord[2]])
            xy2.append([north + xfault ,east + yfault])

        
        stations = pv.PolyData(np.array(coords))

        xmin0, xmax0, ymin0, ymax0, zmin0, zmax0  = stations.bounds
        # update extreme values
        xmin = min(xmin,xmin0)
        xmax = max(xmax,xmax0)
        ymin = min(ymin,ymin0)
        ymax = max(ymax,ymax0)
        zmin = min(zmin,zmin0)
        zmax = max(zmax,zmax0)
  
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    pl.subplot(0,0)
    # if info["plottingStation"].lower() in ["yes","true"]:
        # pl.add_mesh(stations, color="red", point_size=10, render_points_as_spheres=True, label="Stations")
    
    
    if info["plotlayers"].lower() in ["yes","true"]:
        thickness = info["thickness"]
        # insert zero at the begining
        thickness.insert(0,0)
        # cummulative thickness
        cummulative = np.cumsum(thickness)
        if zmax < cummulative[-1]:
            zamx = cummulative[-1] + cummulative[-1]*0.8
        cummulative[-1] = zmax+ cummulative[-1]*0.8
        for j in range(len(thickness)-1):
            pl.add_mesh(pv.Cube(bounds=[xmin,xmax,ymin,ymax,cummulative[j],cummulative[j+1]]),color=colors[j],opacity=0.2,label=f"Layer {j+1}")
        


    
    if not os.path.exists(tmpLocation):
        os.makedirs(tmpLocation)
    pl.link_views()
    pl.show_grid(xtitle='X (North)',ytitle='Y (East)', ztitle='Z (Depth)')
    pl.add_axes()
    pl.camera.up = (0,0,-1)
    pl.camera.elevation = 60
    pl.camera.azimuth = -90
    pl.reset_camera()
    pl.camera.view_frustum(1.0)
    pl.add_legend()
    pl.export_html(f"{tmpLocation}/faults.html")

    transformer = Transformer.from_crs(faultinfo["epsg"], "epsg:4326")
    xy = xy *1000
    #  add (xfault,yfault) to the first row of xy
    xy = np.vstack((xy,[xfault,yfault]))
    x2,y2 = transformer.transform(xy[:,1], xy[:,0])
    geometry1 = [Point(xy) for xy in zip(x2,y2)]
    gdf1 = gpd.GeoDataFrame(geometry=geometry1)
    gdf1["type"] = "Fault"
    gdf1.loc[gdf1.index[-1], "type"] = "Fault Center"

    if info["plottingStation"].lower() in ["yes","true"]:
        xy2 = np.array(xy2) * 1000
        x2,y2 = transformer.transform(xy2[:,1], xy2[:,0])
        geometry2 = [Point(xy) for xy in zip(x2,y2)]
        gdf2 = gpd.GeoDataFrame(geometry=geometry2)
        gdf2["type"] = "Station"
        # make the first row of the type column of gdf2 to be "Fault Center"


        gdf = gpd.GeoDataFrame(pd.concat([gdf1,gdf2],ignore_index=True))
    # plot the data plotly
    else:
        gdf = gdf1
    
    # use different colors for fault and stations
    color_map = {
        "Fault": "blue",
        "Station": "red",
        "Fault Center": "green"
    }
    gdf['size'] = gdf['type'].apply(lambda x: 15 if x != 'Fault' else 7)
    # Plot with custom colors
    fig = px.scatter_mapbox(
        gdf, 
        lat=gdf.geometry.x, 
        lon=gdf.geometry.y, 
        color=gdf["type"], 
        size=gdf["size"],
        color_discrete_map=color_map  # Apply the custom color map
    )
    min_lat, max_lat = gdf.geometry.x.min(), gdf.geometry.x.max()
    min_lon, max_lon = gdf.geometry.y.min(), gdf.geometry.y.max()
    # Update layout and display map
    fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_zoom=10,  # Initial zoom level (adjustable)
    # mapbox_center={"lat": (min_lat + max_lat) / 2, "lon": (min_lon + max_lon) / 2},  # Center the map
    # mapbox_bounds={"west": min_lon, "east": max_lon, "south": min_lat, "north": max_lat}  # Set bounds
    )
    # export the as html
    fig.write_html(f"{tmpLocation}/faults_map.html")





if __name__ == "__main__":
    info = {
        "filteringData": "No",
        "filteringRange": [25, 45],
        "numsataions": 1,
        "stationCoordinates": [[-33.482426, -70.505214,0]],
        "plottingStation": "No",
        "plotlayers": "No",
        "thickness": [0.2,0.8,14.5,0],
        "tmpLocation": "tmp",
    }


    parser = argparse.ArgumentParser()
    parser.add_argument("--tmplocation", help="output location for DRM",required=True) 
    parser.add_argument("--thickness", help="Thickness of layers",required=False)
    parser.add_argument("--plotlayers", help="Plot layers",required=False)
    parser.add_argument("--plotstations", help="Plot stations",required=False)
    parser.add_argument("--numsataions", help="Number of stations",required=False)
    parser.add_argument("--stationcoordinates", help="Station coordinates",required=False)

    # print input arguments

        
    args = parser.parse_args()
    
    if args.tmplocation:
        info["tmpLocation"] = args.tmplocation
    
    if args.thickness:
        info["thickness"] = [float(i) for i in args.thickness.split(";")[:-1]]

    if args.plotlayers:
        info["plotlayers"] = args.plotlayers
        
    if args.plotstations:
        info["plottingStation"] = args.plotstations

    if args.numsataions:
        info["numstations"] = int(args.numsataions)

    if info["plottingStation"].lower() in ["yes","true"] and args.stationcoordinates:
        # delete the first three characters
        stringcoords = args.stationcoordinates[:]
        info["stationCoordinates"] = [[float(i) for i in j.split(",")] for j in stringcoords.split(";")[:-1]]

    
    if info["plottingStation"].lower() in ["yes","true"] and not args.stationcoordinates:
        print("Station coordinates are required")
        exit()
    
    if info["plottingStation"].lower() in ["yes","true"] and len(info["stationCoordinates"]) != info["numstations"]:
        print("Number of stations does not match number of coordinates")
        exit()
    
    if info["plotlayers"].lower() == "yes" and not args.thickness:
        print("Thickness of layers is required")
        exit()
        
    if info["plotlayers"].lower() == "yes" and len(info["thickness"]) < 1:
        print("At least one layer is required")
        exit()



    PlotSources(info)
    
  
            

    





