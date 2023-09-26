# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 The Regents of the University of California
# Copyright (c) 2019 Leland Stanford Junior University
#
# This file is part of SimCenter Backend Applications.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# SimCenter Backend Applications. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Frank McKenna
# Adam Zsarnóczay
# Wael Elhaddad
# Stevan Gavrilovic
# Jinyan Zhao

import argparse, sys, os
import json
import numpy as np
import pandas as pd
import importlib
import shapely
import geopandas as gpd
import momepy
# Remove the nodes with 2 neibours 
# https://stackoverflow.com/questions/56380053/combine-edges-when-node-degree-is-n-in-networkx
# Needs parallel
def remove2neibourEdges(nodesID_to_remove, nodes_to_remove, edges, graph):
    # For each of those nodes
    removedID_list = [] # nodes with two neighbors. Removed from graph
    skippedID_list = [] # nodes involved in loops. Skipped removing.
    error_list = [] #nodes run into error. Left the node in the graph as is. 
    for i in range(len(nodesID_to_remove)):
        nodeid = nodesID_to_remove[i]
        node = nodes_to_remove[i]
        edge1 = edges[edges["node_end"] == nodeid]
        edge2 = edges[edges["node_start"] == nodeid]
        if (edge1.shape[0]==1 and edge2.shape[0]==1 and 
            edge1["node_start"].values[0]!= edge2["node_end"].values[0]):
            pass # Do things after continue
        elif(edge1.shape[0]==0 and edge2.shape[0]==2):
            ns = edges.loc[edge2.index[0],"node_start"]
            ne = edges.loc[edge2.index[0],"node_end"]
            edges.loc[edge2.index[0],"node_start"] = ne
            edges.loc[edge2.index[0],"node_end"] = ns
            # edges.loc[edge2.index[0],"geometry"] = shapely.LineString(list(edges.loc[edge2.index[0],"geometry"].coords)[::-1])
            edges.loc[edge2.index[0],"geometry"] = edges.loc[edge2.index[0],"geometry"].reverse()
            edge1 = edges[edges["node_end"] == nodeid]
            edge2 = edges[edges["node_start"] == nodeid]
        elif(edge1.shape[0]==2 and edge2.shape[0]==0):
            ns = edges.loc[edge1.index[1],"node_start"]
            ne = edges.loc[edge1.index[1],"node_end"]
            edges.loc[edge1.index[1],"node_start"] = ne
            edges.loc[edge1.index[1],"node_end"] = ns
            # edges.loc[edge1.index[1],"geometry"] = shapely.LineString(list(edges.loc[edge1.index[1],"geometry"].coords)[::-1])
            edges.loc[edge1.index[1],"geometry"] = edges.loc[edge1.index[1],"geometry"].reverse()
            edge1 = edges[edges["node_end"] == nodeid]
            edge2 = edges[edges["node_start"] == nodeid]
        else:
            skippedID_list.append(nodeid)
            continue
        try:
            removedID_list.append(nodeid)
            newLineCoords = list(edge1["geometry"].values[0].coords)+list(edge2["geometry"].values[0].coords[1:])
            # newLineCoords.append(edge2["geometry"].values[0].coords[1:])
            edges.loc[edge1.index,"geometry"] = shapely.LineString(newLineCoords)
            edges.loc[edge1.index, "node_end"] = edge2["node_end"].values[0]
            edges.drop(edge2.index, axis = 0, inplace=True)
            newEdge = list(graph.neighbors(node))
            graph.add_edge(newEdge[0], newEdge[1])
            # And delete the node
            graph.remove_node(node)
        except:
            error_list.append(nodeid)
    return edges
# Break down long roads according to delta
def breakDownLongEdges(edges, delta, roadDF, nodesDF, tolerance = 10e-3):
    dropedEdges = []
    newEdges = []
    for row_ind in edges.index:
        LS = edges.loc[row_ind,"geometry"]
        num_seg = int(np.ceil(LS.length/delta))
        if num_seg == 1:
            continue
        distances = np.linspace(0, LS.length, num_seg+1)
        points = shapely.MultiPoint([LS.interpolate(distance) for distance in distances[:-1]] + [LS.boundary.geoms[1]])
        LS = shapely.ops.snap(LS, points, tolerance)
        splittedLS = shapely.ops.split(LS,points).geoms
        currentEdge = edges.loc[row_ind,:].to_frame().T.copy()
        num_segExisting = roadDF["ID"].apply(lambda x: x.split('_')[0] == edges.loc[row_ind,"ID"].split('_')[0]).sum()
        newNodes = []
        currentNodesNum = nodesDF["nodeID"].max()
        for pt in points.geoms:
            newNode = gpd.GeoDataFrame({"nodeID":[currentNodesNum],"geometry":[pt]}, crs=edges.crs)
            currentNodesNum += 1
            newNodes.append(newNode)
        newNodes = newNodes[1:-1] #The first and last points already exists in the nodes DF. delete them
        nodesDF = pd.concat([nodesDF]+newNodes,ignore_index=True)
        for sLS_ind, sLS in enumerate(splittedLS):
            # create new edge
            if sLS_ind ==0:
                newID = currentEdge["ID"].values[0]
            else:
                newID = currentEdge["ID"].values[0].split("_")[0]+"_"+str(num_segExisting+1)
                num_segExisting +=1
            # newGeom = str({'type':'Feature','properties':{},
            #   'geometry':{'type':'LineString','coordinates':
            #               [[pt[0], pt[1]] for pt in sLS.coords]}})
            newGeom = sLS
            if sLS_ind ==0:
                newEdge_ns = currentEdge["node_start"].values[0]
                newEdge_ne = newNodes[sLS_ind]["nodeID"].values[0]
            elif sLS_ind < len(splittedLS)-1:
                newEdge_ns = newNodes[sLS_ind-1]["nodeID"].values[0]
                newEdge_ne = newNodes[sLS_ind]["nodeID"].values[0]
            else:
                newEdge_ns = newNodes[sLS_ind-1]["nodeID"].values[0]
                newEdge_ne = currentEdge["node_end"].values[0]
            newEdge = gpd.GeoDataFrame({"ID":[newID],
                                    "road_type":[currentEdge["road_type"].values[0]],
                                    "geometry":newGeom,
                                    "node_start":newEdge_ns,
                                    "node_end":newEdge_ne}, crs=edges.crs)
            newEdges.append(newEdge)
        dropedEdges.append(currentEdge.index[0])
    edges = edges.drop(dropedEdges)
    edges = pd.concat([edges]+newEdges, ignore_index=True)
    edges = edges.reset_index(drop=True)
    return edges, nodesDF

def create_asset_files(output_file, asset_source_file, bridge_filter,
                        tunnel_filter, road_filter, doParallel, roadSegLength):

    # these imports are here to save time when the app is called without
    # the -getRV flag
    

    # check if running parallel
    numP = 1
    procID = 0
    runParallel = False

    if doParallel == "True":
        mpi_spec = importlib.util.find_spec("mpi4py")
        found = mpi_spec is not None
        if found:
            import mpi4py
            from mpi4py import MPI
            runParallel = True
            comm = MPI.COMM_WORLD
            numP = comm.Get_size()
            procID = comm.Get_rank();
            if numP < 2:
                doParallel = "False"
                runParallel = False
                numP = 1
                procID = 0

    # Get the out dir, may not always be in the results folder if multiple assets are used
    outDir = os.path.dirname(output_file)
    
    # check if a filter is provided
    if bridge_filter is not None:
        bridges_requested = []
        for assets in bridge_filter.split(','):
            if "-" in assets:
                asset_low, asset_high = assets.split("-")
                bridges_requested += list(range(int(asset_low), int(asset_high)+1))
            else:
                bridges_requested.append(int(assets))
        bridges_requested = np.array(bridges_requested)
    # check if a filter is provided
    if tunnel_filter is not None:
        tunnels_requested = []
        for assets in tunnel_filter.split(','):
            if "-" in assets:
                asset_low, asset_high = assets.split("-")
                tunnels_requested += list(range(int(asset_low), int(asset_high)+1))
            else:
                tunnels_requested.append(int(assets))
        tunnels_requested = np.array(tunnels_requested)
    # Roads are filtered after a network is created
        
    # load the JSON file with the asset information
    with open(asset_source_file, "r") as sourceFile:
        assets_dict = json.load(sourceFile)
    
    bridges_array = assets_dict["hwy_bridges"]
    tunnels_array = assets_dict["hwy_tunnels"]
    roads_array = assets_dict["roadways"]
    nodes_dict= assets_dict["nodes"]
    assets_array = []

    # if there is a filter, then pull out only the required assets
    selected_bridges = []
    if bridge_filter is not None:
        assets_available = np.arange(len(bridges_array))
        bridges_to_run = bridges_requested[
            np.where(np.in1d(bridges_requested, assets_available))[0]]
        for i in bridges_to_run:
            selected_bridges.append(bridges_array[i])
    else:
        selected_bridges = bridges_array
        bridges_to_run = list(range(0, len(bridges_array)))
    # if there is a filter, then pull out only the required assets
    selected_tunnels = []
    if tunnel_filter is not None:
        assets_available = np.arange(len(tunnels_array))
        tunnels_to_run = tunnels_requested[
            np.where(np.in1d(tunnels_requested, assets_available))[0]]
        for i in tunnels_to_run:
            selected_tunnels.append(tunnels_array[i])
    else:
        selected_tunnels = tunnels_array
        tunnels_to_run = list(range(0, len(tunnels_array)))
    # if there is a filter, then pull out only the required assets
    # selected_roads = []
    # if road_filter is not None:
    #     assets_available = len(roads_array)
    #     roads_to_run = roads_requested[
    #         np.where(np.in1d(roads_requested, assets_available))[0]]
    #     for i in roads_to_run:
    #         selected_roads.append(roads_array[i])
    # else:
    #     selected_roads = roads_array
    #     roads_to_run = list(range(0, len(roads_array)))

    # Reconstruct road network
    datacrs = assets_dict["crs"]
    roadDF = pd.DataFrame.from_dict(roads_array)
    LineStringList = []
    for ind in roadDF.index:
        start_node = nodes_dict[str(roadDF.loc[ind, "start_node"])]
        end_node = nodes_dict[str(roadDF.loc[ind, "end_node"])]
        LineStringList.append(shapely.geometry.LineString([(start_node["lon"], start_node["lat"]), (end_node["lon"], end_node["lat"])]))
    roadDF["geometry"] = LineStringList
    roadDF = roadDF[["ID","road_type","lanes","capacity","geometry"]]
    roadGDF = gpd.GeoDataFrame(roadDF, geometry="geometry", crs=datacrs)
    graph = momepy.gdf_to_nx(roadGDF.to_crs("epsg:32610"), approach='primal')
    nodes, edges, sw = momepy.nx_to_gdf(graph, points=True, lines=True,
                                        spatial_weights=True)
    # Oneway or twoway is not considered in D&L, remove duplicated edges
    edges = edges[edges.duplicated(['node_start', 'node_end'], keep="first")==False]
    edges = edges.reset_index(drop=True).drop("mm_len",axis=1)
    ### Some edges has start_node as the last point in the geometry and end_node as the first point, check and reorder
    for ind in edges.index:
        start = nodes.loc[edges.loc[ind, "node_start"],"geometry"]
        end = nodes.loc[edges.loc[ind, "node_end"],"geometry"]
        first = shapely.geometry.Point(edges.loc[ind,"geometry"].coords[0])
        last = shapely.geometry.Point(edges.loc[ind,"geometry"].coords[-1])
        #check if first and last are the same
        if (start == first and end == last):
            continue
        elif (start == last and end == first):
            newStartID = edges.loc[ind, "node_end"]
            newEndID = edges.loc[ind, "node_start"]
            edges.loc[ind,"node_start"] = newStartID
            edges.loc[ind,"node_end"] = newEndID
        else:
            print(ind, "th row of edges has wrong start/first, end/last pairs, likely a bug of momepy.gdf_to_nx function")
    nodesID_to_remove = [i for i, n in enumerate(graph.nodes) if len(list(graph.neighbors(n))) == 2]
    nodes_to_remove = [n for i, n in enumerate(graph.nodes) if len(list(graph.neighbors(n))) == 2]
    
    edges = remove2neibourEdges(nodesID_to_remove, nodes_to_remove, edges, graph)
    remainingNodesOldID = list(set(edges["node_start"].values.tolist() + edges["node_end"].values.tolist()))
    nodes = nodes.loc[remainingNodesOldID,:].sort_index()
    nodes = nodes.reset_index(drop=True).reset_index().rename(columns={"index":"nodeID", "nodeID":"oldNodeID"})
    edges = edges.merge(nodes[["nodeID", "oldNodeID"]], left_on="node_start",
             right_on = "oldNodeID", how="left").drop(["node_start", "oldNodeID"], axis=1).rename(columns = {"nodeID":"node_start"})
    edges = edges.merge(nodes[["nodeID", "oldNodeID"]], left_on="node_end",
             right_on = "oldNodeID", how="left").drop(["node_end", "oldNodeID"], axis=1).rename(columns = {"nodeID":"node_end"})
    edges, nodes = breakDownLongEdges(edges, roadSegLength, roadDF, nodes)

    locationGS = gpd.GeoSeries(edges["geometry"].apply(lambda x: x.centroid),crs = edges.crs).to_crs(datacrs)
    edges = edges.to_crs(datacrs).rename(columns = {"node_start":"start_node","node_end":"end_node"})
    edges["location_lon"] = locationGS.apply(lambda x:x.x)
    edges["location_lat"] = locationGS.apply(lambda x:x.y)

    # check if a road filter is provided
    if road_filter is not None:
        roads_requested = []
        for assets in road_filter.split(','):
            if "-" in assets:
                asset_low, asset_high = assets.split("-")
                roads_requested += list(range(int(asset_low), int(asset_high)+1))
            else:
                roads_requested.append(int(assets))
        roads_requested = np.array(roads_requested)
    # if there is a filter, then pull out only the required assets
    if road_filter is not None:
        assets_available = edges.index.values
        roads_to_run = roads_requested[
            np.where(np.in1d(roads_requested, assets_available))[0]]
    else:
        roads_to_run = list(range(0, edges.shape[0]))
    edges = edges.reset_index().rename(columns={"index":"AIM_id"})
    edges["AIM_id"] = edges["AIM_id"].apply(lambda x:"r"+str(x))
    #Edges below are selected edges
    edges = edges.iloc[roads_to_run,:]
    edges.to_file(os.path.join(outDir,"roadNetworkEdges.geojson"), driver = "GeoJSON")

    nodesNeeded = list(set(edges["start_node"].values.tolist() + edges["end_node"].values.tolist()))
    nodes = nodes.loc[nodesNeeded,:]
    nodes = nodes.to_crs(datacrs)[["nodeID","geometry"]]
    nodes.to_file(os.path.join(outDir,"roadNetworkNodes.geojson"), driver = "GeoJSON")


    count = 0
    ind = 0
    for asset in selected_bridges:
        asset_id = "b" + str(bridges_to_run[ind])
        ind += 1
        if runParallel == False or (count % numP) == procID:

            # initialize the AIM file
            locationNodeID = str(asset["location"])
            AIM_i = {
                "RandomVariables": [],
                "GeneralInformation": dict(
                    AIM_id = asset_id,
                    location = {
                        'latitude': nodes_dict[locationNodeID]["lat"],
                        'longitude': nodes_dict[locationNodeID]["lon"]
                    }
                )
            }
            asset.pop("location")
            # save every label as-is
            AIM_i["GeneralInformation"].update(asset)
            AIM_i["GeneralInformation"].update({"locationNode":locationNodeID})
            AIM_i["GeneralInformation"].update({"assetSubtype":"hwy_bridge"})
            AIM_file_name = "{}-AIM.json".format(asset_id)
        
            AIM_file_name = os.path.join(outDir,AIM_file_name)
            
            with open(AIM_file_name, 'w') as f:
                json.dump(AIM_i, f, indent=2)

            assets_array.append(dict(id=str(asset_id), file=AIM_file_name))

        count = count + 1
    
    ind = 0
    for asset in selected_tunnels:
        asset_id = "t" + str(tunnels_to_run[ind])
        ind += 1
        if runParallel == False or (count % numP) == procID:

            # initialize the AIM file
            locationNodeID = str(asset["location"])
            AIM_i = {
                "RandomVariables": [],
                "GeneralInformation": dict(
                    AIM_id = asset_id,
                    location = {
                        'latitude': nodes_dict[locationNodeID]["lat"],
                        'longitude': nodes_dict[locationNodeID]["lon"]
                    }
                )
            }
            asset.pop("location")
            # save every label as-is
            AIM_i["GeneralInformation"].update(asset)
            AIM_i["GeneralInformation"].update({"locationNode":locationNodeID})
            AIM_i["GeneralInformation"].update({"assetSubtype":"hwy_tunnel"})
            AIM_file_name = "{}-AIM.json".format(asset_id)
        
            AIM_file_name = os.path.join(outDir,AIM_file_name)
            
            with open(AIM_file_name, 'w') as f:
                json.dump(AIM_i, f, indent=2)

            assets_array.append(dict(id=str(asset_id), file=AIM_file_name))

        count = count + 1
    
    ind = 0
    for row_ind in edges.index:
        asset_id = "r" + str(row_ind)
        ind += 1
        if runParallel == False or (count % numP) == procID:

            # initialize the AIM file
            AIM_i = {
                "RandomVariables": [],
                "GeneralInformation": dict(
                    AIM_id = asset_id,
                    location = {
                        'latitude': edges.loc[row_ind,"location_lat"],
                        'longitude': edges.loc[row_ind,"location_lon"]
                    }
                )
            }
            AIM_i["GeneralInformation"].update(edges.loc[row_ind,:].drop(["geometry","location_lat","location_lon"]).to_dict())
            geom = {"type":"LineString","coordinates":[[pt[0], pt[1]] for pt in list(edges.loc[0,"geometry"].coords)]}
            AIM_i["GeneralInformation"].update({"geometry":str(geom)})
            AIM_i["GeneralInformation"].update({"assetSubtype":"roadway"})
            AIM_file_name = "{}-AIM.json".format(asset_id)
        
            AIM_file_name = os.path.join(outDir,AIM_file_name)
            
            with open(AIM_file_name, 'w') as f:
                json.dump(AIM_i, f, indent=2)

            assets_array.append(dict(id=str(asset_id), file=AIM_file_name))

        count = count + 1

    if procID != 0:

        # if not P0, write data to output file with procID in name and barrier

        output_file = os.path.join(outDir,f'tmp_{procID}.json')

        with open(output_file, 'w') as f:
            json.dump(assets_array, f, indent=0)
    
        comm.Barrier()        

    else:

        if runParallel == True:

            # if parallel & P0, barrier so that all files written above, then loop over other processor files: open, load data and append
            comm.Barrier()        

            for i in range(1, numP):
                fileToAppend = os.path.join(outDir,f'tmp_{i}.json')
                with open(fileToAppend, 'r') as data_file:
                    json_data = data_file.read()
                assetsToAppend = json.loads(json_data)
                assets_array += assetsToAppend

        with open(output_file, 'w') as f:
            json.dump(assets_array, f, indent=2)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--assetFile',
        help = "Path to the file that will contain a list of asset ids and "
               "corresponding AIM filenames")
    parser.add_argument('--assetSourceFile',
        help = "Path to the JSON file with the transportation asset inventory")
    parser.add_argument('--bridgesFilter',
        help = "Filter applied to select a subset of bridges from the "
               "inventory",
        default=None)
    parser.add_argument('--tunnelsFilter',
        help = "Filter applied to select a subset of assets from the "
               "inventory",
        default=None)
    parser.add_argument('--roadsFilter',
        help = "Filter applied to select a subset of assets from the "
               "inventory",
        default=None)
    parser.add_argument('--roadSegLength',
        help = "Maximum length (m) of road segments in the created AIM "
               "files",
        type=float,
        default=100)
    parser.add_argument('--doParallel', default="False")    
    parser.add_argument("-n", "--numP", default='8')
    parser.add_argument("-m", "--mpiExec", default='mpiexec')
    parser.add_argument('--getRV',
        help = "Identifies the preparational stage of the workflow. This app "
               "is only used in that stage, so it does not do anything if "
               "called without this flag.",
        default=False,
        nargs='?', const=True)

    args = parser.parse_args()

    if args.getRV:
        sys.exit(create_asset_files(args.assetFile, args.assetSourceFile, args.bridgesFilter,
                                    args.tunnelsFilter, args.roadsFilter, args.doParallel, args.roadSegLength))
    else:
        pass # not used
