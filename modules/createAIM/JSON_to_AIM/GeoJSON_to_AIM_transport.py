#  # noqa: INP001, D100
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

import argparse
import importlib
import json
import os
import sys
import warnings

import geopandas as gpd
import momepy
import numpy as np
import pandas as pd
import shapely


# Break down long roads according to delta
def breakDownLongEdges(edges, delta, tolerance=10e-3):  # noqa: ANN001, ANN201, N802, D103
    dropedEdges = []  # noqa: N806
    newEdges = []  # noqa: N806
    crs = edges.crs
    edgesOrig = edges.copy()  # noqa: N806
    edgesOrig['IDbase'] = edgesOrig['ID'].apply(lambda x: x.split('_')[0])
    num_segExistingMap = edgesOrig.groupby('IDbase').count()['ID'].to_dict()  # noqa: N806
    edges_dict = edges.reset_index().to_crs('epsg:6500')
    edges_dict = edges_dict.to_dict(orient='records')
    for row_ind in range(len(edges_dict)):
        LS = edges_dict[row_ind]['geometry']  # noqa: N806
        num_seg = int(np.ceil(LS.length / delta))
        if num_seg == 1:
            continue
        distances = np.linspace(0, LS.length, num_seg + 1)
        points = shapely.MultiPoint(
            [LS.interpolate(distance) for distance in distances[:-1]]
            + [LS.boundary.geoms[1]]
        )
        LS = shapely.ops.snap(LS, points, tolerance)  # noqa: N806
        splittedLS = shapely.ops.split(LS, points).geoms  # noqa: N806
        currentEdge = edges_dict[row_ind].copy()  # noqa: N806
        num_segExisting = num_segExistingMap[currentEdge['ID'].split('_')[0]]  # noqa: N806
        for sLS_ind, sLS in enumerate(splittedLS):  # noqa: N806
            # create new edge
            if sLS_ind == 0:
                newID = currentEdge['ID']  # noqa: N806
            else:
                newID = (  # noqa: N806
                    currentEdge['ID'].split('_')[0] + '_' + str(num_segExisting + 1)
                )
                num_segExisting += 1  # noqa: N806
                num_segExistingMap[currentEdge['ID'].split('_')[0]] = (
                    num_segExistingMap[currentEdge['ID'].split('_')[0]] + 1
                )
            newGeom = sLS  # noqa: N806
            newEdge = currentEdge.copy()  # noqa: N806
            newEdge.update(
                {
                    'ID': newID,
                    'roadType': currentEdge['roadType'],
                    'geometry': newGeom,
                    'maxMPH': currentEdge['maxMPH'],
                    'lanes': currentEdge['lanes'],
                }
            )
            newEdges.append(newEdge)
        dropedEdges.append(row_ind)
    edges = edges.drop(dropedEdges)
    if len(newEdges) > 0:
        newEdges = gpd.GeoDataFrame(newEdges, crs='epsg:6500').to_crs(crs)  # noqa: N806
        edges = pd.concat([edges, newEdges], ignore_index=True)
    edges = edges.reset_index(drop=True)
    return edges  # noqa: RET504


def create_asset_files(  # noqa: ANN201, C901, D103, PLR0912, PLR0913, PLR0915
    output_file,  # noqa: ANN001
    asset_source_road,  # noqa: ANN001
    asset_source_bridge,  # noqa: ANN001
    asset_source_tunnel,  # noqa: ANN001
    bridge_filter,  # noqa: ANN001
    tunnel_filter,  # noqa: ANN001
    road_filter,  # noqa: ANN001
    doParallel,  # noqa: ANN001, N803
    roadSegLength,  # noqa: ANN001, N803
):
    # these imports are here to save time when the app is called without
    # the -getRV flag

    # check if running parallel
    numP = 1  # noqa: N806
    procID = 0  # noqa: N806
    runParallel = False  # noqa: N806

    if doParallel == 'True':
        mpi_spec = importlib.util.find_spec('mpi4py')
        found = mpi_spec is not None
        if found:
            from mpi4py import MPI

            runParallel = True  # noqa: N806
            comm = MPI.COMM_WORLD
            numP = comm.Get_size()  # noqa: N806
            procID = comm.Get_rank()  # noqa: N806
            if numP < 2:  # noqa: PLR2004
                doParallel = 'False'  # noqa: N806
                runParallel = False  # noqa: N806
                numP = 1  # noqa: N806
                procID = 0  # noqa: N806

    # Get the out dir, may not always be in the results folder if multiple assets are used
    outDir = os.path.dirname(output_file)  # noqa: PTH120, N806

    # check if a filter is provided for bridges
    if bridge_filter is not None:
        bridges_requested = []
        for assets in bridge_filter.split(','):
            if '-' in assets:
                asset_low, asset_high = assets.split('-')
                bridges_requested += list(range(int(asset_low), int(asset_high) + 1))
            else:
                bridges_requested.append(int(assets))
        bridges_requested = np.array(bridges_requested)
    # check if a filter is provided for tunnels
    if tunnel_filter is not None:
        tunnels_requested = []
        for assets in tunnel_filter.split(','):
            if '-' in assets:
                asset_low, asset_high = assets.split('-')
                tunnels_requested += list(range(int(asset_low), int(asset_high) + 1))
            else:
                tunnels_requested.append(int(assets))
        tunnels_requested = np.array(tunnels_requested)
    # check if a filter is provided for roads
    if road_filter is not None:
        roads_requested = []
        for assets in road_filter.split(','):
            if '-' in assets:
                asset_low, asset_high = assets.split('-')
                roads_requested += list(range(int(asset_low), int(asset_high) + 1))
            else:
                roads_requested.append(int(assets))
        roads_requested = np.array(roads_requested)

    # load the GeoJSON file with the asset information
    if asset_source_road is not None:
        roadsGDF = gpd.read_file(asset_source_road)  # noqa: N806
        datacrs = roadsGDF.crs
    else:
        roadsGDF = gpd.GeoDataFrame.from_dict({})  # noqa: N806
    if asset_source_bridge is not None:
        bridgesGDF = gpd.read_file(asset_source_bridge)  # noqa: N806
    else:
        bridgesGDF = gpd.GeoDataFrame.from_dict({})  # noqa: N806
    if asset_source_tunnel is not None:
        tunnelsGDF = gpd.read_file(asset_source_tunnel)  # noqa: N806
    else:
        tunnelsGDF = gpd.GeoDataFrame.from_dict({})  # noqa: N806

    # if there is a filter, then pull out only the required bridges
    if bridge_filter is not None:
        assets_available = bridgesGDF.index.values  # noqa: PD011
        bridges_to_run = bridges_requested[
            np.where(np.isin(bridges_requested, assets_available))[0]
        ]
        selected_bridges = bridgesGDF.loc[bridges_to_run]
    else:
        selected_bridges = bridgesGDF
        bridges_to_run = bridgesGDF.index.values  # noqa: PD011
    # if there is a filter, then pull out only the required tunnels
    if tunnel_filter is not None:
        assets_available = tunnelsGDF.index.values  # noqa: PD011
        tunnels_to_run = tunnels_requested[
            np.where(np.isin(tunnels_requested, assets_available))[0]
        ]
        selected_tunnels = tunnelsGDF.loc[tunnels_to_run]
    else:
        selected_tunnels = tunnelsGDF
        tunnels_to_run = tunnelsGDF.index.values  # noqa: PD011
    # if there is a filter, then pull out only the required roads
    if road_filter is not None:
        assets_available = roadsGDF.index.values  # noqa: PD011
        roads_to_run = roads_requested[
            np.where(np.isin(roads_requested, assets_available))[0]
        ]
        selected_roads = roadsGDF.loc[roads_to_run]
    else:
        selected_roads = roadsGDF
        roads_to_run = roadsGDF.index.values  # noqa: PD011

    if len(selected_roads) > 0:
        # Break down road network
        edges = breakDownLongEdges(selected_roads, roadSegLength)

        # Convert find connectivity and add start_node, end_node attributes
        graph = momepy.gdf_to_nx(edges.to_crs('epsg:6500'), approach='primal')
        with warnings.catch_warnings():  # Suppress the warning of disconnected components in the graph
            warnings.simplefilter('ignore')
            nodes, edges, sw = momepy.nx_to_gdf(
                graph, points=True, lines=True, spatial_weights=True
            )
        ### Some edges has start_node as the last point in the geometry and end_node as the first point, check and reorder
        for ind in edges.index:
            start = nodes.loc[edges.loc[ind, 'node_start'], 'geometry']
            end = nodes.loc[edges.loc[ind, 'node_end'], 'geometry']
            first = shapely.geometry.Point(edges.loc[ind, 'geometry'].coords[0])
            last = shapely.geometry.Point(edges.loc[ind, 'geometry'].coords[-1])
            # check if first and last are the same
            if start == first and end == last:
                continue
            elif start == last and end == first:  # noqa: RET507
                newStartID = edges.loc[ind, 'node_end']  # noqa: N806
                newEndID = edges.loc[ind, 'node_start']  # noqa: N806
                edges.loc[ind, 'node_start'] = newStartID
                edges.loc[ind, 'node_end'] = newEndID
            else:
                print(  # noqa: T201
                    ind,
                    'th row of edges has wrong start/first, end/last pairs, likely a bug of momepy.gdf_to_nx function',
                )
        locationGS = gpd.GeoSeries(  # noqa: N806
            edges['geometry'].apply(lambda x: x.centroid), crs=edges.crs
        ).to_crs(datacrs)
        edges = (
            edges.drop('mm_len', axis=1)
            .rename(columns={'node_start': 'start_node', 'node_end': 'end_node'})
            .to_crs(datacrs)
        )
        edges['location_lon'] = locationGS.apply(lambda x: x.x)
        edges['location_lat'] = locationGS.apply(lambda x: x.y)
        edges = edges.reset_index().rename(columns={'index': 'AIM_id'})
        edges['AIM_id'] = edges['AIM_id'].apply(lambda x: 'r' + str(x))
        edges.to_file(
            os.path.join(outDir, 'roadNetworkEdgesSelected.geojson'),  # noqa: PTH118
            driver='GeoJSON',
        )
        nodesNeeded = list(  # noqa: N806
            set(
                edges['start_node'].values.tolist()  # noqa: PD011
                + edges['end_node'].values.tolist()  # noqa: PD011
            )
        )
        nodes = nodes.loc[nodesNeeded, :]
        nodes = nodes.to_crs(datacrs)[['nodeID', 'geometry']]
        nodes.to_file(
            os.path.join(outDir, 'roadNetworkNodesSelected.geojson'),  # noqa: PTH118
            driver='GeoJSON',
        )
    else:
        edges = gpd.GeoDataFrame.from_dict({})

    count = 0
    ind = 0
    assets_array = []
    for ind, asset in selected_bridges.iterrows():
        asset_id = 'b' + str(bridges_to_run[ind])
        ind += 1  # noqa: PLW2901
        if runParallel == False or (count % numP) == procID:  # noqa: E712
            # initialize the AIM file
            # locationNodeID = str(asset["location"])
            AIM_i = {  # noqa: N806
                'RandomVariables': [],
                'GeneralInformation': dict(  # noqa: C408
                    AIM_id=asset_id,
                    location={
                        'latitude': asset['geometry'].centroid.coords[0][1],
                        'longitude': asset['geometry'].centroid.coords[0][0],
                    },
                ),
            }
            asset.pop('geometry')
            # save every label as-is
            AIM_i['GeneralInformation'].update(asset)
            # AIM_i["GeneralInformation"].update({"locationNode":locationNodeID})
            AIM_i['GeneralInformation'].update({'assetSubtype': 'hwyBridge'})
            AIM_file_name = f'{asset_id}-AIM.json'  # noqa: N806

            AIM_file_name = os.path.join(outDir, AIM_file_name)  # noqa: PTH118, N806

            with open(AIM_file_name, 'w', encoding='utf-8') as f:  # noqa: PTH123
                json.dump(AIM_i, f, indent=2)

            assets_array.append(dict(id=str(asset_id), file=AIM_file_name))  # noqa: C408

        count = count + 1

    ind = 0
    for ind, asset in selected_tunnels.iterrows():
        asset_id = 't' + str(tunnels_to_run[ind])
        ind += 1  # noqa: PLW2901
        if runParallel == False or (count % numP) == procID:  # noqa: E712
            # initialize the AIM file
            # locationNodeID = str(asset["location"])
            AIM_i = {  # noqa: N806
                'RandomVariables': [],
                'GeneralInformation': dict(  # noqa: C408
                    AIM_id=asset_id,
                    location={
                        'latitude': asset['geometry'].centroid.coords[0][1],
                        'longitude': asset['geometry'].centroid.coords[0][0],
                    },
                ),
            }
            asset.pop('geometry')
            # save every label as-is
            AIM_i['GeneralInformation'].update(asset)
            # AIM_i["GeneralInformation"].update({"locationNode":locationNodeID})
            AIM_i['GeneralInformation'].update({'assetSubtype': 'hwyTunnel'})
            AIM_file_name = f'{asset_id}-AIM.json'  # noqa: N806

            AIM_file_name = os.path.join(outDir, AIM_file_name)  # noqa: PTH118, N806

            with open(AIM_file_name, 'w', encoding='utf-8') as f:  # noqa: PTH123
                json.dump(AIM_i, f, indent=2)

            assets_array.append(dict(id=str(asset_id), file=AIM_file_name))  # noqa: C408

        count = count + 1

    ind = 0
    for row_ind in edges.index:
        asset_id = 'r' + str(row_ind)
        ind += 1
        if runParallel == False or (count % numP) == procID:  # noqa: E712
            # initialize the AIM file
            AIM_i = {  # noqa: N806
                'RandomVariables': [],
                'GeneralInformation': dict(  # noqa: C408
                    AIM_id=asset_id,
                    location={
                        'latitude': edges.loc[row_ind, 'location_lat'],
                        'longitude': edges.loc[row_ind, 'location_lon'],
                    },
                ),
            }
            AIM_i['GeneralInformation'].update(
                edges.loc[row_ind, :]
                .drop(['geometry', 'location_lat', 'location_lon'])
                .to_dict()
            )
            geom = {
                'type': 'LineString',
                'coordinates': [
                    [pt[0], pt[1]]
                    for pt in list(edges.loc[row_ind, 'geometry'].coords)
                ],
            }
            AIM_i['GeneralInformation'].update({'geometry': str(geom)})
            AIM_i['GeneralInformation'].update({'assetSubtype': 'roadway'})
            AIM_file_name = f'{asset_id}-AIM.json'  # noqa: N806

            AIM_file_name = os.path.join(outDir, AIM_file_name)  # noqa: PTH118, N806

            with open(AIM_file_name, 'w', encoding='utf-8') as f:  # noqa: PTH123
                json.dump(AIM_i, f, indent=2)

            assets_array.append(dict(id=str(asset_id), file=AIM_file_name))  # noqa: C408

        count = count + 1

    if procID != 0:
        # if not P0, write data to output file with procID in name and barrier

        output_file = os.path.join(outDir, f'tmp_{procID}.json')  # noqa: PTH118

        with open(output_file, 'w', encoding='utf-8') as f:  # noqa: PTH123
            json.dump(assets_array, f, indent=0)

        comm.Barrier()

    else:
        if runParallel == True:  # noqa: E712
            # if parallel & P0, barrier so that all files written above, then loop over other processor files: open, load data and append
            comm.Barrier()

            for i in range(1, numP):
                fileToAppend = os.path.join(outDir, f'tmp_{i}.json')  # noqa: PTH118, N806
                with open(fileToAppend, encoding='utf-8') as data_file:  # noqa: PTH123
                    json_data = data_file.read()
                assetsToAppend = json.loads(json_data)  # noqa: N806
                assets_array += assetsToAppend

        with open(output_file, 'w', encoding='utf-8') as f:  # noqa: PTH123
            json.dump(assets_array, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--assetFile',
        help='Path to the file that will contain a list of asset ids and '
        'corresponding AIM filenames',
    )
    parser.add_argument(
        '--assetSourceFileRoad', help='Path to the GIS file with the roads'
    )
    parser.add_argument(
        '--assetSourceFileBridge', help='Path to the JSON file with the bridges'
    )
    parser.add_argument(
        '--assetSourceFileTunnel', help='Path to the JSON file with the tunnels'
    )
    parser.add_argument(
        '--bridgesFilter',
        help='Filter applied to select a subset of bridges from the ' 'inventory',
        default=None,
    )
    parser.add_argument(
        '--tunnelsFilter',
        help='Filter applied to select a subset of assets from the ' 'inventory',
        default=None,
    )
    parser.add_argument(
        '--roadsFilter',
        help='Filter applied to select a subset of assets from the ' 'inventory',
        default=None,
    )
    parser.add_argument(
        '--roadSegLength',
        help='Maximum length (m) of road segments in the created AIM ' 'files',
        type=float,
        default=100,
    )
    parser.add_argument('--doParallel', default='False')
    parser.add_argument('-n', '--numP', default='8')
    parser.add_argument('-m', '--mpiExec', default='mpiexec')
    parser.add_argument(
        '--getRV',
        help='Identifies the preparational stage of the workflow. This app '
        'is only used in that stage, so it does not do anything if '
        'called without this flag.',
        default=False,
        nargs='?',
        const=True,
    )
    # parser.add_argument('--saveFullNetwork',
    #     help = "Save the full network into edges and nodes.",
    #     default=False,
    #     type=bool)

    args = parser.parse_args()

    if args.getRV:
        sys.exit(
            create_asset_files(
                args.assetFile,
                args.assetSourceFileRoad,
                args.assetSourceFileBridge,
                args.assetSourceFileTunnel,
                args.bridgesFilter,
                args.tunnelsFilter,
                args.roadsFilter,
                args.doParallel,
                args.roadSegLength,
            )
        )
    else:
        pass  # not used
