import argparse  # noqa: INP001, D100
import importlib
import json
import os
import posixpath
import shutil
import sys
import warnings

import geopandas as gpd
import momepy
import numpy as np
import pandas as pd
import shapely


# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
class NpEncoder(json.JSONEncoder):  # noqa: D101
    def default(self, obj):  # noqa: ANN001, ANN201, D102
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)  # noqa: UP008


class generalAIMGenerator:  # noqa: N801
    """The generator of general AIM such as buildings, bridges, tunnels
    :param : The arg is used for ...
    :type arg: str
    :param `*args`: The variable arguments are used for ...
    :param `**kwargs`: The keyword arguments are used for ...
    :ivar arg: This is where we store arg
    :vartype arg: str
    """  # noqa: D205, D400, D415

    def __init__(self, output_file):  # noqa: ANN001, ANN204, D107
        self.output_file = output_file
        self.gdf = None
        self.filter = None

    def load_asset_gdf(self, source_file):  # noqa: ANN001, ANN201, D102
        asset_gdf = gpd.read_file(source_file)
        self.gdf = asset_gdf

    def set_asset_gdf(self, asset_gdf):  # noqa: ANN001, ANN201, D102
        self.gdf = asset_gdf

    def selectAssets(self, filter):  # noqa: ANN001, ANN201, A002, N802, D102
        self.filter = filter
        # check if a filter is provided for bridges
        if self.filter is not None:
            asset_requested = []  # noqa: F841
            for assets in self.filter.split(','):
                if '-' in assets:
                    asset_low, asset_high = assets.split('-')
                    assets_requested += list(  # noqa: F821
                        range(int(asset_low), int(asset_high) + 1)
                    )
                else:
                    assets_requested.append(int(assets))
            assets_requested = np.array(assets_requested)
            assets_available = self.gdf.index.values  # noqa: PD011
            assets_to_run = assets_requested[
                np.where(np.isin(assets_requested, assets_available))[0]
            ]
        else:
            assets_to_run = self.gdf.index.values  # noqa: PD011
        self.gdf = self.gdf.loc[assets_to_run, :]
        return assets_to_run

    def createAIM(self, asset_idx, component_type=None):  # noqa: ANN001, ANN201, ARG002, N802, D102
        # initialize the AIM file
        # if component_type is not None:
        #     asset_id = component_type+"_"+str(asset_idx)
        # else:
        #     asset_id = str(asset_idx)
        asset_id = asset_idx
        asset = self.gdf.loc[asset_idx, :]
        AIM_i = {  # noqa: N806
            'RandomVariables': [],
            'GeneralInformation': dict(  # noqa: C408
                AIM_id=str(asset_id),
                location={
                    'latitude': asset['geometry'].centroid.coords[0][1],
                    'longitude': asset['geometry'].centroid.coords[0][0],
                },
            ),
        }
        # save every label as-is
        AIM_i['GeneralInformation'].update(asset)
        AIM_i['GeneralInformation']['geometry'] = AIM_i['GeneralInformation'][
            'geometry'
        ].wkt
        # if component_type is not None:
        #     AIM_i["GeneralInformation"].update({"assetSubtype":component_type})
        return AIM_i

    def dumpAIM(self, AIM_i):  # noqa: ANN001, ANN201, N802, N803, D102
        # assetSubtype = AIM_i['GeneralInformation'].get("assetSubtype", None)
        componentType = AIM_i['GeneralInformation'].get('type', None)  # noqa: N806
        outDir = os.path.dirname(self.output_file)  # noqa: PTH120, N806
        if componentType:
            outDir = os.path.join(outDir, componentType)  # noqa: PTH118, N806
        asset_id = AIM_i['GeneralInformation']['AIM_id']
        AIM_file_name = f'{asset_id}-AIM.json'  # noqa: N806
        AIM_file_name = os.path.join(outDir, AIM_file_name)  # noqa: PTH118, N806
        with open(AIM_file_name, 'w', encoding='utf-8') as f:  # noqa: PTH123
            json.dump(AIM_i, f, indent=2, cls=NpEncoder)
        return AIM_file_name


class lineAIMGenerator(generalAIMGenerator):  # noqa: N801, D101
    def breakDownLongLines(self, delta, tolerance=10e-3):  # noqa: ANN001, ANN201, N802, D102
        edges = self.gdf
        dropedEdges = []  # noqa: N806
        newEdges = []  # noqa: N806
        crs = edges.crs
        edgesOrig = edges.copy()  # noqa: N806
        # edgesOrig["IDbase"] = edgesOrig["OID"].apply(lambda x: x.split('_')[0])
        edgesOrig['IDbase'] = edgesOrig.index
        num_segExistingMap = edgesOrig.groupby('IDbase').count().iloc[:, 0].to_dict()  # noqa: N806
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
                + [LS.coords[-1]]
            )
            LS = shapely.ops.snap(LS, points, tolerance)  # noqa: N806
            with warnings.catch_warnings():  # Suppress the warning of points not on
                # LS. Shaply will first project the point to the line and then split
                warnings.simplefilter('ignore')
                splittedLS = shapely.ops.split(LS, points).geoms  # noqa: N806
            currentEdge = edges_dict[row_ind].copy()  # noqa: N806
            num_segExisting = num_segExistingMap[currentEdge['id']]  # noqa: N806, F841
            for sLS_ind, sLS in enumerate(splittedLS):  # noqa: N806
                # create new edge
                # if sLS_ind ==0:
                #     newID = currentEdge["id"]
                # else:
                #     newID = currentEdge["id"]+"_"+str(num_segExisting)
                #     num_segExisting +=1
                #     num_segExistingMap[currentEdge["id"]] += 1
                newID = currentEdge['id']  # noqa: N806
                newGeom = sLS  # noqa: N806
                newEdge = currentEdge.copy()  # noqa: N806
                newEdge.update({'id': newID, 'geometry': newGeom, 'segID': sLS_ind})
                newEdges.append(newEdge)
            dropedEdges.append(edges_dict[row_ind]['id'])
        edges = edges.drop(dropedEdges)
        edges = edges.reset_index()  # Convert "id" from index into a column
        if len(newEdges) > 0:
            edges['segID'] = 0
            newEdges = gpd.GeoDataFrame(newEdges, crs='epsg:6500').to_crs(crs)  # noqa: N806
            edges = pd.concat([edges, newEdges], ignore_index=True)
            edges = edges.sort_values(['id', 'segID'])
            edges = (
                edges.reset_index()
                .rename(columns={'id': 'idBeforeSegment', 'index': 'id'})
                .drop(columns=['segID'])
            )
        # self.gdf = edges.reset_index().rename(columns={"index":"AIM_id"})
        self.gdf = edges

    def defineConnectivities(  # noqa: ANN201, N802, D102
        self,
        AIM_id_prefix=None,  # noqa: ANN001, N803
        edges_file_name=None,  # noqa: ANN001
        nodes_file_name=None,  # noqa: ANN001
    ):
        # Convert find connectivity and add start_node, end_node attributes
        edges = self.gdf
        datacrs = edges.crs
        graph = momepy.gdf_to_nx(edges.to_crs('epsg:6500'), approach='primal')
        with warnings.catch_warnings():  # Suppress the warning of disconnected components in the graph
            warnings.simplefilter('ignore')
            nodes, edges, sw = momepy.nx_to_gdf(
                graph, points=True, lines=True, spatial_weights=True
            )
        # edges = edges.set_index('ind')
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
        # locationGS = gpd.GeoSeries(edges["geometry"].apply(lambda x: x.centroid),crs = edges.crs).to_crs(datacrs)
        edges = (
            edges.drop('mm_len', axis=1)
            .rename(columns={'node_start': 'StartNode', 'node_end': 'EndNode'})
            .to_crs(datacrs)
        )
        # edges["location_lon"] = locationGS.apply(lambda x:x.x)
        # edges["location_lat"] = locationGS.apply(lambda x:x.y)
        edges = edges.rename(columns={'id': 'AIM_id'})
        if AIM_id_prefix is not None:
            edges['AIM_id'] = edges['AIM_id'].apply(
                lambda x: AIM_id_prefix + '_' + str(x)
            )
        outDir = os.path.dirname(self.output_file)  # noqa: PTH120, N806
        if edges_file_name is not None:
            edges.to_file(
                os.path.join(outDir, f'{edges_file_name}.geojson'),  # noqa: PTH118
                driver='GeoJSON',
            )
        if nodes_file_name is not None:
            nodesNeeded = list(  # noqa: N806
                set(
                    edges['StartNode'].values.tolist()  # noqa: PD011
                    + edges['EndNode'].values.tolist()  # noqa: PD011
                )
            )
            nodes = nodes.loc[nodesNeeded, :]
            nodes = nodes.to_crs(datacrs)[['nodeID', 'geometry']]
            nodes.to_file(
                os.path.join(outDir, f'{nodes_file_name}.geojson'),  # noqa: PTH118
                driver='GeoJSON',
            )
        self.gdf = edges


def split_and_select_components(input_config, asset_source_file):  # noqa: ANN001, ANN201, C901, D103, PLR0912
    component_dict = dict()  # noqa: C408
    with open(asset_source_file, encoding='utf-8') as f:  # noqa: PTH123
        source_data = json.load(f)
    crs = source_data['crs']
    featureList = source_data['features']  # noqa: N806
    requested_dict = dict()  # noqa: C408
    for key, value in input_config.items():
        if isinstance(value, dict):
            filterString = value.get('filter', None)  # noqa: N806
            if filterString is None:
                continue
            assets_requested = []
            if filterString == '':
                assets_requested = np.array(assets_requested)
                requested_dict.update({key: assets_requested})
                component_dict.update({key: []})
            else:
                for assets in filterString.split(','):
                    if '-' in assets:
                        asset_low, asset_high = assets.split('-')
                        assets_requested += list(
                            range(int(asset_low), int(asset_high) + 1)
                        )
                    else:
                        assets_requested.append(int(assets))
                assets_requested = np.array(assets_requested)
                requested_dict.update({key: assets_requested})
                component_dict.update({key: []})
    for feat in featureList:
        component_type = feat['properties'].get('type', None)
        if component_type in component_dict:
            feat_id = int(feat['id'])
            if requested_dict[component_type].size == 0:
                component_dict.pop(component_type)
                continue
            if feat_id in requested_dict[component_type]:
                feat['properties'].update({'id': feat_id})
                component_dict[component_type].append(feat)
    for component in component_dict:
        component_dict[component] = gpd.GeoDataFrame.from_features(
            component_dict[component], crs=crs['properties']['name']
        ).set_index('id')
    return component_dict


def init_workdir(component_dict, outDir):  # noqa: ANN001, ANN201, N803, D103
    os.chdir(outDir)
    for dir_or_file in os.listdir(outDir):
        if dir_or_file != 'log.txt':
            if os.path.isdir(dir_or_file):  # noqa: PTH112
                shutil.rmtree(dir_or_file)
            else:
                os.remove(dir_or_file)  # noqa: PTH107
    component_dir = dict()  # noqa: C408
    for comp in component_dict.keys():  # noqa: SIM118
        compDir = posixpath.join(outDir, comp)  # noqa: N806
        os.mkdir(compDir)  # noqa: PTH102
        component_dir.update({comp: compDir})
    return component_dir


def create_asset_files(  # noqa: ANN201, C901, D103, PLR0912, PLR0915
    output_file,  # noqa: ANN001
    asset_source_file,  # noqa: ANN001
    asset_type,  # noqa: ANN001
    input_file,  # noqa: ANN001
    doParallel,  # noqa: ANN001, N803
):
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
    outDir = os.path.dirname(output_file)  # noqa: PTH120, N806

    with open(input_file, encoding='utf-8') as f:  # noqa: PTH123
        input_data = json.load(f)
    input_config = input_data['Applications']['Assets'][asset_type][
        'ApplicationData'
    ]
    # if input_config.get("Roadway", None):
    #     roadSegLength = float(input_config['Roadway'].get('maxRoadLength_m', "100000"))

    # assetSourceFile passed through command may be different from input_config when run on designsafe
    component_dict = split_and_select_components(input_config, asset_source_file)
    component_dir = init_workdir(component_dict, outDir)  # noqa: F841
    assets_array = []
    for component_type, component_data in component_dict.items():
        geom_type = type(component_data['geometry'].values[0])  # noqa: PD011
        if geom_type in [shapely.Point, shapely.Polygon]:
            # if component_type in ["HwyBridge", "HwyTunnel"]:
            AIMgenerator = generalAIMGenerator(output_file)  # noqa: N806
            AIMgenerator.set_asset_gdf(component_data)
            selected_Asset_idxs = AIMgenerator.selectAssets(None)  # noqa: N806
        # elif component_type in ["Roadway"]:
        elif geom_type in [shapely.LineString]:
            AIMgenerator = lineAIMGenerator(output_file)  # noqa: N806
            AIMgenerator.set_asset_gdf(component_data)
            selected_Asset_idxs = AIMgenerator.selectAssets(None)  # noqa: N806
            # AIMgenerator.breakDownLongLines(roadSegLength)
            # # AIMgenerator.defineConnectivities(None, "hwy_edges",\
            # #                                   "hwy_nodes")
            # # Because the number of asset changes after break long lines.
            # # Run this to select all assets
            # selected_Asset_idxs = AIMgenerator.selectAssets(None)
        else:
            sys.exit(
                (f'The geometry type {geom_type} defined for the')  # noqa: ISC003
                + (f'components {component_type} is not supported in ')
                + (f'the assets {asset_type}')
            )
        # for each asset...
        count = 0
        for asset_idx in selected_Asset_idxs:
            if runParallel == False or (count % numP) == procID:  # noqa: E712
                # initialize the AIM file
                AIM_i = AIMgenerator.createAIM(asset_idx, component_type)  # noqa: N806
                AIM_file_name = AIMgenerator.dumpAIM(AIM_i)  # noqa: N806
                assets_array.append(
                    dict(  # noqa: C408
                        id=AIM_i['GeneralInformation']['AIM_id'], file=AIM_file_name
                    )
                )
            count = count + 1
    if procID != 0:
        # if not P0, write data to output file with procID in name and barrier
        output_file_p = os.path.join(outDir, f'tmp_{procID}.json')  # noqa: PTH118
        with open(output_file_p, 'w', encoding='utf-8') as f:  # noqa: PTH123
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
            json.dump(assets_array, f, indent=2, cls=NpEncoder)
    # else:
    #     print(f"The asset_type {asset_type} is not one of Buildings, TransportationNetwork or WaterNetwork, and is currently not supported")
    #     sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assetFile')
    parser.add_argument('--assetSourceFile')
    parser.add_argument('--assetType')
    parser.add_argument('--inputJsonFile')
    parser.add_argument('--doParallel', default='False')
    parser.add_argument('-n', '--numP', default='8')
    parser.add_argument('-m', '--mpiExec', default='mpiexec')
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    if args.getRV:
        sys.exit(
            create_asset_files(
                args.assetFile,
                args.assetSourceFile,
                args.assetType,
                args.inputJsonFile,
                args.doParallel,
            )
        )
    else:
        pass  # not used
