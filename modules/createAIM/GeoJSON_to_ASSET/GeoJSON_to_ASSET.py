import os, sys, argparse, posixpath, ntpath, json,importlib
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely, warnings, momepy, shutil

# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
class generalAIMGenerator:
    '''
    The generator of general AIM such as buildings, bridges, tunnels
    :param : The arg is used for ...
    :type arg: str
    :param `*args`: The variable arguments are used for ...
    :param `**kwargs`: The keyword arguments are used for ...
    :ivar arg: This is where we store arg
    :vartype arg: str
    '''
    def __init__(self, output_file):
        self.output_file = output_file
        self.gdf = None
        self.filter = None
    def load_asset_gdf(self, source_file):
        asset_gdf = gpd.read_file(source_file)
        self.gdf = asset_gdf
    def set_asset_gdf(self, asset_gdf):
        self.gdf = asset_gdf
    def selectAssets(self, filter):
        self.filter = filter
         # check if a filter is provided for bridges
        if self.filter is not None:
            asset_requested = []
            for assets in self.filter.split(','):
                if "-" in assets:
                    asset_low, asset_high = assets.split("-")
                    assets_requested += list(range(int(asset_low), int(asset_high)+1))
                else:
                    assets_requested.append(int(assets))
            assets_requested = np.array(assets_requested)
            assets_available = self.gdf.index.values
            assets_to_run = assets_requested[
                np.where(np.in1d(assets_requested, assets_available))[0]]
        else:
            assets_to_run = self.gdf.index.values
        self.gdf = self.gdf.loc[assets_to_run,:]
        return assets_to_run
    def createAIM(self, asset_idx, component_type = None):
        # initialize the AIM file
        # if component_type is not None:
        #     asset_id = component_type+"_"+str(asset_idx)
        # else:
        #     asset_id = str(asset_idx)
        asset_id = asset_idx
        asset = self.gdf.loc[asset_idx,:]
        AIM_i = {
            "RandomVariables": [],
            "GeneralInformation": dict(
                AIM_id = str(asset_id),
                location = {
                    'latitude': asset["geometry"].centroid.coords[0][1],
                    'longitude': asset["geometry"].centroid.coords[0][0]
                }
            )
        }
        # save every label as-is
        AIM_i["GeneralInformation"].update(asset)
        AIM_i["GeneralInformation"]['geometry'] = AIM_i["GeneralInformation"]['geometry'].wkt
        # if component_type is not None:
        #     AIM_i["GeneralInformation"].update({"assetSubtype":component_type})
        return AIM_i
    def dumpAIM(self, AIM_i):
        # assetSubtype = AIM_i['GeneralInformation'].get("assetSubtype", None)
        componentType = AIM_i['GeneralInformation'].get("type", None)
        outDir = os.path.dirname(self.output_file)
        if componentType:
            outDir = os.path.join(outDir, componentType)
        asset_id = AIM_i["GeneralInformation"]["AIM_id"]
        AIM_file_name = "{}-AIM.json".format(asset_id)
        AIM_file_name = os.path.join(outDir,AIM_file_name)
        with open(AIM_file_name, 'w', encoding="utf-8") as f:
            json.dump(AIM_i, f, indent=2, cls=NpEncoder)
        return AIM_file_name

class lineAIMGenerator(generalAIMGenerator):
    def breakDownLongLines(self, delta, tolerance = 10e-3):
        edges = self.gdf
        dropedEdges = []
        newEdges = []
        crs = edges.crs
        edgesOrig = edges.copy()
        # edgesOrig["IDbase"] = edgesOrig["OID"].apply(lambda x: x.split('_')[0])
        edgesOrig["IDbase"] = edgesOrig.index
        num_segExistingMap = edgesOrig.groupby("IDbase").count().iloc[:,0].to_dict()
        edges_dict = edges.reset_index().to_crs("epsg:6500")
        edges_dict = edges_dict.to_dict(orient='records')
        for row_ind in range(len(edges_dict)):
            LS = edges_dict[row_ind]["geometry"]
            num_seg = int(np.ceil(LS.length/delta))
            if num_seg == 1:
                continue
            distances = np.linspace(0, LS.length, num_seg+1)
            points = shapely.MultiPoint([LS.interpolate(distance) for distance in \
                                        distances[:-1]] + [LS.coords[-1]])
            LS = shapely.ops.snap(LS, points, tolerance)
            with warnings.catch_warnings(): #Suppress the warning of points not on 
                # LS. Shaply will first project the point to the line and then split
                warnings.simplefilter("ignore")
                splittedLS = shapely.ops.split(LS,points).geoms
            currentEdge = edges_dict[row_ind].copy()
            num_segExisting = num_segExistingMap[currentEdge["id"]]
            for sLS_ind, sLS in enumerate(splittedLS):
                # create new edge
                # if sLS_ind ==0:
                #     newID = currentEdge["id"]
                # else:
                #     newID = currentEdge["id"]+"_"+str(num_segExisting)
                #     num_segExisting +=1
                #     num_segExistingMap[currentEdge["id"]] += 1
                newID = currentEdge["id"]
                newGeom = sLS
                newEdge = currentEdge.copy()
                newEdge.update({"id":newID,"geometry":newGeom,\
                                "segID":sLS_ind})
                newEdges.append(newEdge)
            dropedEdges.append(edges_dict[row_ind]["id"])
        edges = edges.drop(dropedEdges)
        edges = edges.reset_index() # Convert "id" from index into a column
        if len(newEdges)>0:
            edges["segID"] = 0
            newEdges = gpd.GeoDataFrame(newEdges, crs="epsg:6500").to_crs(crs)
            edges = pd.concat([edges, newEdges], ignore_index=True)
            edges = edges.sort_values(['id','segID'])
            edges = edges.reset_index().rename(columns = {
                "id":"idBeforeSegment","index":"id"}).drop(columns = ["segID"])
        # self.gdf = edges.reset_index().rename(columns={"index":"AIM_id"})
        self.gdf = edges
        return
    def defineConnectivities(self, AIM_id_prefix = None, edges_file_name = None,\
                             nodes_file_name = None):
        # Convert find connectivity and add start_node, end_node attributes
        edges = self.gdf
        datacrs = edges.crs
        graph = momepy.gdf_to_nx(edges.to_crs("epsg:6500"), approach='primal')
        with warnings.catch_warnings(): #Suppress the warning of disconnected components in the graph
            warnings.simplefilter("ignore")
            nodes, edges, sw = momepy.nx_to_gdf(graph, points=True, lines=True,
                                                spatial_weights=True)
        # edges = edges.set_index('ind')
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
        # locationGS = gpd.GeoSeries(edges["geometry"].apply(lambda x: x.centroid),crs = edges.crs).to_crs(datacrs)
        edges = edges.drop("mm_len", axis = 1).rename(columns={"node_start":\
                        "StartNode", "node_end":"EndNode"}).to_crs(datacrs)
        # edges["location_lon"] = locationGS.apply(lambda x:x.x)
        # edges["location_lat"] = locationGS.apply(lambda x:x.y)
        edges = edges.rename(columns={"id":"AIM_id"})
        if AIM_id_prefix is not None:
            edges["AIM_id"] = edges["AIM_id"].apply(lambda x:AIM_id_prefix+"_"+str(x))
        outDir = os.path.dirname(self.output_file)
        if edges_file_name is not None:
            edges.to_file(os.path.join(outDir,f"{edges_file_name}.geojson"),\
                          driver = "GeoJSON")
        if nodes_file_name is not None:
            nodesNeeded = list(set(edges["StartNode"].values.tolist() +\
                                edges["EndNode"].values.tolist()))
            nodes = nodes.loc[nodesNeeded,:]
            nodes = nodes.to_crs(datacrs)[["nodeID","geometry"]]
            nodes.to_file(os.path.join(outDir,f"{nodes_file_name}.geojson"),\
                          driver = "GeoJSON")
        self.gdf = edges
        return

def split_and_select_components(input_config, asset_source_file):
    component_dict = dict()
    with open(asset_source_file, 'r', encoding="utf-8") as f:
        source_data = json.load(f)
    crs = source_data["crs"]
    featureList = source_data["features"]
    requested_dict = dict()
    for key, value in input_config.items():
        if isinstance(value, dict):
            filterString = value.get('filter', None)
            if filterString is None:
                continue
            assets_requested = []
            if filterString == '':
                assets_requested = np.array(assets_requested)
                requested_dict.update({key:assets_requested})
                component_dict.update({key:[]})
            else:
                for assets in filterString.split(','):
                    if "-" in assets:
                        asset_low, asset_high = assets.split("-")
                        assets_requested += list(range(int(asset_low), int(asset_high)+1))
                    else:
                        assets_requested.append(int(assets))
                assets_requested = np.array(assets_requested)
                requested_dict.update({key:assets_requested})
                component_dict.update({key:[]})
    for feat in featureList:
        component_type = feat["properties"].get("type", None)
        if (component_type in component_dict.keys()):
            feat_id = int(feat["id"])
            if requested_dict[component_type].size == 0:
                component_dict.pop(component_type)
                continue
            if (feat_id in requested_dict[component_type]):
                feat["properties"].update({"id":feat_id})
                component_dict[component_type].append(feat)
    for component in component_dict.keys():
        component_dict[component] = gpd.GeoDataFrame.from_features(\
            component_dict[component],crs=crs["properties"]["name"])\
                .set_index('id')
    return component_dict
def init_workdir(component_dict, outDir):
    os.chdir(outDir)
    for dir_or_file in os.listdir(outDir):
        if dir_or_file != 'log.txt':
            if os.path.isdir(dir_or_file):
                shutil.rmtree(dir_or_file)
            else:
                os.remove(dir_or_file)
    component_dir = dict()
    for comp in component_dict.keys():
        compDir = posixpath.join(outDir, comp)
        os.mkdir(compDir)
        component_dir.update({comp:compDir})
    return component_dir
    
def create_asset_files(output_file, asset_source_file,
    asset_type, input_file, doParallel): 
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
    outDir = os.path.dirname(output_file)
                
    with open(input_file, 'r', encoding="utf-8") as f:
        input_data = json.load(f)
    input_config = input_data["Applications"]["Assets"][asset_type]\
        ["ApplicationData"]
    # if input_config.get("Roadway", None):
    #     roadSegLength = float(input_config['Roadway'].get('maxRoadLength_m', "100000"))

    # assetSourceFile passed through command may be different from input_config when run on designsafe
    component_dict = split_and_select_components(input_config, asset_source_file)
    component_dir = init_workdir(component_dict, outDir)
    assets_array = []
    for component_type, component_data in component_dict.items():
        geom_type = type(component_data['geometry'].values[0])
        if geom_type in [shapely.Point, shapely.Polygon]:
        # if component_type in ["HwyBridge", "HwyTunnel"]:
            AIMgenerator = generalAIMGenerator(output_file)
            AIMgenerator.set_asset_gdf(component_data)
            selected_Asset_idxs = AIMgenerator.selectAssets(None)
        # elif component_type in ["Roadway"]:
        elif geom_type in [shapely.LineString]:
            AIMgenerator = lineAIMGenerator(output_file)
            AIMgenerator.set_asset_gdf(component_data)
            selected_Asset_idxs = AIMgenerator.selectAssets(None)
            # AIMgenerator.breakDownLongLines(roadSegLength)
            # # AIMgenerator.defineConnectivities(None, "hwy_edges",\
            # #                                   "hwy_nodes")
            # # Because the number of asset changes after break long lines.
            # # Run this to select all assets
            # selected_Asset_idxs = AIMgenerator.selectAssets(None)
        else:
            sys.exit((f"The geometry type {geom_type} defined for the") + \
                     (f"components {component_type} is not supported in ")+\
                     (f"the assets {asset_type}"))    
        # for each asset...
        count = 0
        for asset_idx in selected_Asset_idxs:
            if runParallel == False or (count % numP) == procID:
                # initialize the AIM file
                AIM_i = AIMgenerator.createAIM(asset_idx, component_type)
                AIM_file_name = AIMgenerator.dumpAIM(AIM_i)
                assets_array.append(dict(id=AIM_i['GeneralInformation']['AIM_id'], file=AIM_file_name))
            count = count + 1
    if procID != 0:
    # if not P0, write data to output file with procID in name and barrier
        output_file_p = os.path.join(outDir,f'tmp_{procID}.json')
        with open(output_file_p, 'w', encoding="utf-8") as f:
            json.dump(assets_array, f, indent=0)
        comm.Barrier() 
    else:
        if runParallel == True:
            # if parallel & P0, barrier so that all files written above, then loop over other processor files: open, load data and append
            comm.Barrier()        
            for i in range(1, numP):
                fileToAppend = os.path.join(outDir,f'tmp_{i}.json')
                with open(fileToAppend, 'r', encoding="utf-8") as data_file:
                    json_data = data_file.read()
                assetsToAppend = json.loads(json_data)
                assets_array += assetsToAppend
        with open(output_file, 'w', encoding="utf-8") as f:
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
    parser.add_argument('--doParallel', default="False")    
    parser.add_argument("-n", "--numP", default='8')
    parser.add_argument("-m", "--mpiExec", default='mpiexec')
    parser.add_argument('--getRV', nargs='?', const=True, default=False)
    args = parser.parse_args()

    if args.getRV:
        sys.exit(create_asset_files(args.assetFile, args.assetSourceFile,\
                                    args.assetType,\
                                    args.inputJsonFile, args.doParallel))
    else:
        pass # not used