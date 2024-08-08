import numpy as np
import rasterio as rio
from scipy.interpolate import interp2d
import sys, warnings, shapely, pandas, os
from pyproj import Transformer
from pyproj import CRS
from enum import Enum
import geopandas as gpd
from scipy.spatial import ConvexHull

## Helper functions
def sampleRaster(raster_file_path, raster_crs, x, y, interp_scheme = 'nearest',\
                 dtype = None):
    """performs 2D interpolation at (x,y) pairs. Accepted interp_scheme = 'nearest', 'linear', 'cubic', and 'quintic'"""
    print(f"Sampling from the Raster File: {os.path.basename(raster_file_path)}...")
    invalid_value = np.nan
    xy_crs = CRS.from_user_input(4326)
    raster_crs = CRS.from_user_input(raster_crs)
    with rio.open(raster_file_path) as raster_file:
        try:
            raster_data = raster_file.read()
            if raster_data.shape[0] > 1:
                warnings.warn(f"More than one band in the file {raster_file_path}, the first band is used.")
        except:
            sys.exit(f"Can not read data from {raster_file_path}")
        if xy_crs != raster_crs:
            # make transformer for reprojection
            transformer_xy_to_data = Transformer.from_crs(xy_crs, raster_crs,\
                                                          always_xy=True)
            # reproject and store
            x_proj, y_proj = transformer_xy_to_data.transform(x, y)
            x = x_proj
            y = y_proj
        n_sample = len(x)
        if interp_scheme == 'nearest':
            sample = np.array([val[0] for val in raster_file.sample(list(zip(x,y)))])
        else:
            # create x and y ticks for grid
            x_tick = np.linspace(raster_file.bounds.left, \
                raster_file.bounds.right, raster_file.width,  endpoint=False)
            y_tick = np.linspace(raster_file.bounds.bottom,\
                raster_file.bounds.top, raster_file.height, endpoint=False)
            # create interp2d function
            interp_function = interp2d(
                x_tick, y_tick, np.flipud(raster_file.read(1)),
                kind=interp_scheme, fill_value=invalid_value)
            # get samples
            sample = np.transpose(
                [interp_function(x[i],y[i]) for i in range(n_sample)]
            )[0]
    # convert to target datatype
    if dtype is not None:
        sample = sample.astype(dtype)
    # clean up invalid values (returned as 1e38 by NumPy)
    sample[abs(sample)>1e10] = invalid_value
    return sample

## Helper functions
def sampleVector(vector_file_path, vector_crs, x, y, dtype = None):
    """performs spatial join of vector_file with xy'"""
    print(f"Sampling from the Vector File: {os.path.basename(vector_file_path)}...")
    invalid_value = np.nan
    xy_crs = CRS.from_user_input(4326)
    vector_gdf = gpd.read_file(vector_file_path)
    try:
        user_crs_input = CRS.from_user_input(vector_crs).to_epsg()
        if vector_gdf.crs.to_epsg() != user_crs_input:
            sys.exit(f"The CRS of vector file {vector_file_path} is {vector_gdf.crs}, and doesn't match the input CRS ({xy_crs}) defined for liquefaction triggering models")
    except:
        print("The input CRS ({xy_crs}) defined for liquefaction triggering models is invalid. The CRS of vector files are used")
    # if vector_gdf.crs != vector_crs:
    #     sys.exit(f"The CRS of vector file {vector_file_path} is {vector_gdf.crs}, and doesn't match the input CRS ({xy_crs}) defined for liquefaction triggering models")
    if xy_crs != vector_crs:
        # make transformer for reprojection
        transformer_xy_to_data = Transformer.from_crs(xy_crs, vector_crs,\
                                                      always_xy=True)
        # reproject and store
        x_proj, y_proj = transformer_xy_to_data.transform(x, y)
        x = x_proj
        y = y_proj
    # Create a convex hull containing all sites    
    sites = np.array([x, y]).transpose()
    try:
        hull = ConvexHull(sites)
        vertices = hull.vertices
        vertices = sites[np.append(vertices, vertices[0])]
        centroid = np.mean(vertices, axis=0)
        vertices = vertices + 0.05 * (vertices - centroid)
        RoI = shapely.geometry.Polygon(vertices)
    except:
        centroid = shapely.geometry.Point(np.mean(x), np.mean(y))
        points = [shapely.geometry.Point(x[i], y[i]) for i in range(len(x))]
        if len(points) == 1:
            distances = [0.1] # Degree
        else:
            distances = [point.distance(centroid) for point in points]
        max_distance = max(distances)*1.2
        angles = np.linspace(0, 2 * np.pi, 36)
        circle_points = [(centroid.x + max_distance * np.cos(angle), \
                        centroid.y + max_distance * np.sin(angle)) for angle in angles]
        RoI = shapely.geometry.Polygon(circle_points)
    data = dict()
    for col in vector_gdf.columns:
        data.update({col:[]})
    for row_index in vector_gdf.index:
        new_geom = RoI.intersection(vector_gdf.loc[row_index, 'geometry'])
        if new_geom.is_empty:
            continue
        columns = list(vector_gdf.columns)
        columns.remove('geometry')
        for col in columns:
            data[col].append(vector_gdf.loc[row_index, col])
        data['geometry'].append(new_geom)
    del vector_gdf
    gdf_roi = gpd.GeoDataFrame(data, geometry="geometry", crs=4326)
    geometry = [shapely.geometry.Point(lon, lat) for lon, lat in zip(x, y)]
    gdf_sites = gpd.GeoDataFrame(geometry=geometry, crs=4326).reset_index()
    merged = gpd.GeoDataFrame.sjoin(gdf_roi, gdf_sites, how = 'inner', predicate = 'contains')
    merged = merged.set_index('index_right').sort_index().drop(columns=['geometry'])
    gdf_sites = pandas.merge(gdf_sites, merged, on = 'index', how = 'left')
    gdf_sites.drop(columns=['geometry', 'index'], inplace=True)
    return gdf_sites

def find_additional_output_req(liq_info, current_step):
    additional_output_keys = []
    if current_step == 'Triggering':
        trigging_parameters = liq_info['Triggering']\
            ['Parameters'].keys()
        triger_dist_water = liq_info['Triggering']['Parameters'].get('DistWater', None)
        if triger_dist_water is None:
            return additional_output_keys
        lat_dist_water = liq_info['LateralSpreading']['Parameters'].get('DistWater', None)
        if 'LateralSpreading' in liq_info.keys():
            lat_dist_water = liq_info['LateralSpreading']['Parameters'].get('DistWater', None)
            if (liq_info['LateralSpreading']['Model'] == 'Hazus2020')\
                  and (lat_dist_water==triger_dist_water):
                additional_output_keys.append('dist_to_water')
    return additional_output_keys


class liq_susc_enum(Enum):
    very_high = 5
    high = 4
    moderate = 3
    low = 2
    very_low = 1
    none = 0

## Triggering:
class Liquefaction:
    def __init__(self) -> None:
        pass

# -----------------------------------------------------------
class ZhuEtal2017(Liquefaction):
    """
    A map-based procedure to quantify liquefaction at a given location using logistic models by Zhu et al. (2017). Two models are provided:

    1. For distance to coast < cutoff, **prob_liq** = f(**pgv**, **vs30**, **precip**, **dist_coast**, **dist_river**)
    2. For distance to coast >= cutoff, **prob_liq** = f(**pgv**, **vs30**, **precip**, **dist_coast**, **dist_river**, **gw_depth**)
    
    Parameters
    ----------
    From upstream PBEE:
    pgv: float, np.ndarray or list
        [cm/s] peak ground velocity
    mag: float, np.ndarray or list
        moment magnitude
    pga: float, np.ndarray or list
        [g] peak ground acceleration, only to check threshold where prob_liq(pga<0.1g)=0
    stations: list
        a list of dict containing the site infomation. Keys in the dict are 'ID',
        'lon', 'lat', 'vs30', 'z1pt0', 'z2pt5', 'vsInferred', 'rRup', 'rJB', 'rX'
        
    Geotechnical/geologic:
    vs30: float, np.ndarray or list
        [m/s] time-averaged shear wave velocity in the upper 30-meters
    precip: float, np.ndarray or list
        [mm] mean annual precipitation
    dist_coast: float, np.ndarray or list
        [km] distance to nearest coast
    dist_river: float, np.ndarray or list
        [km] distance to nearest river
    dist_water: float, np.ndarray or list
        [km] distance to nearest river, lake, or coast
    gw_depth: float, np.ndarray or list
        [m] groundwater table depth
        
    Fixed:
    # dist_water_cutoff: float, optional
    #     [km] distance to water cutoff for switching between global and coastal model, default = 20 km

    Returns
    -------
    prob_liq : float, np.ndarray
        probability for liquefaciton
    liq_susc_val : str, np.ndarray
        liquefaction susceptibility category value
    
    References
    ----------
    .. [1] Zhu, J., Baise, L.G., and Thompson, E.M., 2017, An Updated Geospatial Liquefaction Model for Global Application, Bulletin of the Seismological Society of America, vol. 107, no. 3, pp. 1365-1385.
    
    """
    def __init__(self, parameters, stations) -> None:
        self.stations = stations
        self.parameters = parameters
        self.dist_to_water = None #(km)
        self.dist_to_river = None #(km)
        self.dist_to_coast = None #(km)
        self.gw_depth = None #(m)
        self.precip = None # (mm)
        self.vs30 = None #(m/s)
        self.interpolate_spatial_parameters(parameters)

    def interpolate_spatial_parameters(self, parameters):
        # site coordinate in CRS 4326
        lat_station = [site['lat'] for site in self.stations]
        lon_station = [site['lon'] for site in self.stations]
        # dist_to_water 
        if parameters["DistWater"] == "Defined (\"distWater\") in Site File (.csv)":
            self.dist_to_water = np.array([site['distWater'] for site in self.stations])
        else:
            self.dist_to_water = sampleRaster(parameters["DistWater"], parameters["inputCRS"],\
                     lon_station, lat_station)
        # dist_to_river
        if parameters["DistRiver"] == "Defined (\"distRiver\") in Site File (.csv)":
            self.dist_to_river = np.array([site['distRiver'] for site in self.stations])
        else:
            self.dist_to_river = sampleRaster(parameters["DistRiver"], parameters["inputCRS"],\
                     lon_station, lat_station)
        # dist_to_coast
        if parameters["DistCoast"] == "Defined (\"distCoast\") in Site File (.csv)":
            self.dist_to_coast = np.array([site['distCoast'] for site in self.stations])
        else:
            self.dist_to_coast = sampleRaster(parameters["DistCoast"], parameters["inputCRS"],\
                     lon_station, lat_station)
        # gw_water
        if parameters["GwDepth"] == "Defined (\"gwDepth\") in Site File (.csv)":
            self.gw_depth = np.array([site['gwDepth'] for site in self.stations])
        else:
            self.gw_depth = sampleRaster(parameters["GwDepth"], parameters["inputCRS"],\
                     lon_station, lat_station)
        # precipitation 
        if parameters["Precipitation"] == "Defined (\"precipitation\") in Site File (.csv)":
            self.precip = np.array([site['precipitation'] for site in self.stations])
        else:
            self.precip = sampleRaster(parameters["Precipitation"], parameters["inputCRS"],\
                     lon_station, lat_station)
        self.vs30 = np.array([site['vs30'] for site in self.stations])
        print("Sampling finished")
    
    def run(self, ln_im_data, eq_data, im_list, output_keys, additional_output_keys):
        if ('PGA' in im_list) and ('PGV' in im_list):
            num_stations = len(self.stations)
            num_scenarios = len(eq_data)
            PGV_col_id = [i for i, x in enumerate(im_list) if x == 'PGV'][0]
            PGA_col_id = [i for i, x in enumerate(im_list) if x == 'PGA'][0]
            for scenario_id in range(num_scenarios):
                num_rlzs = ln_im_data[scenario_id].shape[2]
                im_data_scen = np.zeros([num_stations,\
                                    len(im_list)+len(output_keys), num_rlzs])
                im_data_scen[:,0:len(im_list),:] = ln_im_data[scenario_id]
                for rlz_id in range(num_rlzs):
                    pgv = np.exp(ln_im_data[scenario_id][:,PGV_col_id,rlz_id])
                    pga = np.exp(ln_im_data[scenario_id][:,PGA_col_id,rlz_id])
                    mag = float(eq_data[scenario_id][0])
                    model_output = self.model(pgv, pga, mag)
                    for i, key in enumerate(output_keys):
                        im_data_scen[:,len(im_list)+i,rlz_id] = model_output[key]
                ln_im_data[scenario_id] = im_data_scen
            im_list = im_list + output_keys
            additional_output = dict()
            for key in additional_output_keys:
                item = getattr(self, key, None)
                if item is None:
                    warnings.warn(f"Additional output {key} is not avaliable in the liquefaction trigging model 'ZhuEtal2017'.")
                else:
                    additional_output.update({key:item})
        else:
            sys.exit(f"At least one of 'PGA' and 'PGV' is missing in the selected intensity measures and the liquefaction trigging model 'ZhuEtal2017' can not be computed.")
            # print(f"At least one of 'PGA' and 'PGV' is missing in the selected intensity measures and the liquefaction trigging model 'ZhuEtal2017' can not be computed."\
            #       , file=sys.stderr)
            # sys.stderr.write("test")
            # sys.exit(-1)
        return ln_im_data, eq_data, im_list, additional_output
    
    def model(self, pgv, pga, mag):
        """Model"""
        # zero prob_liq
        zero_prob_liq = 1e-5 # decimal
        
        # distance cutoff for model
        model_transition = 20 # km

        # initialize arrays
        x_logistic = np.empty(pgv.shape)
        prob_liq = np.empty(pgv.shape)
        liq_susc_val = np.ones(pgv.shape)*-99
        liq_susc = np.empty(pgv.shape, dtype=int)
        
        # magnitude correction, from Baise & Rashidian (2020) and Allstadt et al. (2022)
        pgv_mag = pgv/(1+np.exp(-2*(mag-6)))
        pga_mag = pga/(10**2.24/mag**2.56)

        # find where dist_water <= cutoff for model of 20 km
        # coastal model
        ind_coastal = self.dist_to_water<=model_transition
        # global model
        # ind_global = list(set(list(range(pgv.shape[0]))).difference(set(ind_coastal)))
        ind_global = ~(self.dist_to_water<=model_transition)

        # set cap of precip to 1700 mm
        self.precip[self.precip>1700] = 1700

        # x = b0 + b1*var1 + ...
        # if len(ind_global) > 0:
        # liquefaction susceptbility value, disregard pgv term
        liq_susc_val[ind_global] = \
            8.801 + \
            -1.918   * np.log(self.vs30[ind_global]) + \
            5.408e-4 * self.precip[ind_global] + \
            -0.2054  * self.dist_to_water[ind_global] + \
            -0.0333  * self.gw_depth[ind_global]
        # liquefaction susceptbility value, disregard pgv term
        liq_susc_val[ind_coastal] = \
            12.435 + \
            -2.615   * np.log(self.vs30[ind_coastal]) + \
            5.556e-4 * self.precip[ind_coastal] + \
            -0.0287  * np.sqrt(self.dist_to_coast[ind_coastal]) + \
            0.0666   * self.dist_to_river[ind_coastal] + \
            -0.0369  * self.dist_to_river[ind_coastal]*np.sqrt(self.dist_to_coast[ind_coastal])
        # catch nan values
        liq_susc_val[np.isnan(liq_susc_val)] = -99.
        # x-term for logistic model = liq susc val + pgv term
        x_logistic[ind_global] = liq_susc_val[ind_global] + 0.334*np.log(pgv_mag[ind_global])
        # x-term for logistic model = liq susc val + pgv term
        x_logistic[ind_coastal] = liq_susc_val[ind_coastal] + 0.301*np.log(pgv_mag[ind_coastal])

        # probability of liquefaction
        prob_liq = 1/(1+np.exp(-x_logistic)) # decimal
        prob_liq = np.maximum(prob_liq,zero_prob_liq) # set prob to > "0" to avoid 0% in log

        # for pgv_mag < 3 cm/s, set prob to "0"
        prob_liq[pgv_mag<3] = zero_prob_liq
        # for pga_mag < 0.1 g, set prob to "0"
        prob_liq[pga_mag<0.1] = zero_prob_liq
        # for vs30 > 620 m/s, set prob to "0"
        prob_liq[self.vs30>620] = zero_prob_liq

        # calculate sigma_mu
        sigma_mu = (np.exp(0.25)-1) * prob_liq

        # determine liquefaction susceptibility category
        liq_susc[liq_susc_val>-1.15]  = liq_susc_enum['very_high'].value
        liq_susc[liq_susc_val<=-1.15] = liq_susc_enum['high'].value
        liq_susc[liq_susc_val<=-1.95] = liq_susc_enum['moderate'].value
        liq_susc[liq_susc_val<=-3.15] = liq_susc_enum['low'].value
        liq_susc[liq_susc_val<=-3.20] = liq_susc_enum['very_low'].value
        liq_susc[liq_susc_val<=-38.1] = liq_susc_enum['none'].value

        # liq_susc[prob_liq==zero_prob_liq] = 'none'
        
        return {"liq_prob":prob_liq, "liq_susc":liq_susc}
    
# -----------------------------------------------------------
class Hazus2020(Liquefaction):
    """
    Compute probability of liquefaction at a given location using a simplified method after Liao et al. (1988).
    Also called Youd and Perkins (1978) with Hazus (2020)
    
    Parameters
    ----------
    From upstream PBEE:
    pga: float, np.ndarray or list
        [g] peak ground acceleration
    mag: float, np.ndarray or list
        moment magnitude
        
    Geotechnical/geologic:
    gw_depth: float, np.ndarray or list
        [m] groundwater table depth
        
    Fixed:
    liq_susc: str, np.ndarray or list
        susceptibility category to liquefaction (none, very low, low, moderate, high, very high)

    Returns
    -------
    prob_liq : float, np.ndarray
        probability for liquefaciton
    
    References
    ----------
    .. [1] Federal Emergency Management Agency (FEMA), 2020, Hazus Earthquake Model - Technical Manual, Hazus 4.2 SP3, 436 pp. https://www.fema.gov/flood-maps/tools-resources/flood-map-products/hazus/user-technical-manuals.
    .. [2] Liao, S.S., Veneziano, D., and Whitman, R.V., 1988, Regression Models for Evaluating Liquefaction Probability, Journal of Geotechnical Engineering, vol. 114, no. 4, pp. 389-411.
    
    """
    def __init__(self, parameters, stations) -> None:
        self.stations = stations
        self.parameters = parameters
        self.gw_depth = None #(m)
        self.interpolate_spatial_parameters(parameters)
    
    def interpolate_spatial_parameters(self, parameters):
        # site coordinate in CRS 4326
        lat_station = [site['lat'] for site in self.stations]
        lon_station = [site['lon'] for site in self.stations]
        # gw_water
        if parameters["GwDepth"] == "Defined (\"gwDepth\") in Site File (.csv)":
            self.gw_depth = np.array([site['gwDepth'] for site in self.stations])
        else:
            self.gw_depth = sampleRaster(parameters["GwDepth"], parameters["inputCRS"],\
                     lon_station, lat_station)
        # liq_susc
        if parameters["LiqSusc"] == "Defined (\"liqSusc\") in Site File (.csv)":
            liq_susc_samples = pandas.DataFrame(np.array([site['liqSusc'] \
                for site in self.stations]), columns = ['liqSusc'])
            SusceptibilityKey = 'liqSusc'
        else:
            SusceptibilityFile = parameters["SusceptibilityFile"]
            liq_susc_samples = sampleVector(SusceptibilityFile,
                                   parameters["inputCRS"],\
                                    lon_station, lat_station)
            SusceptibilityKey = parameters["SusceptibilityKey"]
        self.liq_susc = []
        for susc in liq_susc_samples[SusceptibilityKey].unique():
            if not susc in list(liq_susc_enum.__members__.keys()):
                warnings.warn(f"Unkown susceptibility \"{susc}\" defined, and is treated as \"none\".")      
        for row_index in liq_susc_samples.index:
            if pandas.isna(liq_susc_samples.loc[row_index, SusceptibilityKey]):
                self.liq_susc.append(0)
            elif hasattr(liq_susc_enum, liq_susc_samples.loc[row_index, SusceptibilityKey]):
                self.liq_susc.append(liq_susc_enum[liq_susc_samples.loc[row_index, SusceptibilityKey]].value)
            else:
                self.liq_susc.append(0)
        self.liq_susc = np.array(self.liq_susc)
        # liq_susc = liq_susc_samples[parameters["SusceptibilityKey"]].fillna("NaN")
        # self.liq_susc = liq_susc.to_numpy()
        print("Sampling finished")


    def run(self, ln_im_data, eq_data, im_list, output_keys, additional_output_keys):
        if ('PGA' in im_list):
            num_stations = len(self.stations)
            num_scenarios = len(eq_data)
            PGA_col_id = [i for i, x in enumerate(im_list) if x == 'PGA'][0]
            for scenario_id in range(num_scenarios):
                num_rlzs = ln_im_data[scenario_id].shape[2]
                im_data_scen = np.zeros([num_stations,\
                                    len(im_list)+len(output_keys), num_rlzs])
                im_data_scen[:,0:len(im_list),:] = ln_im_data[scenario_id]
                for rlz_id in range(num_rlzs):
                    pga = np.exp(ln_im_data[scenario_id][:,PGA_col_id,rlz_id])
                    mag = float(eq_data[scenario_id][0])
                    model_output = self.model(pga, mag, self.gw_depth, self.liq_susc)
                    for i, key in enumerate(output_keys):
                        im_data_scen[:,len(im_list)+i,rlz_id] = model_output[key]
                ln_im_data[scenario_id] = im_data_scen
            im_list = im_list + output_keys
            additional_output = dict()
            for key in additional_output_keys:
                item = getattr(self, key, None)
                if item is None:
                    warnings.warn(f"Additional output {key} is not avaliable in the liquefaction trigging model 'Hazus2020'.")
                else:
                    additional_output.update({key:item})
        else:
            sys.exit(f"'PGA'is missing in the selected intensity measures and the liquefaction trigging model 'Hazus2020' can not be computed.")
        return ln_im_data, eq_data, im_list, additional_output
    @staticmethod
    # @njit
    def model(
        pga, mag, # upstream PBEE RV
        gw_depth, # geotechnical/geologic
        liq_susc, # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        # zero prob_liq
        zero_prob_liq = 1e-5 # decimal
        
        # initialize arrays
        prob_liq_pga = np.zeros(pga.shape)
        p_ml = np.zeros(pga.shape)
        
        # if gw_depth is nan
        gw_depth[np.isnan(gw_depth)] = 999        
        
        # correction factor for moment magnitudes other than M=7.5, eq. 4-21
        k_mag = 0.0027*mag**3 - 0.0267*mag**2 - 0.2055*mag + 2.9188
        # correction for groudnwater depths other than 5 feet, eq. 4-22
        k_gw_depth = 0.022 * gw_depth*3.28 + 0.93
        
        # get uncorrected p_liq given pga
        prob_liq_pga[liq_susc==liq_susc_enum['very_high'].value] = \
            np.maximum(np.minimum(9.09*pga[liq_susc==liq_susc_enum['very_high'].value]-0.82,1),0)
        prob_liq_pga[liq_susc==liq_susc_enum['high'].value] = \
            np.maximum(np.minimum(7.67*pga[liq_susc==liq_susc_enum['high'].value]-0.92,1),0)
        prob_liq_pga[liq_susc==liq_susc_enum['moderate'].value] = \
            np.maximum(np.minimum(6.67*pga[liq_susc==liq_susc_enum['moderate'].value]-1.00,1),0)
        prob_liq_pga[liq_susc==liq_susc_enum['low'].value] = \
            np.maximum(np.minimum(5.57*pga[liq_susc==liq_susc_enum['low'].value]-1.18,1),0)
        prob_liq_pga[liq_susc==liq_susc_enum['very_low'].value] = \
            np.maximum(np.minimum(4.16*pga[liq_susc==liq_susc_enum['very_low'].value]-1.08,1),0)
        prob_liq_pga[liq_susc==liq_susc_enum['none'].value] = 0

        # get portion of map unit susceptible to liquefaction
        p_ml[liq_susc==liq_susc_enum['very_high'].value] = 0.25
        p_ml[liq_susc==liq_susc_enum['high'].value] = 0.20
        p_ml[liq_susc==liq_susc_enum['moderate'].value] = 0.10
        p_ml[liq_susc==liq_susc_enum['low'].value] = 0.05
        p_ml[liq_susc==liq_susc_enum['very_low'].value] = 0.02
        p_ml[liq_susc==liq_susc_enum['none'].value] = 0.00
        
        # liquefaction likelihood, p_liq
        prob_liq = prob_liq_pga / k_mag / k_gw_depth * p_ml # eq. 4-20
        prob_liq = np.maximum(prob_liq,zero_prob_liq) # set prob to > "0" to avoid 0% in log

        # Zhu et al. (2017) boundary constraints
        # for pga_mag < 0.1 g, set prob to "0"
        # magnitude correction, from Baise & Rashidian (2020) and Allstadt et al. (2022)
        pga_mag = pga/(10**2.24/mag**2.56)
        prob_liq[pga_mag<0.1] = zero_prob_liq

        return {"liq_prob":prob_liq, "liq_susc":liq_susc}
        
# -----------------------------------------------------------
class Hazus2020_with_ZhuEtal2017(ZhuEtal2017):
    """
    Compute probability of liquefaction using Hazus (FEMA, 2020), with liq. susc. category from Zhu et al. (2017).
    
    Parameters
    ----------
    From upstream PBEE:
    pga: float, np.ndarray or list
        [g] peak ground acceleration
    mag: float, np.ndarray or list
        moment magnitude
        
    Geotechnical/geologic:
    vs30: float, np.ndarray or list
        [m/s] time-averaged shear wave velocity in the upper 30-meters
    precip: float, np.ndarray or list
        [mm] mean annual precipitation
    dist_coast: float, np.ndarray or list
        [km] distance to nearest coast
    dist_river: float, np.ndarray or list
        [km] distance to nearest river
    dist_water: float, np.ndarray or list
        [km] distance to nearest river, lake, or coast
    gw_depth: float, np.ndarray or list
        [m] groundwater table depth
        
    Fixed:
    # liq_susc: str, np.ndarray or list
    #     susceptibility category to liquefaction (none, very low, low, moderate, high, very high)

    Returns
    -------
    prob_liq : float, np.ndarray
        probability for liquefaciton
    
    References
    ----------
    .. [1] Federal Emergency Management Agency (FEMA), 2020, Hazus Earthquake Model - Technical Manual, Hazus 4.2 SP3, 436 pp. https://www.fema.gov/flood-maps/tools-resources/flood-map-products/hazus/user-technical-manuals.
    .. [2] Liao, S.S., Veneziano, D., and Whitman, R.V., 1988, Regression Models for Evaluating Liquefaction Probability, Journal of Geotechnical Engineering, vol. 114, no. 4, pp. 389-411.
    .. [3] Zhu, J., Baise, L.G., and Thompson, E.M., 2017, An Updated Geospatial Liquefaction Model for Global Application, Bulletin of the Seismological Society of America, vol. 107, no. 3, pp. 1365-1385.
    
    """
    def model(self, pgv, pga, mag):
        """Model"""
        # zero prob_liq
        zero_prob_liq = 1e-5 # decimal
        
        # distance cutoff for model
        model_transition = 20 # km

        # initialize arrays
        prob_liq = np.empty(pgv.shape)
        liq_susc_val = np.ones(pgv.shape)*-99
        liq_susc = np.empty(pgv.shape, dtype=int)

        # find where dist_water <= cutoff for model of 20 km
        # coastal model
        ind_coastal = self.dist_to_water<=model_transition
        # global model
        # ind_global = list(set(list(range(pgv.shape[0]))).difference(set(ind_coastal)))
        ind_global = ~(self.dist_to_water<=model_transition)

        # set cap of precip to 1700 mm
        self.precip[self.precip>1700] = 1700

        # x = b0 + b1*var1 + ...
        # if len(ind_global) > 0:
        # liquefaction susceptbility value, disregard pgv term
        liq_susc_val[ind_global] = \
            8.801 + \
            -1.918   * np.log(self.vs30[ind_global]) + \
            5.408e-4 * self.precip[ind_global] + \
            -0.2054  * self.dist_to_water[ind_global] + \
            -0.0333  * self.gw_depth[ind_global]
        # liquefaction susceptbility value, disregard pgv term
        liq_susc_val[ind_coastal] = \
            12.435 + \
            -2.615   * np.log(self.vs30[ind_coastal]) + \
            5.556e-4 * self.precip[ind_coastal] + \
            -0.0287  * np.sqrt(self.dist_to_coast[ind_coastal]) + \
            0.0666   * self.dist_to_river[ind_coastal] + \
            -0.0369  * self.dist_to_river[ind_coastal]*np.sqrt(self.dist_to_coast[ind_coastal])
        # catch nan values
        liq_susc_val[np.isnan(liq_susc_val)] = -99.

        # determine liquefaction susceptibility category
        liq_susc[liq_susc_val>-1.15]  = liq_susc_enum['very_high'].value
        liq_susc[liq_susc_val<=-1.15] = liq_susc_enum['high'].value
        liq_susc[liq_susc_val<=-1.95] = liq_susc_enum['moderate'].value
        liq_susc[liq_susc_val<=-3.15] = liq_susc_enum['low'].value
        liq_susc[liq_susc_val<=-3.20] = liq_susc_enum['very_low'].value
        liq_susc[liq_susc_val<=-38.1] = liq_susc_enum['none'].value
        # Below are HAZUS
        # magnitude correction, from Baise & Rashidian (2020) and Allstadt et al. (2022)
        pga_mag = pga/(10**2.24/mag**2.56)
        # initialize arrays
        prob_liq_pga = np.zeros(pga.shape)
        p_ml = np.zeros(pga.shape)
        # correction factor for moment magnitudes other than M=7.5, eq. 4-21
        k_mag = 0.0027*mag**3 - 0.0267*mag**2 - 0.2055*mag + 2.9188
        # correction for groudnwater depths other than 5 feet, eq. 4-22
        k_gw_depth = 0.022 * self.gw_depth*3.28 + 0.93
        # get uncorrected p_liq given pga
        prob_liq_pga[liq_susc==liq_susc_enum['very_high'].value] = \
            np.maximum(np.minimum(9.09*pga[liq_susc==liq_susc_enum['very_high'].value]-0.82,1),0)
        prob_liq_pga[liq_susc==liq_susc_enum['high'].value] = \
            np.maximum(np.minimum(7.67*pga[liq_susc==liq_susc_enum['high'].value]-0.92,1),0)
        prob_liq_pga[liq_susc==liq_susc_enum['moderate'].value] = \
            np.maximum(np.minimum(6.67*pga[liq_susc==liq_susc_enum['moderate'].value]-1.00,1),0)
        prob_liq_pga[liq_susc==liq_susc_enum['low'].value] = \
            np.maximum(np.minimum(5.57*pga[liq_susc==liq_susc_enum['low'].value]-1.18,1),0)
        prob_liq_pga[liq_susc==liq_susc_enum['very_low'].value] = \
            np.maximum(np.minimum(4.16*pga[liq_susc==liq_susc_enum['very_low'].value]-1.08,1),0)
        prob_liq_pga[liq_susc==liq_susc_enum['none'].value] = 0

        # get portion of map unit susceptible to liquefaction
        p_ml[liq_susc==liq_susc_enum['very_high'].value] = 0.25
        p_ml[liq_susc==liq_susc_enum['high'].value] = 0.20
        p_ml[liq_susc==liq_susc_enum['moderate'].value] = 0.10
        p_ml[liq_susc==liq_susc_enum['low'].value] = 0.05
        p_ml[liq_susc==liq_susc_enum['very_low'].value] = 0.02
        p_ml[liq_susc==liq_susc_enum['none'].value] = 0.00

        # liquefaction likelihood, p_liq
        prob_liq = prob_liq_pga / k_mag / k_gw_depth * p_ml # decimal, eq. 4-20
        prob_liq = np.maximum(prob_liq,zero_prob_liq) # set prob to > "0" to avoid 0% in log

        # Zhu et al. (2017) boundary constraints
        # for pga_mag < 0.1 g, set prob to "0"
        prob_liq[pga_mag<0.1] = zero_prob_liq
        # for vs30 > 620 m/s, set prob to "0"
        prob_liq[self.vs30>620] = zero_prob_liq
        # for precip > 1700 mm, set prob to "0"
        prob_liq[self.precip>1700] = zero_prob_liq

        return {"liq_prob":prob_liq, "liq_susc":liq_susc}



## Lateral Spreading:
class LateralSpread:
    def __init__(self) -> None:
        pass

# -----------------------------------------------------------
class Hazus2020Lateral(LateralSpread):
    """
    Compute lateral spreading, same methodology as Grant et al. (2016).
    
    Parameters
    ----------
    From upstream PBEE:
    pga: float, np.ndarray or list
        [g] peak ground acceleration
    mag: float, np.ndarray or list
        moment magnitude
        
    Geotechnical/geologic:
    prob_liq: float, np.ndarray or list
        probability of liquefaction
    dist_water: float, np.ndarray or list, optional
        [km] distance to nearest river, lake, or coast; site is only susceptible to lateral spread if distance is less than 25 meters
    
    Fixed:
    liq_susc: str, np.ndarray or list
        susceptibility category to liquefaction (none, very low, low, moderate, high, very high)

    Returns
    -------
    pgdef : float, np.ndarray
        [m] permanent ground deformation
    sigma_pgdef : float, np.ndarray
        aleatory variability for ln(pgdef)
    
    References
    ----------
    .. [1] Federal Emergency Management Agency (FEMA), 2020, Hazus Earthquake Model - Technical Manual, Hazus 4.2 SP3, 436 pp. https://www.fema.gov/flood-maps/tools-resources/flood-map-products/hazus/user-technical-manuals.
    
    """
    def __init__(self, stations, parameters):
        super().__init__()
        self.stations = stations
        dist_to_water = parameters.get("DistWater")
        if type(dist_to_water) == np.array:
            self.dist_to_water = dist_to_water
        elif dist_to_water == "Defined (\"distWater\") in Site File (.csv)":
            self.dist_to_water = np.array([site['distWater'] for site in self.stations])
        elif os.path.exists(os.path.dirname(dist_to_water)):
            lat_station = [site['lat'] for site in self.stations]
            lon_station = [site['lon'] for site in self.stations]
            self.dist_to_water = sampleRaster(dist_to_water, \
                    parameters["inputCRS"],lon_station, lat_station)
        else:
            self.dist_to_water = np.zeros(len(self.stations))
        

    def run(self, ln_im_data, eq_data, im_list):
        output_keys = ['liq_PGD_h']
        if ('PGA' in im_list)  and ('liq_prob' in im_list) and \
            ('liq_susc' in im_list):
            num_stations = len(self.stations)
            num_scenarios = len(eq_data)
            PGA_col_id = [i for i, x in enumerate(im_list) if x == 'PGA'][0]
            liq_prob_col_id = [i for i, x in enumerate(im_list) if \
                               x == 'liq_prob'][0]
            liq_susc_col_id = [i for i, x in enumerate(im_list) if \
                               x == 'liq_susc'][0]
            for scenario_id in range(num_scenarios):
                num_rlzs = ln_im_data[scenario_id].shape[2]
                im_data_scen = np.zeros([num_stations,\
                                    len(im_list)+len(output_keys), num_rlzs])
                im_data_scen[:,0:len(im_list),:] = ln_im_data[scenario_id]
                for rlz_id in range(num_rlzs):
                    liq_prob = ln_im_data[scenario_id][:,liq_prob_col_id,rlz_id]
                    liq_susc = ln_im_data[scenario_id][:,liq_susc_col_id,rlz_id]
                    pga = np.exp(ln_im_data[scenario_id][:,PGA_col_id,rlz_id])
                    mag = float(eq_data[scenario_id][0])
                    model_output = self.model(pga, mag, liq_prob, \
                        self.dist_to_water, liq_susc)
                    for i, key in enumerate(output_keys):
                        im_data_scen[:,len(im_list)+i,rlz_id] = model_output[key]
                ln_im_data[scenario_id] = im_data_scen
            im_list = im_list + output_keys
        else:
            sys.exit(f"At least one of 'PGA' and 'PGV' is missing in the selected intensity measures and the liquefaction trigging model 'ZhuEtal2017' can not be computed.")
        return ln_im_data, eq_data, im_list

    @staticmethod
    # @njit
    def model(
        pga, mag, # upstream PBEE RV
        prob_liq, dist_water, # geotechnical/geologic
        liq_susc, # fixed/toggles
        extrapolate_expected_pgdef=True
    ):
        """Model"""
        
        # initialize arrays
        
        # get threshold pga against liquefaction
        pga_t = np.ones(pga.shape)*np.nan
        pga_t[liq_susc==liq_susc_enum['very_high'].value] = 0.09 # g
        pga_t[liq_susc==liq_susc_enum['high'].value] = 0.12 # g
        pga_t[liq_susc==liq_susc_enum['moderate'].value] = 0.15 # g
        pga_t[liq_susc==liq_susc_enum['low'].value] = 0.21 # g
        pga_t[liq_susc==liq_susc_enum['very_low'].value] = 0.26 # g
        pga_t[liq_susc==liq_susc_enum['none'].value] = 1. # g
        
        # pga factor of safety
        ratio = pga/pga_t\
        
        # get normalized displacement in inches, a, for M=7
        expected_pgdef = np.ones(pga.shape)*np.nan
        expected_pgdef[ratio<=1] = 1e-3 # above 1e-3 cm, or 1e-5 m
        expected_pgdef[np.logical_and(ratio>1,ratio<=2)] = 12*ratio[np.logical_and(ratio>1,ratio<=2)] - 12
        expected_pgdef[np.logical_and(ratio>2,ratio<=3)] = 18*ratio[np.logical_and(ratio>2,ratio<=3)] - 24
        if extrapolate_expected_pgdef is True:
            expected_pgdef[ratio>3] = 70*ratio[ratio>3] - 180
        else:
            expected_pgdef[np.logical_and(ratio>3,ratio<=4)] = 70*ratio[np.logical_and(ratio>3,ratio<=4)] - 180
            expected_pgdef[ratio>4] = 100
        expected_pgdef *= 2.54 # convert from inches to cm
        
        # magnitude correction
        k_delta = 0.0086*mag**3 - 0.0914*mag**2 + 0.4698*mag - 0.9835
        
        # susceptibility to lateral spreading only for deposits found near water body (dw < dw_cutoff)
        pgdef = k_delta * expected_pgdef * prob_liq
        pgdef = pgdef/100 # also convert from cm to m
        pgdef[dist_water>25] = 1e-5
        
        # keep pgdef to minimum of 1e-5 m
        pgdef = np.maximum(pgdef,1e-5)
        
        # prepare outputs
        output = {'liq_PGD_h':pgdef}
        # get intermediate values if requested
        # if return_inter_params:
        #     output['k_delta'] = k_delta
        #     output['expected_pgdef'] = expected_pgdef
        #     output['pga_t'] = pga_t
        #     output['ratio'] = ratio
        
        # return
        return output
    

## Settlement:
class GroundSettlement:
    def __init__(self) -> None:
        pass


class Hazus2020Vertical(GroundSettlement):
    """
    Compute volumetric settlement at a given location using a simplified deterministic approach (after Tokimatsu and Seed, 1987).
    
    Parameters
    ----------
    From upstream PBEE:
    
    Geotechnical/geologic:
    prob_liq: float, np.ndarray or list
        probability of liquefaction
    
    Fixed:
    liq_susc: str, np.ndarray or list
        susceptibility category to liquefaction (none, very low, low, moderate, high, very high)

    Returns
    -------
    pgdef : float, np.ndarray
        [m] permanent ground deformation
    sigma_pgdef : float, np.ndarray
        aleatory variability for ln(pgdef)
    
    References
    ----------
    .. [1] Federal Emergency Management Agency (FEMA), 2020, Hazus Earthquake Model - Technical Manual, Hazus 4.2 SP3, 436 pp. https://www.fema.gov/flood-maps/tools-resources/flood-map-products/hazus/user-technical-manuals.
    .. [2] Tokimatsu, K., and Seed, H.B., 1987, Evaluation of Settlements in Sands Due to Earthquake Shaking. Journal of Geotechnical Engineering, vol. 113, no. 8, pp. 861-878.

    
    """
    @staticmethod
    # @njit
    def model(
        prob_liq, # geotechnical/geologic
        liq_susc, # fixed/toggles
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        
        # initialize arrays
        # get threshold pga against liquefaction, in cm
        pgdef = np.ones(liq_susc.shape)*np.nan
        pgdef[liq_susc==liq_susc_enum['very_high'].value] = 30
        pgdef[liq_susc==liq_susc_enum['high'].value] = 15
        pgdef[liq_susc==liq_susc_enum['moderate'].value] = 5
        pgdef[liq_susc==liq_susc_enum['low'].value] = 2.5
        pgdef[liq_susc==liq_susc_enum['very_low'].value] = 1
        pgdef[liq_susc==liq_susc_enum['none'].value] = 1e-3
        
        # condition with prob_liq
        pgdef = pgdef * prob_liq
        
        # convert from cm to m
        pgdef = pgdef/100
        
        # limit deformations to 1e-5
        pgdef = np.maximum(pgdef,1e-5)
        
        # prepare outputs
        output = {'liq_PGD_v':pgdef}
        # get intermediate values if requested
        if return_inter_params:
            pass
        
        # return
        return output
    
    def run(self, ln_im_data, eq_data, im_list):
        output_keys = ['liq_PGD_v']
        if ('liq_susc' in im_list)  and ('liq_prob' in im_list):
            num_stations = ln_im_data[0].shape[0]
            num_scenarios = len(eq_data)
            liq_prob_col_id = [i for i, x in enumerate(im_list) if \
                               x == 'liq_prob'][0]
            liq_susc_col_id = [i for i, x in enumerate(im_list) if \
                               x == 'liq_susc'][0]
            for scenario_id in range(num_scenarios):
                num_rlzs = ln_im_data[scenario_id].shape[2]
                im_data_scen = np.zeros([num_stations,\
                                    len(im_list)+len(output_keys), num_rlzs])
                im_data_scen[:,0:len(im_list),:] = ln_im_data[scenario_id]
                for rlz_id in range(num_rlzs):
                    liq_prob = ln_im_data[scenario_id][:,liq_prob_col_id,rlz_id]
                    liq_susc = ln_im_data[scenario_id][:,liq_susc_col_id,rlz_id]
                    model_output = self.model(liq_prob, liq_susc)
                    for i, key in enumerate(output_keys):
                        im_data_scen[:,len(im_list)+i,rlz_id] = model_output[key]
                ln_im_data[scenario_id] = im_data_scen
            im_list = im_list + output_keys
        else:
            sys.exit(f"At least one of 'liq_susc' and 'liq_prob' is missing in the selected intensity measures and the liquefaction trigging model 'ZhuEtal2017' can not be computed.")
        return ln_im_data, eq_data, im_list