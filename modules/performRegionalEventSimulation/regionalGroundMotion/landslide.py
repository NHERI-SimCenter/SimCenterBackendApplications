import numpy as np
import rasterio as rio
from scipy.interpolate import interp2d
import sys, warnings, shapely, pandas, os
from pyproj import Transformer
from pyproj import CRS
from enum import Enum
import geopandas as gpd
from scipy.spatial import ConvexHull
import pandas as pd

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
    if vector_gdf.crs != vector_crs:
        sys.exit(f"The CRS of vector file {vector_file_path} is {vector_gdf.crs}, and doesn't match the input CRS ({xy_crs}) defined for liquefaction triggering models")
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

def infer_from_geologic_map(map_path, map_crs, lon_station, lat_station):
    gdf_units = sampleVector(map_path, map_crs, lon_station, lat_station, dtype = None)
    gdf_units = gdf_units['UnitAbbr', 'geometry']
    gdf_units = gdf_units.fillna('water')
    default_geo_prop_fpath = os.path.join(os.path.abspath(__file__), 'database',\
            'groundfailure', 'Wills_etal_2015_CA_Geologic_Properties.csv')
    default_geo_prop = pd.read_csv(default_geo_prop_fpath)
    unique_geo_unit = np.unique(gdf_units['UnitAbbr'])
    phi_mean = np.empty_like(gdf_units['UnitAbbr'])
    coh_mean = np.empty_like(gdf_units['UnitAbbr'])
    for each in unique_geo_unit:
        rows_with_geo_unit = np.where(gdf_units['UnitAbbr'].values==each)[0]
        rows_for_param = np.where(default_geo_prop['Unit Abbreviation'].values==each)[0][0]
        phi_mean[rows_with_geo_unit] = \
            default_geo_prop['Friction Angle - Median (degrees)'][rows_for_param]
        coh_mean[rows_with_geo_unit] = \
            default_geo_prop['Cohesion - Median (kPa)'][rows_for_param]
    return phi_mean, coh_mean

def erf2(x):
    """modified from https://www.johndcook.com/blog/2009/01/19/stand-alone-error-function-erf"""
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    # Save the sign of x
    signs = np.sign(x)
    x = np.abs(x)
    # A & S 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x**2)
    return signs*y

def norm2_cdf(x, loc, scale):
    """
    modified implementation of norm.cdf function from numba_stats, using self-implemented erf function
    https://github.com/HDembinski/numba-stats/blob/main/src/numba_stats/norm.py
    """
    inter = (x - loc)/scale
    return 0.5 * (1 + erf2(inter * np.sqrt(0.5)))

def erf2_2d(x):
    """modified from https://www.johndcook.com/blog/2009/01/19/stand-alone-error-function-erf"""
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    # Save the sign of x
    signs = np.sign(x)
    x = np.abs(x)
    # A & S 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*np.exp(-x**2)
    return signs*y

def norm2_cdf_2d(x, loc, scale):
    """
    modified implementation of norm.cdf function from numba_stats, using self-implemented erf function
    https://github.com/HDembinski/numba-stats/blob/main/src/numba_stats/norm.py
    """
    inter = (x - loc)/scale
    return 0.5 * (1 + erf2_2d(inter * np.sqrt(0.5)))

def nb_round(x, decimals):
    out = np.empty_like(x)
    return np.round_(x, decimals, out)

def erfinv_coeff(order=20):
    # initialize
    c = np.empty(order+1)
    # starting value
    c[0] = 1
    for i in range(1,order+1):
        c[i] = sum([c[j]*c[i-1-j]/(j+1)/(2*j+1) for j in range(i)])
    # return
    return c

def erfinv(x, order=20):
    """returns inverse erf(x)"""
    # get coeffcients
    c = erfinv_coeff(order)
    # initialize
    root_pi_over_2 = np.sqrt(np.pi)/2
    y = np.zeros(x.shape)
    for i in range(order):
        y += c[i]/(2*i+1)*(root_pi_over_2*x)**(2*i+1)
    # return
    return y

def norm2_ppf(p, loc, scale):
    """
    modified implementation of norm.ppf function from numba_stats, using self-implemented erfinv function
    https://github.com/HDembinski/numba-stats/blob/main/src/numba_stats/norm.py
    """
    inter = np.sqrt(2) * erfinv(2*p-1,order=20)
    return scale * inter + loc

def erfinv_2d(x, order=20):
    """returns inverse erf(x)"""
    # get coeffcients
    c = erfinv_coeff(order)
    # initialize
    root_pi_over_2 = np.sqrt(np.pi)/2
    y = np.zeros(x.shape)
    for i in range(order):
        y += c[i]/(2*i+1)*(root_pi_over_2*x)**(2*i+1)
    # return
    return y

def norm2_ppf_2d(p, loc, scale):
    """
    modified implementation of norm.ppf function from numba_stats, using self-implemented erfinv function
    https://github.com/HDembinski/numba-stats/blob/main/src/numba_stats/norm.py
    """
    inter = np.sqrt(2) * erfinv_2d(2*p-1,order=20)
    return scale * inter + loc

class Landslide:
    def __init__(self) -> None:
        pass

# -----------------------------------------------------------
class BrayMacedo2019(Landslide):
    """
    Compute landslide deformation at a given location using the Bray and Macedo (2007) probabilistic model.
    Regression models based on three sets of ground motions are provided:

    1. **Ordinary**: **d** = f(ky, Sa(T), Ts, M)
    2. **Near-fault**: **d** = f(ky, Sa(T), Ts, M, pgv) - unavailable for this version of OpenSRA
    3. **General** (default): **d** = f(ky, Sa(T), Ts, M, pgv) - unavailable for this version of OpenSRA

    The default relationship for **ky** uses **coh_soil**, **phi_soil**, **gamma_soil**, **t_slope**, **slope**

    **PGA** is used in place of **Sa(T)** (i.e., Ts=0)

    Parameters
    ----------
    From upstream PBEE:
    pga: float, np.ndarray or list
        [g] peak ground acceleration
    mag: float, np.ndarray or list
        moment magnitude
        
    Geotechnical/geologic:
    slope: float, np.ndarray or list
        [deg] slope angle
    t_slope: float, np.ndarray or list
        [m] slope thickness (infinite-slope problem)
    gamma_soil: float, np.ndarray or list
        [kN/m^3] unit weight of soil
    phi_soil: float, np.ndarray or list
        [deg] friction angle of soil
    coh_soil: float, np.ndarray or list
        [kPa] cohesion of soil
        
    Fixed:

    Returns
    -------
    pgdef : float, np.ndarray
        [m] permanent ground deformation
    sigma_pgdef : float, np.ndarray
        aleatory variability for ln(pgdef)
    
    References
    ----------
    .. [1] Bray, J.D., and Macedo, J., 2019, Procedure for Estimating Shear-Induced Seismic Slope Displacement for Shallow Crustal Earthquakes, Journal of Geotechnical and Geoenvironmental Engineering, vol. 145, pp. 12, 04019106.
    
    """
    def __init__(self, parameters, stations) -> None:
        self.stations = stations
        self.parameters = parameters
        self.slope = None #(km)
        self.t_slope = None #(km)
        self.gamma_soil = None #(km)
        self.phi_soil = None #(m)
        self.coh_soil = None # (mm)
        self.interpolate_spatial_parameters(parameters)
    
    def interpolate_spatial_parameters(self, parameters):
        # site coordinate in CRS 4326
        lat_station = [site['lat'] for site in self.stations]
        lon_station = [site['lon'] for site in self.stations]
        # slope 
        if parameters["Slope"] == "Defined (\"Slope\") in Site File (.csv)":
            self.slope = np.array([site['Slope'] for site in self.stations])
        else:
            self.slope = sampleRaster(parameters["Slope"], parameters["inputCRS"],\
                     lon_station, lat_station)
        # t_slope
        if parameters["SlopeThickness"] == "Defined (\"SlopeThickness\") in Site File (.csv)":
            self.t_slope = np.array([site['SlopeThickness'] for site in self.stations])
        elif parameters["SlopeThickness"] == "Use constant value (m)":
            self.t_slope = np.array(parameters["SlopeThicknessValue"])
        else:
            self.t_slope = sampleRaster(parameters["SlopeThickness"], parameters["inputCRS"],\
                     lon_station, lat_station)
        # gamma_soil
        if parameters["GammaSoil"] == "Defined (\"GammaSoil\") in Site File (.csv)":
            self.gamma_soil = np.array([site['GammaSoil'] for site in self.stations])
        elif parameters["GammaSoil"] == "Use constant value (m)":
            self.gamma_soil = np.array(parameters["GammaSoilValue"])
        elif parameters["GammaSoil"] == "Infer from Geologic Map":
            self.gamma_soil = infer_from_geologic_map(parameters["GammaSoilGeoMap"],\
                                parameters['inputCRS'], lon_station, lat_station)
        else:
            self.gamma_soil = sampleRaster(parameters["GammaSoil"], parameters["inputCRS"],\
                     lon_station, lat_station)
        # coh_soil
        if parameters["CohesionSoil"] == "Defined (\"CohesionSoil\") in Site File (.csv)":
            self.coh_soil = np.array([site['CohesionSoil'] for site in self.stations])
        elif parameters["CohesionSoil"] == "Use constant value (m)":
            self.coh_soil = np.array(parameters["CohesionSoilValue"])
        elif parameters["CohesionSoil"] == "Infer from Geologic Map":
            self.coh_soil = infer_from_geologic_map(parameters["CohesionSoilGeoMap"],\
                                parameters['inputCRS'], lon_station, lat_station)
        else:
            self.coh_soil = sampleRaster(parameters["CohesionSoil"], parameters["inputCRS"],\
                     lon_station, lat_station)

        print("Initiation finished")

    def run(self, ln_im_data, eq_data, im_list, output_keys, additional_output_keys = []):
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
                    model_output = self.model(pga, mag, self.slope, self.t_slope,
                                              self.gamma_soil, self.phi_soil,
                                              self.coh_soil)
                    for i, key in enumerate(output_keys):
                        im_data_scen[:,len(im_list)+i,rlz_id] = model_output[key]
                ln_im_data[scenario_id] = im_data_scen
            im_list = im_list + output_keys
            additional_output = dict()
            for key in additional_output_keys:
                item = getattr(self, key, None)
                if item is None:
                    warnings.warn(f"Additional output {key} is not avaliable in the landslide model 'BrayMacedo2019'.")
                else:
                    additional_output.update({key:item})
        else:
            sys.exit(f"'PGA' is missing in the selected intensity measures and the landslide model 'BrayMacedo2019' can not be computed.")
            # print(f"At least one of 'PGA' and 'PGV' is missing in the selected intensity measures and the liquefaction trigging model 'ZhuEtal2017' can not be computed."\
            #       , file=sys.stderr)
            # sys.stderr.write("test")
            # sys.exit(-1)
        return ln_im_data, eq_data, im_list, additional_output
    
    def model(
        self,
        pga, mag, # upstream PBEE RV
        slope, t_slope, gamma_soil, phi_soil, coh_soil, # geotechnical/geologic
        return_inter_params=False # to get intermediate params
    ):
        """Model"""
        
         # get dimensions
        ndim = pga.ndim
        if ndim == 1:
            n_site = len(pga)
            n_sample = 1
            shape = (n_site)
        else:
            shape = pga.shape
            n_site = shape[0]
            n_sample = shape[1]
            
        # initialize
        pgdef = np.zeros(shape)
        ky = np.zeros(shape)
        prob_d_eq_0 = np.zeros(shape)
        ln_pgdef_trunc = np.zeros(shape)
        nonzero_median_cdf = np.zeros(shape)
        
        # convert from deg to rad
        slope_rad = slope*np.pi/180
        phi_soil_rad = phi_soil*np.pi/180
        
        # yield acceleration
        ky = np.tan(phi_soil_rad-slope_rad) + \
            coh_soil/(
                gamma_soil * t_slope * np.cos(slope_rad)**2 * \
                (1+np.tan(phi_soil_rad)*np.tan(slope_rad)))
        ky = np.maximum(ky,0.01) # to avoid ky = 0

        # aleatory
        sigma_val = 0.72

        # deformation, eq 3b
        ln_pgdef_trunc = \
            -4.684 + \
            -2.482*np.log(ky) + \
            -0.244*(np.log(ky))**2 + \
            0.344*np.log(ky)*np.log(pga) + \
            2.649*np.log(pga) + \
            -0.090*(np.log(pga))**2 + \
            0.603*mag # cm
        nonzero_ln_pgdef = ln_pgdef_trunc.copy()
                
        # probability of zero displacement, eq. 2 with Ts=0
        if ndim == 1:
            prob_d_eq_0 = 1 - norm2_cdf(
                -2.480 + \
                -2.970*np.log(ky) + \
                -0.120*(np.log(ky))**2 + \
                2.780*np.log(pga),
            0, 1)
        else:
            prob_d_eq_0 = 1 - norm2_cdf_2d(
                -2.480 + \
                -2.970*np.log(ky) + \
                -0.120*(np.log(ky))**2 + \
                2.780*np.log(pga),
            0, 1)
        prob_d_eq_0 = nb_round(prob_d_eq_0, decimals=15)
        
        # apply non-zero displacement correction/condition, eq 11
        nonzero_median_cdf = 1 - .5/(1-prob_d_eq_0)
        
        # loop through numper of samples
        if ndim == 1:
            nonzero_ln_pgdef[nonzero_median_cdf>0] = ln_pgdef_trunc[nonzero_median_cdf>0] + \
                sigma_val*norm2_ppf(nonzero_median_cdf[nonzero_median_cdf>0], 0.0, 1.0)
        else:
            for i in range(n_sample):
                cond = nonzero_median_cdf[:,i]>0
                nonzero_ln_pgdef[cond,i] = ln_pgdef_trunc[cond,i] + \
                    sigma_val*norm2_ppf(nonzero_median_cdf[cond,i], 0.0, 1.0)
        
        # rest of actions
        pgdef = np.exp(nonzero_ln_pgdef)/100 # also convert from cm to m
        pgdef = np.maximum(pgdef,1e-5) # limit to 
        output = {'lsd_PGD_h':pgdef}
        return output