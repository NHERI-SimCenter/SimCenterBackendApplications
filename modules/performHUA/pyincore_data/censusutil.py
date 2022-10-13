# Copyright (c) 2021 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

# Modified by Dr. Stevan Gavrilovic, UC Berkeley, SimCenter to reduce number of required dependencies

import json
from math import isnan

import requests
import os
import pandas as pd
import geopandas as gpd
import urllib.request
import shutil

from pyincore_data import globals
from zipfile import ZipFile

logger = globals.LOGGER

class CensusUtil():
    """Utility methods for Census data and API"""

    @staticmethod
    def get_census_data(state:str=None, county:str=None, year:str=None, data_source:str=None, columns:str=None,
                         geo_type:str = None, data_name:str = None):
        """Create json and pandas DataFrame for census api request result.

            Args:
                state (str): A string of state FIPS with comma separated format. e.g, '41, 42' or '*'
                county (str): A string of county FIPS with comma separated format. e.g, '017,029,045,091,101' or '*'
                year (str): Census Year.
                data_source (str): Census dataset name. Can be found from https://api.census.gov/data.html
                columns (str): Column names for request data with comma separated format.
                    e.g, 'GEO_ID,NAME,P005001,P005003,P005004,P005010'
                geo_type (str): Name of geo area. e.g, 'tract:*' or 'block%20group:*'
                data_name (str): Optional for getting different dataset. e.g, 'component'

            Returns:
                dict, object: A json list and a dataframe for census api result

        """
        # create census api data url
        data_url = CensusUtil.generate_census_api_url(state, county, year, data_source, columns, geo_type, data_name)

        api_json, api_df = CensusUtil.request_census_api(data_url)

        return api_json, api_df

    @staticmethod
    def generate_census_api_url(state:str=None, county:str=None, year:str=None, data_source:str=None, columns:str=None,
                                geo_type:str = None, data_name:str = None):
        """Create url string to access census data api.

            Args:
                state (str): A string of state FIPS with comma separated format. e.g, '41, 42' or '*'
                county (str): A string of county FIPS with comma separated format. e.g, '017,029,045,091,101' or '*'
                year (str): Census Year.
                data_source (str): Census dataset name. Can be found from https://api.census.gov/data.html
                columns (str): Column names for request data with comma separated format.
                    e.g, 'GEO_ID,NAME,P005001,P005003,P005004,P005010'
                geo_type (str): Name of geo area. e.g, 'tract:*' or 'block%20group:*'
                data_name (str): Optional for getting different dataset. e.g, 'component'

            Returns:
                string: A string for representing census api url

        """
        # check if the state is not none
        if state is None:
            error_msg = "State value must be provided."
            logger.error(error_msg)
            raise Exception(error_msg)

        if geo_type is not None:
            if county is None:
                error_msg = "State and county value must be provided when geo_type is provided."
                logger.error(error_msg)
                raise Exception(error_msg)

        # Set up url for Census API
        base_url = f'https://api.census.gov/data/{year}/{data_source}'
        if data_name is not None:
            base_url = f'https://api.census.gov/data/{year}/{data_source}/{data_name}'

        data_url = f'{base_url}?get={columns}'
        if county is None:  # only state is provided. There shouldn't be any geo_type
            data_url = f'{data_url}&for=state:{state}'
        else: # county has been provided and there could be geo_type or not
            if geo_type is None:
                data_url = f'{data_url}&in=state:{state}&for=county:{county}'
            else:
                data_url = f'{data_url}&for={geo_type}&in=state:{state}&in=county:{county}'

        return data_url

    @staticmethod
    def request_census_api(data_url):
        """Request census data to api and gets the output data

            Args:
                data_url (str): url for obtaining the data from census api
            Returns:
                dict, object: A json list and a dataframe for census api result

        """
        # Obtain Census API JSON Data
        request_json = requests.get(data_url)

        if request_json.status_code != 200:
            error_msg = "Failed to download the data from Census API. Please check your parameters."
            # logger.error(error_msg)
            raise Exception(error_msg)

        # Convert the requested json into pandas dataframe

        api_json = request_json.json()
        api_df = pd.DataFrame(columns=api_json[0], data=api_json[1:])

        return api_json, api_df

    @staticmethod
    def get_fips_by_state_county(state:str, county:str, year:str=2010):
        """Get FIPS code by using state and county name.

            Args:
                state (str): State name. e.g, 'illinois'
                county (str): County name. e.g, 'champaign'
            Returns:
                str: A string of FIPS code

        """
        api_url = f'https://api.census.gov/data/{year}/dec/sf1?get=NAME&for=county:*'
        out_fips = None
        api_json = requests.get(api_url)
        query_value = county + ' County, ' + state
        if api_json.status_code != 200:
            error_msg = "Failed to download the data from Census API. Please look up Google for getting the FIPS code."
            raise Exception(error_msg)

        # content_json = api_json.json()
        df = pd.DataFrame(columns=api_json.json()[0], data=api_json.json()[1:])
        selected_row = df.loc[df['NAME'].str.lower() == query_value.lower()]
        if selected_row.size > 0:
            out_fips = selected_row.iloc[0]['state'] + selected_row.iloc[0]['county']
        else:
            error_msg = "There is no FIPS code for given state and county combination."
            logger.error(error_msg)
            raise Exception(error_msg)

        return out_fips

    @staticmethod
    def get_fips_by_state(state:str, year:str=2010):
        """Create Geopandas DataFrame for population dislocation analysis from census dataset.

            Args:
                state (str): State name. e.g, 'illinois'

            Returns:
                obj: A json list of county FIPS code in the given state

        """
        api_url = f'https://api.census.gov/data/{year}/dec/sf1?get=NAME&for=county:*'
        api_json = requests.get(api_url)
        if api_json.status_code != 200:
            error_msg = "Failed to download the data from Census API."
            logger.error(error_msg)
            raise Exception(error_msg)

        # content_json = api_json.json()
        out_fips = api_json.json()

        return out_fips

    @staticmethod
    def get_blockgroupdata_for_dislocation(state_counties: list, vintage: str = "2010", dataset_name: str = 'dec/sf1',
                                           out_csv: bool = False, out_shapefile: bool = False,
                                           out_geopackage: bool = False, out_geojson: bool = False,
                                           geo_name: str = "geo_name", program_name: str = "program_name"):

        """Create Geopandas DataFrame for population dislocation analysis from census dataset.

        Args:
            state_counties (list): A List of concatenated State and County FIPS Codes.
                see full list https://www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697
            vintage (str): Census Year.
            dataset_name (str): Census dataset name.
            out_csv (bool): Save output dataframe as csv.
            out_shapefile (bool): Save processed census geodataframe as shapefile.
            out_geopackage (bool): Save processed census geodataframe as geopackage
            out_geojson (bool): Save processed census geodataframe as geojson
            geo_name (str): Name of geo area - used for naming output files.
            program_name (str): Name of directory used to save output files.

        Returns:
            obj, dict: A dataframe for dislocation analysis, and
            a dictionary containing geodataframe and folium map

        """
        
        if vintage == '2000' or '2010' :
            # Variable parameters
            get_vars = 'GEO_ID,NAME,P005001,P005003,P005004,P005010'
            
            # List variables to convert from dtype object to integer
            int_vars = ['P005001', 'P005003', 'P005004', 'P005010']
            
            # GEO_ID  = Geographic ID
            # NAME    = Geographic Area Name
            # P005001 = Total
            # P005003 = Total!!Not Hispanic or Latino!!White alone
            # P005004 = Total!!Not Hispanic or Latino!!Black or African American alone
            # P005010 = Total!!Hispanic or Latino
#        elif vintage == '2020' :
            # TODO: 2020 support
        else :
            print('Only 2000 and 2010 decennial census supported')
            return

        # Make directory to save output
        if not os.path.exists(program_name):
            os.mkdir(program_name)

        # Make a directory to save downloaded shapefiles - folder will be made then deleted
        shapefile_dir = 'shapefiletemp'
        if not os.path.exists(shapefile_dir):
            os.mkdir(shapefile_dir)

        # loop through counties
        appended_countydata = []  # start an empty container for the county data
        for state_county in state_counties:
            # deconcatenate state and county values
            state = state_county[0:2]
            county = state_county[2:5]
            logger.debug('State:  '+state)
            logger.debug('County: '+county)

            # Set up hyperlink for Census API
            api_hyperlink = CensusUtil.generate_census_api_url(
                state, county, vintage, dataset_name, get_vars, 'block%20group')

            logger.info("Census API data from: " + api_hyperlink)

            # Obtain Census API JSON Data
            apijson, apidf = CensusUtil.request_census_api(api_hyperlink)
            print(apidf.size)
            # Append county data makes it possible to have multiple counties
            appended_countydata.append(apidf)

        # Create dataframe from appended county data
        cen_blockgroup = pd.concat(appended_countydata, ignore_index=True)

        # Add variable named "Survey" that identifies Census survey program and survey year
        cen_blockgroup['Survey'] = vintage+' '+dataset_name

        # Set block group FIPS code by concatenating state, county, tract and block group fips
        cen_blockgroup['bgid'] = (cen_blockgroup['state']+cen_blockgroup['county'] +
                                  cen_blockgroup['tract']+cen_blockgroup['block group'])

        # To avoid problems with how the block group id is read saving it
        # as a string will reduce possibility for future errors
        cen_blockgroup['bgidstr'] = cen_blockgroup['bgid'].apply(lambda x: "BG"+str(x).zfill(12))

        # Convert variables from dtype object to integer
        for var in int_vars:
            cen_blockgroup[var] = cen_blockgroup[var].astype(int)
            print(var+' converted from object to integer')

        # Generate new variables
        cen_blockgroup['pwhitebg'] = cen_blockgroup['P005003'] / cen_blockgroup['P005001'] * 100
        cen_blockgroup['pblackbg'] = cen_blockgroup['P005004'] / cen_blockgroup['P005001'] * 100
        cen_blockgroup['phispbg'] = cen_blockgroup['P005010'] / cen_blockgroup['P005001'] * 100

        # ### Obtain Data - Download and extract shapefiles
        # The Block Group IDs in the Census data are associated with the Block Group boundaries that can be mapped.
        # To map this data, we need the shapefile information for the block groups in the select counties.
        #
        # These files can be found online at:
        # https://www2.census.gov/geo/tiger/TIGER2010/BG/2010/

        # ### Download and extract shapefiles
        # Block group shapefiles are downloaded for each of the selected counties from
        # the Census TIGER/Line Shapefiles at https://www2.census.gov/geo/tiger.
        # Each counties file is downloaded as a zipfile and the contents are extracted.
        # The shapefiles are reprojected to EPSG 4326 and appended as a single shapefile
        # (as a GeoPandas GeoDataFrame) containing block groups for all of the selected counties.
        #
        # *EPSG: 4326 uses a coordinate system (Lat, Lon)
        # This coordinate system is required for mapping with folium.

        # loop through counties
        appended_countyshp = []  # start an empty container for the county shapefiles
        for state_county in state_counties:

            # county_fips = state+county
            filename = f'tl_2010_{state_county}_bg10'

            # Use wget to download the TIGER Shapefile for a county
            # options -quiet = turn off wget output
            # add directory prefix to save files to folder named after program name
            shapefile_url = 'https://www2.census.gov/geo/tiger/TIGER2010/BG/2010/' + filename + '.zip'
            print(('Downloading Shapefiles for State_County: '
                   + state_county + ' from: '+shapefile_url).format(filename=filename))

            zip_file = os.path.join(shapefile_dir, filename + '.zip')
            urllib.request.urlretrieve(shapefile_url, zip_file)

            with ZipFile(zip_file, 'r') as zip_obj:
                zip_obj.extractall(path="shapefiletemp")
            # Read shapefile to GeoDataFrame
            gdf = gpd.read_file(f'shapefiletemp/{filename}.shp')

            # Set projection to EPSG 4326, which is required for folium
            gdf = gdf.to_crs(epsg=4326)

            # Append county data
            appended_countyshp.append(gdf)

        # Create dataframe from appended county data
        shp_blockgroup = pd.concat(appended_countyshp)

        # Clean Data - Merge Census demographic data to the appended shapefiles
        cen_shp_blockgroup_merged = pd.merge(shp_blockgroup, cen_blockgroup,
                                             left_on='GEOID10', right_on='bgid', how='left')

        # Set paramaters for file save
        save_columns = ['bgid', 'bgidstr', 'Survey', 'pblackbg', 'phispbg']  # set column names to save

        # ### Explore Data - Map merged block group shapefile and Census data

        savefile = geo_name  # set file name

        if out_csv:
            CensusUtil.convert_dislocation_pd_to_csv(cen_blockgroup, save_columns, program_name, savefile)

        if out_shapefile:
            CensusUtil.convert_dislocation_gpd_to_shapefile(cen_shp_blockgroup_merged, program_name, savefile)

        if out_geopackage:
            CensusUtil.convert_dislocation_gpd_to_geopackage(cen_shp_blockgroup_merged, program_name, savefile)
            
        if out_geojson:
            CensusUtil.convert_dislocation_gpd_to_geojson(cen_shp_blockgroup_merged, program_name, savefile)

        # clean up shapefile temp directory
        # Try to remove tree; if failed show an error using try...except on screen
        try:
            shutil.rmtree(shapefile_dir)
            if not out_shapefile and not out_csv and not out_html and not out_geopackage and not out_geojson:
                shutil.rmtree(program_name)
        except OSError as e:
            error_msg = "Error: Failed to remove either " + shapefile_dir \
                        + " or " + program_name + " directory"
            logger.error(error_msg)
            raise Exception(error_msg)

        return cen_blockgroup[save_columns]

    @staticmethod
    def convert_dislocation_gpd_to_shapefile(in_gpd, programname, savefile):
        """Create shapefile of dislocation geodataframe.

        Args:
            in_gpd (object): Geodataframe of the dislocation.
            programname (str): Output directory name.
            savefile (str): Output shapefile name.

        """
        # save cen_shp_blockgroup_merged shapefile
        print('Shapefile data file saved to: '+programname+'/'+savefile+".shp")
        in_gpd.to_file(programname+'/'+savefile+".shp")


    @staticmethod
    def convert_dislocation_gpd_to_geojson(in_gpd, programname, savefile):
        """Create geojson of dislocation geodataframe.

        Args:
            in_gpd (object): Geodataframe of the dislocation.
            programname (str): Output directory name.
            savefile (str): Output geojson name.

        """
        # save cen_shp_blockgroup_merged geojson
        print('Geodatabase data file saved to: '+programname+'/'+savefile+".geojson")
        in_gpd.to_file(programname+'/'+savefile+".geojson", driver="GeoJSON")


    @staticmethod
    def convert_dislocation_gpd_to_geopackage(in_gpd, programname, savefile):
        """Create shapefile of dislocation geodataframe.

        Args:
            in_gpd (object): Geodataframe of the dislocation.
            programname (str): Output directory name.
            savefile (str): Output shapefile name.

        """
        # save cen_shp_blockgroup_merged shapefile
        print('GeoPackage data file saved to: '+programname+'/'+savefile+".gpkg")
        in_gpd.to_file(programname+'/'+savefile+".gpkg", driver="GPKG", layer=savefile)

    @staticmethod
    def convert_dislocation_pd_to_csv(in_pd, save_columns, programname, savefile):
        """Create csv of dislocation dataframe using the column names.

        Args:
            in_pd (object): Geodataframe of the dislocation.
            save_columns (list): A list of column names to use.
            programname (str): Output directory name.
            savefile (str): Output csv file name.

        """

        # Save cen_blockgroup dataframe with save_column variables to csv named savefile
        print('CSV data file saved to: '+programname+'/'+savefile+".csv")
        in_pd[save_columns].to_csv(programname+'/'+savefile+".csv", index=False)



    @staticmethod
    def create_choro_data_from_pd(pd, key):
        """Create choropleth's choro-data from dataframe.

        Args:
            pd (object): an Input dataframe.
            key (str): a string for dictionary key
        Returns:
            obj : A dictionary of dataframe

        """
        temp_id = list(range(len(pd[key])))
        temp_id = [str(i) for i in temp_id]
        choro_data = dict(zip(temp_id, pd[key]))
        # check the minimum value to use it to nan value, since nan value makes an error.
        min_val = pd[key].min()
        for item in choro_data:
            if isnan(choro_data[item]):
                choro_data[item] = 0

        return choro_data

    
