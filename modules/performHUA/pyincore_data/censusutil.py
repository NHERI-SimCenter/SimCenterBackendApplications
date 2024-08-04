# Based on the IN-CORE censusutil method  # noqa: INP001, D100
# Modified by Dr. Stevan Gavrilovic, UC Berkeley, SimCenter


# Copyright (c) 2021 University of Illinois and others. All rights reserved.
#
# This program and the accompanying materials are made available under the
# terms of the Mozilla Public License v2.0 which accompanies this distribution,
# and is available at https://www.mozilla.org/en-US/MPL/2.0/

import os
import urllib.request
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
import requests
from pyincore_data import globals

logger = globals.LOGGER


class CensusUtil:
    """Utility methods for Census data and API"""  # noqa: D400

    @staticmethod
    def generate_census_api_url(  # noqa: ANN205
        state: str = None,  # noqa: RUF013
        county: str = None,  # noqa: RUF013
        year: str = None,  # noqa: RUF013
        data_source: str = None,  # noqa: RUF013
        columns: str = None,  # noqa: RUF013
        geo_type: str = None,  # noqa: RUF013
        data_name: str = None,  # noqa: RUF013
    ):
        """Create url string to access census data api.

        Args:
        ----
            state (str): A string of state FIPS with comma separated format. e.g, '41, 42' or '*'
            county (str): A string of county FIPS with comma separated format. e.g, '017,029,045,091,101' or '*'
            year (str): Census Year.
            data_source (str): Census dataset name. Can be found from https://api.census.gov/data.html
            columns (str): Column names for request data with comma separated format.
                e.g, 'GEO_ID,NAME,P005001,P005003,P005004,P005010'
            geo_type (str): Name of geo area. e.g, 'tract:*' or 'block%20group:*'
            data_name (str): Optional for getting different dataset. e.g, 'component'

        Returns
        -------
            string: A string for representing census api url

        """
        # check if the state is not none
        if state is None:
            error_msg = 'State value must be provided.'
            logger.error(error_msg)
            raise Exception(error_msg)  # noqa: DOC501, TRY002

        if geo_type is not None:
            if county is None:
                error_msg = 'State and county value must be provided when geo_type is provided.'
                logger.error(error_msg)
                raise Exception(error_msg)  # noqa: DOC501, TRY002

        # Set up url for Census API
        base_url = f'https://api.census.gov/data/{year}/{data_source}'
        if data_name is not None:
            base_url = (
                f'https://api.census.gov/data/{year}/{data_source}/{data_name}'
            )

        data_url = f'{base_url}?get={columns}'
        if county is None:  # only state is provided. There shouldn't be any geo_type
            data_url = f'{data_url}&for=state:{state}'
        elif geo_type is None:
            data_url = f'{data_url}&in=state:{state}&for=county:{county}'
        else:
            data_url = (
                f'{data_url}&for={geo_type}&in=state:{state}&in=county:{county}'
            )

        return data_url

    @staticmethod
    def request_census_api(data_url):  # noqa: ANN001, ANN205
        """Request census data to api and gets the output data

        Args:
        ----
            data_url (str): url for obtaining the data from census api
        Returns:
            dict, object: A json list and a dataframe for census api result

        """  # noqa: D400
        # Obtain Census API JSON Data
        request_json = requests.get(data_url)  # noqa: S113

        if request_json.status_code != 200:  # noqa: PLR2004
            error_msg = 'Failed to download the data from Census API. Please check your parameters.'
            # logger.error(error_msg)
            raise Exception(error_msg)  # noqa: DOC501, TRY002

        # Convert the requested json into pandas dataframe

        api_json = request_json.json()
        api_df = pd.DataFrame(columns=api_json[0], data=api_json[1:])

        return api_df  # noqa: RET504

    @staticmethod
    def get_blockdata_for_demographics(  # noqa: ANN205, C901
        state_counties: list,
        census_vars: list,
        vintage: str = '2010',
        out_csv: bool = False,  # noqa: FBT001, FBT002
        out_shapefile: bool = False,  # noqa: FBT001, FBT002
        out_geopackage: bool = False,  # noqa: FBT001, FBT002
        out_geojson: bool = False,  # noqa: FBT001, FBT002
        file_name: str = 'file_name',
        output_dir: str = 'output_dir',
    ):
        """Generate population demographics dataset from census

        Args:
        ----
            state_counties (list): A List of concatenated State and County FIPS Codes.
                see full list https://www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697
            vintage (str): Census Year.
            out_csv (bool): Save output dataframe as csv.
            out_shapefile (bool): Save processed census geodataframe as shapefile.
            out_geopackage (bool): Save processed census geodataframe as geopackage
            out_geojson (bool): Save processed census geodataframe as geojson
            file_name (str): Name of the output files.
            output_dir (str): Name of directory used to save output files.

        """  # noqa: D400
        # ***********************
        # Get the population data
        # ***********************

        # dataset_name (str): Census dataset name.
        dataset_name = 'dec'

        get_pop_vars = 'GEO_ID,NAME'
        int_vars = census_vars

        if vintage == '2000' or vintage == '2010':  # noqa: PLR1714
            dataset_name += '/sf1'

            # If no variable parameters passed by the user, use the default for 2000 and 2010 vintage
            if not census_vars:
                get_pop_vars += ',P005001,P005003,P005004,P005010'

                # GEO_ID  = Geographic ID
                # NAME    = Geographic Area Name
                # P005001 = Total
                # P005003 = Total!!Not Hispanic or Latino!!White alone
                # P005004 = Total!!Not Hispanic or Latino!!Black or African American alone
                # P005010 = Total!!Hispanic or Latino

                # List variables to convert from dtype object to integer
                int_vars = ['P005001', 'P005003', 'P005004', 'P005010']
            else:
                # Add the variables provided by the user
                for var in census_vars:
                    get_pop_vars += ',' + var

        elif vintage == '2020':
            dataset_name += '/pl'

            # Variable parameters
            # If no variable parameters passed by the user, use the default for 2000 and 2010 vintage
            if not census_vars:
                get_pop_vars += ',P2_001N,P2_002N,P2_005N,P2_006N'

                # GEO_ID  = Geographic ID
                # NAME    = Geographic Area Name
                # P2_001N=!!Total:
                # P2_002N=!!Total:!!Hispanic or Latino
                # P2_005N=!!Total:!!Not Hispanic or Latino:!!Population of one race:!!White alone
                # P2_006N=!!Total:!!Not Hispanic or Latino:!!Population of one race:!!Black or African American alone

                # List variables to convert from dtype object to integer
                int_vars = ['P2_001N', 'P2_002N', 'P2_005N', 'P2_006N']
            else:
                # Add the variables provided by the user
                for var in census_vars:
                    get_pop_vars += ',' + var

        else:
            print('Only 2000, 2010, and 2020 decennial census supported')  # noqa: T201
            return None

        # Make directory to save output
        if not os.path.exists(output_dir):  # noqa: PTH110
            os.mkdir(output_dir)  # noqa: PTH102

        # Make a directory to save downloaded shapefiles
        shapefile_dir = Path(output_dir) / 'shapefiletemp'

        if not os.path.exists(shapefile_dir):  # noqa: PTH110
            os.mkdir(shapefile_dir)  # noqa: PTH102

        # Set to hold the states - needed for 2020 census shapefile download
        stateSet = set()  # noqa: N806

        # loop through counties
        appended_countydata = []  # start an empty container for the county data
        for state_county in state_counties:
            # deconcatenate state and county values
            state = state_county[0:2]
            county = state_county[2:5]
            logger.debug('State:  ' + state)  # noqa: G003
            logger.debug('County: ' + county)  # noqa: G003

            # Add the state to the set
            stateSet.add(state)

            # Set up hyperlink for Census API
            api_hyperlink = CensusUtil.generate_census_api_url(
                state, county, vintage, dataset_name, get_pop_vars, 'block:*'
            )

            logger.info('Census API data from: ' + api_hyperlink)  # noqa: G003

            # Obtain Census API JSON Data
            apidf = CensusUtil.request_census_api(api_hyperlink)

            # Append county data makes it possible to have multiple counties
            appended_countydata.append(apidf)

        # Create dataframe from appended county data
        cen_block = pd.concat(appended_countydata, ignore_index=True)

        # Add variable named "Survey" that identifies Census survey program and survey year
        cen_block['Survey'] = vintage + ' ' + dataset_name

        # Set block group FIPS code by concatenating state, county, tract, block fips
        cen_block['blockid'] = (
            cen_block['state']
            + cen_block['county']
            + cen_block['tract']
            + cen_block['block']
        )

        # To avoid problems with how the block group id is read saving it
        # as a string will reduce possibility for future errors
        cen_block['blockidstr'] = cen_block['blockid'].apply(
            lambda x: 'BLOCK' + str(x).zfill(15)
        )

        # Convert variables from dtype object to integer
        for var in int_vars:
            cen_block[var] = cen_block[var].astype(int)
            # cen_block[var] = pd.to_numeric(cen_block[var], errors='coerce').convert_dtypes()
            print(var + ' converted from object to integer')  # noqa: T201

        if (vintage == '2000' or vintage == '2010') and not census_vars:  # noqa: PLR1714
            # Generate new variables
            cen_block['pwhitebg'] = cen_block['P005003'] / cen_block['P005001'] * 100
            cen_block['pblackbg'] = cen_block['P005004'] / cen_block['P005001'] * 100
            cen_block['phispbg'] = cen_block['P005010'] / cen_block['P005001'] * 100

            # GEO_ID  = Geographic ID
            # NAME    = Geographic Area Name
            # P005001 = Total
            # P005003 = Total!!Not Hispanic or Latino!!White alone
            # P005004 = Total!!Not Hispanic or Latino!!Black or African American alone
            # P005010 = Total!!Hispanic or Latino

        elif vintage == '2020' and not census_vars:
            cen_block['pwhitebg'] = cen_block['P2_005N'] / cen_block['P2_001N'] * 100
            cen_block['pblackbg'] = cen_block['P2_006N'] / cen_block['P2_001N'] * 100
            cen_block['phispbg'] = cen_block['P2_002N'] / cen_block['P2_001N'] * 100

            # GEO_ID  = Geographic ID
            # NAME    = Geographic Area Name
            # P2_001N=!!Total:
            # P2_002N=!!Total:!!Hispanic or Latino
            # P2_005N=!!Total:!!Not Hispanic or Latino:!!Population of one race:!!White alone
            # P2_006N=!!Total:!!Not Hispanic or Latino:!!Population of one race:!!Black or African American alone

        # *******************************
        # Download and extract shapefiles
        # *******************************

        # Download the shapefile information for the block groups in the select counties.
        #
        # These files can be found online at:

        # For 2010 Census
        # https://www2.census.gov/geo/tiger/TIGER2010/TABBLOCK/2010/

        # For 2020 Census
        # https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/

        # Block group shapefiles are downloaded for each of the selected counties from
        # the Census TIGER/Line Shapefiles at https://www2.census.gov/geo/tiger.
        # Each counties file is downloaded as a zipfile and the contents are extracted.
        # The shapefiles are reprojected to EPSG 4326 and appended as a single shapefile
        # (as a GeoPandas GeoDataFrame) containing block groups for all of the selected counties.
        #
        # *EPSG: 4326 uses a coordinate system (Lat, Lon)
        # This coordinate system is required for mapping with folium.

        appended_shp_files = []  # start an empty container for the county shapefiles

        merge_id = 'GEOID' + vintage[2:4]

        # Tigerline provides the blocks for each county, thus each county needs to be downloaded individually
        if vintage == '2000' or vintage == '2010':  # noqa: PLR1714
            if vintage == '2000':
                merge_id = 'BLKIDFP00'

            # loop through counties
            for state_county in state_counties:
                # county_fips = state+county
                filename = f'tl_2010_{state_county}_tabblock' + vintage[2:4]

                # Use wget to download the TIGER Shapefile for a county
                # options -quiet = turn off wget output
                # add directory prefix to save files to folder named after program name
                shapefile_url = (
                    f'https://www2.census.gov/geo/tiger/TIGER2010/TABBLOCK/{vintage}/'
                    + filename
                    + '.zip'
                )
                print(  # noqa: T201
                    (
                        'Downloading Census Block Shapefiles for State_County: '
                        + state_county
                        + ' from: '
                        + shapefile_url
                    ).format(filename=filename)
                )

                zip_file = os.path.join(shapefile_dir, filename + '.zip')  # noqa: PTH118
                urllib.request.urlretrieve(shapefile_url, zip_file)  # noqa: S310

                with ZipFile(zip_file, 'r') as zip_obj:
                    zip_obj.extractall(path=shapefile_dir)

                # Delete the zip file
                os.remove(zip_file)  # noqa: PTH107

                if Path(zip_file).is_file() == True:  # noqa: E712
                    print('Error deleting the zip file ', zip_file)  # noqa: T201

                print('filename', f'{filename}.shp')  # noqa: T201

                # Read shapefile to GeoDataFrame
                gdf = gpd.read_file(f'{shapefile_dir}/{filename}.shp')

                # Set projection to EPSG 4326, which is required for folium
                gdf = gdf.to_crs(epsg=4326)

                # Append county data
                appended_shp_files.append(gdf)

        elif vintage == '2020':
            # loop through the states
            for state in stateSet:
                filename = f'tl_2020_{state}_tabblock20'

                # Check if file is cached
                path = Path(f'{shapefile_dir}/{filename}.shp')

                # if file does not exist
                if path.is_file() == False:  # noqa: E712
                    # Use wget to download the TIGER Shapefile for a county
                    # options -quiet = turn off wget output
                    # add directory prefix to save files to folder named after program name
                    shapefile_url = (
                        'https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/'
                        + filename
                        + '.zip'
                    )

                    print(  # noqa: T201
                        (
                            'Downloading Census Block Shapefiles for State: '
                            + state
                            + ' from: '
                            + shapefile_url
                        ).format(filename=filename)
                    )

                    zip_file = os.path.join(shapefile_dir, filename + '.zip')  # noqa: PTH118
                    urllib.request.urlretrieve(shapefile_url, zip_file)  # noqa: S310

                    with ZipFile(zip_file, 'r') as zip_obj:
                        zip_obj.extractall(path=shapefile_dir)

                    # Delete the zip file
                    os.remove(zip_file)  # noqa: PTH107

                    if Path(zip_file).is_file() == True:  # noqa: E712
                        print('Error deleting the zip file ', zip_file)  # noqa: T201

                else:
                    print(f'Found file {filename}.shp in cache')  # noqa: T201

                # Read shapefile to GeoDataFrame
                gdf = gpd.read_file(f'{shapefile_dir}/{filename}.shp')

                # Set projection to EPSG 4326, which is required for folium
                gdf = gdf.to_crs(epsg=4326)

                # Append county data
                appended_shp_files.append(gdf)

        # Create dataframe from appended block files
        shp_block = pd.concat(appended_shp_files)

        print(  # noqa: T201
            'Merging the census population demographics information to the shapefile'
        )

        # Clean Data - Merge Census demographic data to the appended shapefiles
        cen_shp_block_merged = pd.merge(  # noqa: PD015
            shp_block, cen_block, left_on=merge_id, right_on='blockid', how='left'
        )

        # Set parameters for file save
        save_columns = [
            'blockid',
            'blockidstr',
            'Survey',
        ]  # set column names to save

        if not census_vars:
            save_columns.extend(['pblackbg', 'phispbg', 'pwhitebg'])

        # ### Explore Data - Map merged block group shapefile and Census data

        savefile = file_name  # set file name

        if out_csv:
            CensusUtil.convert_dislocation_pd_to_csv(
                cen_block, save_columns, output_dir, savefile
            )

        if out_shapefile:
            CensusUtil.convert_dislocation_gpd_to_shapefile(
                cen_shp_block_merged, output_dir, savefile
            )

        if out_geopackage:
            CensusUtil.convert_dislocation_gpd_to_geopackage(
                cen_shp_block_merged, output_dir, savefile
            )

        if out_geojson:
            CensusUtil.convert_dislocation_gpd_to_geojson(
                cen_shp_block_merged, output_dir, savefile
            )

        # clean up shapefile temp directory
        # Try to remove tree; if failed show an error using try...except on screen
        #        try:
        #            shutil.rmtree(shapefile_dir)
        #            if not out_shapefile and not out_csv and not out_html and not out_geopackage and not out_geojson:
        #                shutil.rmtree(output_dir)
        #        except OSError as e:
        #            error_msg = "Error: Failed to remove either " + shapefile_dir \
        #                        + " or " + output_dir + " directory"
        #            logger.error(error_msg)
        #            raise Exception(error_msg)

        print('Done creating population demographics shapefile')  # noqa: T201

        return cen_block[save_columns]

    @staticmethod
    def get_blockgroupdata_for_income(  # noqa: ANN205, C901
        state_counties: list,
        acs_vars: list,
        vintage: str = '2010',
        out_csv: bool = False,  # noqa: FBT001, FBT002
        out_shapefile: bool = False,  # noqa: FBT001, FBT002
        out_geopackage: bool = False,  # noqa: FBT001, FBT002
        out_geojson: bool = False,  # noqa: FBT001, FBT002
        file_name: str = 'file_name',
        output_dir: str = 'output_dir',
    ):
        """Generate household income dataset from census

        Args:
        ----
            state_counties (list): A List of concatenated State and County FIPS Codes.
                see full list https://www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697
            vintage (str): Census Year.
            out_csv (bool): Save output dataframe as csv.
            out_shapefile (bool): Save processed census geodataframe as shapefile.
            out_geopackage (bool): Save processed census geodataframe as geopackage
            out_geojson (bool): Save processed census geodataframe as geojson
            file_name (str): Name of the output files.
            output_dir (str): Name of directory used to save output files.

        """  # noqa: D400
        # dataset_name (str): ACS dataset name.
        dataset_name = 'acs/acs5'

        # *****************************
        # Get the household income data
        # *****************************

        get_income_vars = 'GEO_ID,NAME'
        int_vars = acs_vars

        # Use the default vars if none provided by the user
        if not acs_vars:
            # Income data variable tags for 2010, 2015, and 2020 5-year ACS
            # B19001_001E - Estimate!!Total
            # B19001_002E - Estimate!!Total!!Less than $10,000
            # B19001_003E - Estimate!!Total!!$10,000 to $14,999
            # B19001_004E - Estimate!!Total!!$15,000 to $19,999
            # B19001_005E - Estimate!!Total!!$20,000 to $24,999
            # B19001_006E - Estimate!!Total!!$25,000 to $29,999
            # B19001_007E - Estimate!!Total!!$30,000 to $34,999
            # B19001_008E - Estimate!!Total!!$35,000 to $39,999
            # B19001_009E - Estimate!!Total!!$40,000 to $44,999
            # B19001_010E - Estimate!!Total!!$45,000 to $49,999
            # B19001_011E - Estimate!!Total!!$50,000 to $59,999
            # B19001_012E - Estimate!!Total!!$60,000 to $74,999
            # B19001_013E - Estimate!!Total!!$75,000 to $99,999
            # B19001_014E - Estimate!!Total!!$100,000 to $124,999
            # B19001_015E - Estimate!!Total!!$125,000 to $149,999
            # B19001_016E - Estimate!!Total!!$150,000 to $199,999
            # B19001_017E - Estimate!!Total!!$200,000 or more
            # B19013_001E - Estimate!!Median household income in the past 12 months (in 2016 inflation-adjusted dollars)

            get_income_vars += ',B19001_001E,B19001_002E,B19001_003E,B19001_004E,\
B19001_005E,B19001_006E,B19001_007E,B19001_008E,B19001_009E,B19001_010E,\
B19001_011E,B19001_012E,B19001_013E,B19001_014E,B19001_015E,\
B19001_016E,B19001_017E,B19013_001E'

            int_vars = [
                'B19001_001E',
                'B19001_002E',
                'B19001_003E',
                'B19001_004E',
                'B19001_005E',
                'B19001_006E',
                'B19001_007E',
                'B19001_008E',
                'B19001_009E',
                'B19001_010E',
                'B19001_011E',
                'B19001_012E',
                'B19001_013E',
                'B19001_014E',
                'B19001_015E',
                'B19001_016E',
                'B19001_017E',
                'B19013_001E',
            ]

        else:
            # Add the variables provided by the user
            for var in acs_vars:
                get_income_vars += ',' + var

        # Make directory to save output
        if not os.path.exists(output_dir):  # noqa: PTH110
            os.mkdir(output_dir)  # noqa: PTH102

        # Make a directory to save downloaded shapefiles
        shapefile_dir = Path(output_dir) / 'shapefiletemp'

        if not os.path.exists(shapefile_dir):  # noqa: PTH110
            os.mkdir(shapefile_dir)  # noqa: PTH102

        # Set to hold the states - needed for 2020 census shapefile download
        stateSet = set()  # noqa: N806

        # loop through counties
        appended_countydata = []  # start an empty container for the county data
        for state_county in state_counties:
            # deconcatenate state and county values
            state = state_county[0:2]
            county = state_county[2:5]
            logger.debug('State:  ' + state)  # noqa: G003
            logger.debug('County: ' + county)  # noqa: G003

            # Add the state to the set
            stateSet.add(state)

            # Set up hyperlink for Census API
            api_hyperlink = ''

            if vintage == '2010':
                api_hyperlink = CensusUtil.generate_census_api_url(
                    state, county, vintage, dataset_name, get_income_vars, 'tract'
                )
            else:
                # Set up hyperlink for Census API
                api_hyperlink = CensusUtil.generate_census_api_url(
                    state,
                    county,
                    vintage,
                    dataset_name,
                    get_income_vars,
                    'block%20group',
                )

            logger.info('Census API data from: ' + api_hyperlink)  # noqa: G003

            # Obtain Census API JSON Data
            apidf = CensusUtil.request_census_api(api_hyperlink)

            # Append county data makes it possible to have multiple counties
            appended_countydata.append(apidf)

        # Create dataframe from appended county data
        cen_blockgroup = pd.concat(appended_countydata, ignore_index=True)

        # Add variable named "Survey" that identifies Census survey program and survey year
        cen_blockgroup['Survey'] = vintage + ' ' + dataset_name

        # 2010 ACS API does not support block group level resolution, use tract
        if vintage == '2010':
            # Set tract FIPS code by concatenating state, county, and tract
            cen_blockgroup['tractid'] = (
                cen_blockgroup['state']
                + cen_blockgroup['county']
                + cen_blockgroup['tract']
            )

            # To avoid problems with how the tract id is read saving it
            # as a string will reduce possibility for future errors
            cen_blockgroup['tractidstr'] = cen_blockgroup['tractid'].apply(
                lambda x: 'TRACT' + str(x).zfill(11)
            )
        else:
            # Set block group FIPS code by concatenating state, county, tract and block group fips
            cen_blockgroup['bgid'] = (
                cen_blockgroup['state']
                + cen_blockgroup['county']
                + cen_blockgroup['tract']
                + cen_blockgroup['block group']
            )

            # To avoid problems with how the block group id is read saving it
            # as a string will reduce possibility for future errors
            cen_blockgroup['bgidstr'] = cen_blockgroup['bgid'].apply(
                lambda x: 'BG' + str(x).zfill(12)
            )

        # Convert variables from dtype object to integer
        for var in int_vars:
            cen_blockgroup[var] = pd.to_numeric(
                cen_blockgroup[var], errors='coerce'
            ).convert_dtypes()
            # cen_blockgroup[var] = cen_blockgroup[var].astype(int)
            print(var + ' converted from object to integer')  # noqa: T201

        # ### Obtain Data - Download and extract shapefiles
        # The Block Group IDs in the Census data are associated with the Block Group boundaries that can be mapped.
        # To map this data, we need the shapefile information for the block groups in the select counties.
        #
        # These files can be found online at:
        # https://www2.census.gov/geo/tiger/TIGER2010/BG/2010/

        # *******************************
        # Download and extract shapefiles
        # *******************************

        # Download the shapefile information for the block groups in the select counties.
        #
        # These files can be found online at:

        # For 2010 ACS - API only supports up to the tract level
        # https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/

        # For 2015 and 2020 ACS - API supports up to the block group level
        # https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/

        # Block group shapefiles are downloaded for each of the selected counties from
        # the Census TIGER/Line Shapefiles at https://www2.census.gov/geo/tiger.

        # Each state/counties file is downloaded as a zipfile and the contents are extracted.
        # The shapefiles are reprojected to EPSG 4326 and appended as a single shapefile
        # (as a GeoPandas GeoDataFrame) containing block groups for all of the selected counties.
        #
        # *EPSG: 4326 uses a coordinate system (Lat, Lon)
        # This coordinate system is required for mapping with folium.

        appended_shp_files = []  # start an empty container for the county/state shapefiles

        # Feature attributes that need to match to join layers
        merge_id_left = 'GEOID'
        merge_id_right = ''

        # Tigerline provides the blocks for each county, thus each county needs to be downloaded individually
        if vintage == '2010':
            merge_id_left += '10'

            merge_id_right = 'tractid'

            # loop through counties
            for state_county in state_counties:
                # county_fips = state+county
                filename = f'tl_2010_{state_county}_tract10'

                # Use wget to download the TIGER Shapefile for a county
                # options -quiet = turn off wget output
                # add directory prefix to save files to folder named after program name
                shapefile_url = (
                    'https://www2.census.gov/geo/tiger/TIGER2010/TRACT/2010/'
                    + filename
                    + '.zip'
                )

                print(  # noqa: T201
                    (
                        'Downloading Census Block Shapefiles for State_County: '
                        + state_county
                        + ' from: '
                        + shapefile_url
                    ).format(filename=filename)
                )

                zip_file = os.path.join(shapefile_dir, filename + '.zip')  # noqa: PTH118
                urllib.request.urlretrieve(shapefile_url, zip_file)  # noqa: S310

                with ZipFile(zip_file, 'r') as zip_obj:
                    zip_obj.extractall(path=shapefile_dir)

                # Delete the zip file
                os.remove(zip_file)  # noqa: PTH107

                if Path(zip_file).is_file() == True:  # noqa: E712
                    print('Error deleting the zip file ', zip_file)  # noqa: T201

                # Read shapefile to GeoDataFrame
                gdf = gpd.read_file(f'{shapefile_dir}/{filename}.shp')

                # Set projection to EPSG 4326, which is required for folium
                gdf = gdf.to_crs(epsg=4326)

                # Append county data
                appended_shp_files.append(gdf)

        elif vintage == '2015' or vintage == '2020':  # noqa: PLR1714
            merge_id_right = 'bgid'

            # loop through the states
            for state in stateSet:
                filename = f'tl_{vintage}_{state}_bg'

                # Check if file is cached
                path = Path(f'{shapefile_dir}/{filename}.shp')

                # if file does not exist
                if path.is_file() == False:  # noqa: E712
                    # Use wget to download the TIGER Shapefile for the state
                    # options -quiet = turn off wget output
                    # add directory prefix to save files to folder named after program name
                    shapefile_url = (
                        f'https://www2.census.gov/geo/tiger/TIGER{vintage}/BG/'
                        + filename
                        + '.zip'
                    )

                    print(  # noqa: T201
                        (
                            'Downloading Census Block Shapefiles for State: '
                            + state
                            + ' from: '
                            + shapefile_url
                        ).format(filename=filename)
                    )

                    zip_file = os.path.join(shapefile_dir, filename + '.zip')  # noqa: PTH118
                    urllib.request.urlretrieve(shapefile_url, zip_file)  # noqa: S310

                    with ZipFile(zip_file, 'r') as zip_obj:
                        zip_obj.extractall(path=shapefile_dir)

                    # Delete the zip file
                    os.remove(zip_file)  # noqa: PTH107

                    if Path(zip_file).is_file() == True:  # noqa: E712
                        print('Error deleting the zip file ', zip_file)  # noqa: T201

                else:
                    print(f'Found file {filename}.shp in cache: ', path)  # noqa: T201

                # Read shapefile to GeoDataFrame
                gdf = gpd.read_file(f'{shapefile_dir}/{filename}.shp')

                # Set projection to EPSG 4326, which is required for folium
                gdf = gdf.to_crs(epsg=4326)

                # Append county data
                appended_shp_files.append(gdf)

        # Create dataframe from appended county data
        shp_blockgroup = pd.concat(appended_shp_files)

        print('Merging the ACS household income information to the shapefile')  # noqa: T201

        # Clean Data - Merge Census demographic data to the appended shapefiles
        cen_shp_blockgroup_merged = pd.merge(  # noqa: PD015
            shp_blockgroup,
            cen_blockgroup,
            left_on=merge_id_left,
            right_on=merge_id_right,
            how='left',
        )

        # Set parameters for file save
        if vintage == '2010':
            save_columns = [
                'tractid',
                'tractidstr',
                'Survey',
            ]  # set column names to save
        else:
            save_columns = ['bgid', 'bgidstr', 'Survey']  # set column names to save

        # ### Explore Data - Map merged block group shapefile and Census data

        savefile = file_name  # set file name

        if out_csv:
            CensusUtil.convert_dislocation_pd_to_csv(
                cen_blockgroup, save_columns, output_dir, savefile
            )

        if out_shapefile:
            CensusUtil.convert_dislocation_gpd_to_shapefile(
                cen_shp_blockgroup_merged, output_dir, savefile
            )

        if out_geopackage:
            CensusUtil.convert_dislocation_gpd_to_geopackage(
                cen_shp_blockgroup_merged, output_dir, savefile
            )

        if out_geojson:
            CensusUtil.convert_dislocation_gpd_to_geojson(
                cen_shp_blockgroup_merged, output_dir, savefile
            )

        # clean up shapefile temp directory
        # Try to remove tree; if failed show an error using try...except on screen
        #        try:
        #            shutil.rmtree(shapefile_dir)
        #            if not out_shapefile and not out_csv and not out_html and not out_geopackage and not out_geojson:
        #                shutil.rmtree(output_dir)
        #        except OSError as e:
        #            error_msg = "Error: Failed to remove either " + shapefile_dir \
        #                        + " or " + output_dir + " directory"
        #            logger.error(error_msg)
        #            raise Exception(error_msg)

        print('Done creating household income shapefile')  # noqa: T201

        return cen_blockgroup[save_columns]

    @staticmethod
    def convert_dislocation_gpd_to_shapefile(in_gpd, programname, savefile):  # noqa: ANN001, ANN205
        """Create shapefile of dislocation geodataframe.

        Args:
        ----
            in_gpd (object): Geodataframe of the dislocation.
            programname (str): Output directory name.
            savefile (str): Output shapefile name.

        """
        # save cen_shp_blockgroup_merged shapefile
        print(  # noqa: T201
            'Shapefile data file saved to: ' + programname + '/' + savefile + '.shp'
        )
        in_gpd.to_file(programname + '/' + savefile + '.shp')

    @staticmethod
    def convert_dislocation_gpd_to_geojson(in_gpd, programname, savefile):  # noqa: ANN001, ANN205
        """Create geojson of dislocation geodataframe.

        Args:
        ----
            in_gpd (object): Geodataframe of the dislocation.
            programname (str): Output directory name.
            savefile (str): Output geojson name.

        """
        # save cen_shp_blockgroup_merged geojson
        print(  # noqa: T201
            'Geodatabase data file saved to: '
            + programname
            + '/'
            + savefile
            + '.geojson'
        )
        in_gpd.to_file(programname + '/' + savefile + '.geojson', driver='GeoJSON')

    @staticmethod
    def convert_dislocation_gpd_to_geopackage(in_gpd, programname, savefile):  # noqa: ANN001, ANN205
        """Create shapefile of dislocation geodataframe.

        Args:
        ----
            in_gpd (object): Geodataframe of the dislocation.
            programname (str): Output directory name.
            savefile (str): Output shapefile name.

        """
        # save cen_shp_blockgroup_merged shapefile
        print(  # noqa: T201
            'GeoPackage data file saved to: '
            + programname
            + '/'
            + savefile
            + '.gpkg'
        )
        in_gpd.to_file(
            programname + '/' + savefile + '.gpkg', driver='GPKG', layer=savefile
        )

    @staticmethod
    def convert_dislocation_pd_to_csv(in_pd, save_columns, programname, savefile):  # noqa: ANN001, ANN205
        """Create csv of dislocation dataframe using the column names.

        Args:
        ----
            in_pd (object): Geodataframe of the dislocation.
            save_columns (list): A list of column names to use.
            programname (str): Output directory name.
            savefile (str): Output csv file name.

        """
        # Save cen_blockgroup dataframe with save_column variables to csv named savefile
        print('CSV data file saved to: ' + programname + '/' + savefile + '.csv')  # noqa: T201
        in_pd[save_columns].to_csv(
            programname + '/' + savefile + '.csv', index=False
        )
