#
# Copyright (c) 2018 The Regents of the University of California
#
# This file is part of the SimCenter Backend Applications
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
# this file. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Dr. Stevan Gavrilovic, UC Berkeley
#

import argparse
import importlib
import json
import os
import subprocess
import sys

if __name__ == '__main__':
    print('Pulling census data')

    # Get any missing dependencies
    packageInstalled = False

    import requests

    if not hasattr(requests, 'get'):
        print('Installing the requests package')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'requests'])
        packageInstalled = True

    packages = ['geopandas']
    for p in packages:
        if importlib.util.find_spec(p) is None:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', p])
            packageInstalled = True
            print('Installing the ' + p + ' package')

    if packageInstalled == True:
        print('New packages were installed. Please restart the process.')
        sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--census_config')
    args = parser.parse_args()
    with open(args.census_config) as f:
        config_info = json.load(f)

    # Output directory
    output_dir = config_info['OutputDirectory']

    try:
        os.mkdir(f'{output_dir}')
    except:
        print('Output folder already exists.')

    # State counties, e.g., ['01001', '01003']
    state_counties = config_info['CountiesArray']

    # Population demographics vintage, e.g., "2010"
    popDemoVintage = config_info['PopulationDemographicsVintage']

    # Custom census vars
    census_vars = config_info['CensusVariablesArray']

    # Custom ACS vars
    acs_vars = config_info['ACSVariablesArray']

    if (
        popDemoVintage != '2000'
        and popDemoVintage != '2010'
        and popDemoVintage != '2020'
    ):
        print(
            'Only 2000, 2010, and 2020 decennial census data supported. The provided vintage ',
            popDemoVintage,
            ' is not supported',
        )

        sys.exit(-1)

    # Vintage for household demographics
    houseIncomeVintage = config_info['HouseholdIncomeVintage']

    if (
        houseIncomeVintage != '2010'
        and houseIncomeVintage != '2015'
        and houseIncomeVintage != '2020'
    ):
        print(
            'Only 2010, 2015, and 2020 ACS 5-yr data supported. The provided vintage ',
            houseIncomeVintage,
            ' is not supported',
        )
        sys.exit(-1)

    from pyincore_data.censusutil import CensusUtil

    # Get the population demographics at the block level
    CensusUtil.get_blockdata_for_demographics(
        state_counties,
        census_vars,
        popDemoVintage,
        out_csv=False,
        out_shapefile=True,
        out_geopackage=False,
        out_geojson=False,
        file_name='PopulationDemographicsCensus' + popDemoVintage,
        output_dir=output_dir,
    )

    # sys.exit(0)

    print('Done pulling census population demographics data')

    # Get the household income at the tract (2010 ACS) or block group level (2015 and 2020 ACS)
    CensusUtil.get_blockgroupdata_for_income(
        state_counties,
        acs_vars,
        houseIncomeVintage,
        out_csv=False,
        out_shapefile=True,
        out_geopackage=False,
        out_geojson=False,
        file_name='HouseholdIncomeACS' + houseIncomeVintage,
        output_dir=output_dir,
    )

    print('Done pulling ACS household income data')

    sys.exit(0)
