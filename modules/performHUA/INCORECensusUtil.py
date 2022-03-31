# -*- coding: utf-8 -*-
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

import os
import sys
import argparse, posixpath, json
  
from pyincore_data.censusutil import CensusUtil

if __name__ == '__main__':
     
    print('Pulling census data')


    parser = argparse.ArgumentParser()
    parser.add_argument('--census_config')
    args = parser.parse_args()
    with open(args.census_config) as f:
        config_info = json.load(f)

    # Output directory
    output_dir = config_info['OutputDirectory']
    
    try:
        os.mkdir(f"{output_dir}")
    except:
        print('Output folder already exists.')
    
    
    # State counties, e.g., ['01001', '01003']
    state_counties = config_info['CountiesArray']


    # Vintage, e.g., "2010"
    vintage = config_info['Vintage']


    disloc_df = CensusUtil.get_blockgroupdata_for_dislocation(state_counties, vintage,out_csv=False, out_shapefile=True, out_geopackage=False,out_geojson=False,geo_name="CensusData"+vintage , program_name=output_dir)

   
   
    print('Done pulling census data')
    
    sys.exit(0)
