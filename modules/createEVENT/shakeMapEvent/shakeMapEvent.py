#  # noqa: INP001, D100
# Copyright (c) 2018 Leland Stanford Junior University
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
# Stevan Gavrilovic
#

import argparse  # noqa: I001
import xml.etree.ElementTree as ET
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path


class IntensityMeasureTypeError(Exception):
    def __init__(self, im_type):
        super().__init__(f'Intensity measure type {im_type} not found in grid data')


def create_shakemap_event(eventDirectory, eventPath, IMTypes):  # noqa: D103, N803
    IMTypesList = eval(IMTypes)  # noqa: S307, N806

    print('Creating shakemap event')  # noqa: T201

    xml_file_path = Path(eventPath) / 'grid.xml'

    # Parse the XML file
    tree = ET.parse(xml_file_path)  # noqa: S314
    root = tree.getroot()

    # Find the grid_data element
    grid_data = root.find('{http://earthquake.usgs.gov/eqcenter/shakemap}grid_data')

    # Prepare lists to store data
    points = []
    attributes = []

    # Get the attribute_mapping
    namespace = {'ns': 'http://earthquake.usgs.gov/eqcenter/shakemap'}
    grid_fields = {}
    for grid_field in root.findall('ns:grid_field', namespace):
        index = grid_field.get('index')
        name = grid_field.get('name')
        units = grid_field.get('units')
        grid_fields[name] = {'index': index, 'units': units}
    attribute_mapping = {}
    for im_type in ['PGA', 'PGV', 'MMI', 'PSA03', 'PSA10', 'PSA30']:
        if im_type not in grid_fields:
            raise IntensityMeasureTypeError(im_type)
        attribute_mapping[im_type] = int(grid_fields[im_type]['index']) - 1
    # Parse the grid data
    for line in grid_data.text.strip().split('\n'):
        values = line.split()
        lon, lat = float(values[0]), float(values[1])
        point = Point(lon, lat)
        points.append(point)

        # Store only the specified attributes
        attr = {}
        for im_type in IMTypesList:
            if im_type in attribute_mapping:
                attr[im_type] = float(values[attribute_mapping[im_type]])

        attributes.append(attr)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(attributes, geometry=points, crs='EPSG:4326')

    if 'PGA' in gdf.columns:
        gdf['PGA'] = gdf['PGA'] / 100 # convert from pct g to g

    if 'PSA03' in gdf.columns:
        gdf['PSA03'] = gdf['PSA03'] / 100 # convert from pct g to g
        gdf = gdf.rename(columns={'PSA03': 'SA_0.3'})
    if 'PSA10' in gdf.columns:
        gdf['PSA10'] = gdf['PSA10'] / 100 # convert from pct g to g
        gdf = gdf.rename(columns={'PSA10': 'SA_1.0'})

    # Display the first few rows
    print('Saving shakemap to gpkg')  # noqa: T201

    # Save as a GeoPackage file
    gdf_path = Path(eventPath) / 'EventGrid.gpkg'
    gdf.to_file(gdf_path, driver='GPKG')

    return  # noqa: PLR1711


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Input file')
    parser.add_argument('--Directory', help='Directory path')
    parser.add_argument('--EventPath', help='Event path')
    parser.add_argument('--IntensityMeasureType', help='types of intensity measures')

    args = parser.parse_args()

    create_shakemap_event(args.Directory, args.EventPath, args.IntensityMeasureType)
