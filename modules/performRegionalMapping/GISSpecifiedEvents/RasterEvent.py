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

import os
import argparse  # noqa: I001
import json, csv  # noqa: E401
from pathlib import Path
import rasterio
import pyproj
from rasterio.transform import rowcol


def sample_raster_at_latlon(src, lat, lon):  # noqa: D103
    # Get the row and column indices in the raster
    row, col = rowcol(src.transform, lon, lat)  # Note the order: lon, lat

    # Ensure the indices are within the bounds of the raster
    if row < 0 or row >= src.height or col < 0 or col >= src.width:
        raise IndexError('Transformed coordinates are out of raster bounds')  # noqa: EM101, TRY003

    # Read the raster value at the given row and column for all layers NOT just the first
    #raster_value = src.read(1)[row, col]
    data = src.read()
    raster_values = data[:, row, col]
    return raster_values  # noqa: RET504


def create_event(asset_file, event_grid_file, num_entry):  # noqa: C901, D103, N803, RUF100

    #print(f'asset_file: {asset_file}, entry: {num_entry}')    
    #print(f'event_grid_file: {event_grid_file}, entry: {num_entry}')
    
    # read the event grid data file
    event_grid_path = Path(event_grid_file).resolve()
    event_dir = event_grid_path.parent
    event_grid_file = event_grid_path.name

    src = rasterio.open(event_grid_path)

    # Get the raster's CRS
    raster_crs = pyproj.CRS.from_wkt(src.crs.to_wkt())

    # Define the source CRS (EPSG:4326)
    src_crs = pyproj.CRS('EPSG:4326')

    # Transform the lat/lon to the raster's coordinate system
    transformer = pyproj.Transformer.from_crs(src_crs, raster_crs, always_xy=True)

    # iterate through the assets and store the selected events in the AIM
    with open(asset_file, encoding='utf-8') as f:  # noqa: PTH123
        asset_dict = json.load(f)

    data_final = [
        ['GP_file', 'Latitude', 'Longitude'],
    ]

    # Iterate through each asset
    for asset in asset_dict:
        asset_id = asset['id']
        asset_file_path = asset['file']

        # Load the corresponding file for each asset
        with open(asset_file_path, encoding='utf-8') as asset_file:  # noqa: PTH123, PLR1704
            # Load the asset data
            asset_data = json.load(asset_file)

            if num_entry == 0:
                im_tag = asset_data['RegionalEvent']['intensityMeasures']
                units = asset_data['RegionalEvent']['units']
            else:
                im_tag = asset_data['RegionalEvent']['multiple'][num_entry-1]['intensityMeasures']
                units = asset_data['RegionalEvent']['multiple'][num_entry-1]['units']
            

            # Extract the latitude and longitude
            lat = float(asset_data['GeneralInformation']['location']['latitude'])
            lon = float(asset_data['GeneralInformation']['location']['longitude'])

            # Transform the coordinates
            lon_transformed, lat_transformed = transformer.transform(lon, lat)

            # Check if the transformed coordinates are within the raster bounds
            bounds = src.bounds
            if (
                bounds.left <= lon_transformed <= bounds.right
                and bounds.bottom <= lat_transformed <= bounds.top
            ):
                try:
                    val = sample_raster_at_latlon(
                        src=src, lat=lat_transformed, lon=lon_transformed
                    )
                    
                    data = [im_tag, val]

                    # Save the simcenter file name
                    file_name = f'Site_{asset_id}.csvx{0}x{int(asset_id):05d}'

                    data_final.append([file_name, lat, lon])

                    csv_save_path = event_dir / f'Site_{asset_id}.csv'

                    if num_entry == 0:

                        # if first entry
                        #    create the csv file and add the data
                        
                        with open(csv_save_path, 'w', newline='') as file:  # noqa: PTH123
                            # Create a CSV writer object
                            writer = csv.writer(file)

                            # Write the data to the CSV file
                            writer.writerows(data)
                    else:

                        # subsequent entries
                        #  read existing file, append header and row data, 
                        #  and finally write new file with updated data
                        #
                        
                        # Read the existing file
                        if os.path.exists(csv_save_path):
                            with open(csv_save_path, mode='r') as f:
                                reader = csv.DictReader(f)
                                rows = list(reader)
                                fieldnames = reader.fieldnames or []
                        else:
                            rows = []
                            fieldnames = []

                        # extend field names and row data with additional stuff to be added
                        # IS IM_TAG a single value or an array .. should be array!

                        extra = dict(zip(im_tag, val))
                        for k in extra:
                            if k not in fieldnames:
                                fieldnames.append(k)
                        for row in rows:
                            row.update(extra)
                        
                        # Overwrite existing file
                        with open(csv_save_path, mode='w', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(rows)                        


                    # prepare a dictionary of events
                    event_list_json = [[file_name, 1.0]]


                    # in asset file, add event info including now units
                    if num_entry == 0:

                        # if first, add an event field
                        asset_data['Events'] = [{}]
                        asset_data['Events'][0] = {
                            'EventFolderPath': str(event_dir),
                            'Events': event_list_json,
                            'type': 'intensityMeasure',
                            'units': units
                        }

                    else:
                        
                        # if additional, update units to include new                       
                        asset_data['Events'][0]['units'].update(units)

                    with open(asset_file_path, 'w', encoding='utf-8') as f:  # noqa: PTH123
                        json.dump(asset_data, f, indent=2)

                except IndexError as e:
                    print(f'Error for asset ID {asset_id}: {e}')  # noqa: T201
            else:
                print(f'Asset ID: {asset_id} is outside the raster bounds')  # noqa: T201

        # # save the event dictionary to the BIM
        # asset_data['Events'] = [{}]
        # asset_data['Events'][0] = {
        #     # "EventClassification": "Earthquake",
        #     'EventFolderPath': str(event_dir),
        #     'Events': event_list_json,
        #     'type': event_type,
        #     # "type": "SimCenterEvents"
        # }

        # with open(asset_file, 'w', encoding='utf-8') as f:  # noqa: PTH123, RUF100
        #     json.dump(asset_data, f, indent=2)

    # Save the final event grid
    csv_save_path = event_dir / 'EventGrid.csv'
    with open(csv_save_path, 'w', newline='') as file:  # noqa: PTH123
        # Create a CSV writer object
        writer = csv.writer(file)

        # Write the data to the CSV file
        writer.writerows(data_final)

    # Perform cleanup
    src.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assetFile')
    parser.add_argument('--filenameEVENTgrid')
    args = parser.parse_args()

    create_event(args.assetFile, args.filenameEVENTgrid)
