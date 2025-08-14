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
import json
import argparse  # noqa: I001
from pathlib import Path
import xml.etree.ElementTree as ET

from RasterEvent import create_event as create_raster_event


def is_raster_file(filename):  # noqa: D103
    # Define a set of common raster file extensions
    raster_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif'}

    # Create a Path object from the filename
    file_path = Path(filename)

    # Extract the file extension and check if it is in the set of raster extensions
    return file_path.suffix.lower() in raster_extensions


def is_xml_file(filename):  # noqa: D103
    # Check if the file has an .xml extension
    if not filename.lower().endswith('.xml'):
        return False

    # Try to parse the file as XML
    try:
        ET.parse(filename)  # noqa: S314
        return True  # noqa: TRY300
    except ET.ParseError:
        return False


def create_event(asset_file: str, event_grid_file: str, workflow_input: str, do_parallel: bool):  # noqa: C901, D103, N803, RUF100

    event_grid_file_path = os.path.dirname(event_grid_file)

    #
    # open input file & get Regional Event data
    #

    json_path = os.path.join(os.getcwd(), workflow_input)

    with open(json_path, 'r') as f:
	data = json.load(f)

    regional_event = data.get("RegionalEvent")
    
    if is_raster_file(event_grid_file):
        create_raster_event(asset_file, event_grid_file, 0, do_parallel, regional_event)
    elif is_xml_file(event_grid_file):  # noqa: RET505
        # Here you would call a function to handle XML files
        # For now, we'll just raise a NotImplementedError
        raise NotImplementedError('XML file handling is not yet implemented.')  # noqa: EM101
    else:
        raise ValueError(  # noqa: TRY003
            f'{event_grid_file} is not a raster. Only rasters are currently supported.'  # noqa: EM102
        )

    #
    # If multiple exists, update event_file
    #   note: create_raster_event modified for this purpose
    #
    
    #print(f'ORIGINAL: {event_grid_file}')
    
    multiple_entries = data.get("RegionalEvent", {}).get("multiple", [])
    for i, entry in enumerate(multiple_entries):

        # is this assumption correct on file paths?
        next_file = data['RegionalEvent']['multiple'][i]['eventFile']
        next_file_path = os.path.join(event_grid_file_path, next_file)
        create_raster_event(asset_file, next_file_path, i+1, do_parallel, regional_event)
    
#
# main function
#

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assetFile')
    parser.add_argument('--filenameEVENTgrid')
    parser.add_argument('--workflowInput')
    parser.add_argument('--doParallel', default='False')
    parser.add_argument('-n', '--numP', default='8')
    parser.add_argument('-m', '--mpiExec', default='mpiexec')    
    args = parser.parse_args()

    create_event(args.assetFile,
                 args.filenameEVENTgrid,
                 args.workflowInput,
                 args.doParallel)
