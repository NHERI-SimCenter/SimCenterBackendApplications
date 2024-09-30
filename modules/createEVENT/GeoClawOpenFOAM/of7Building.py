#  # noqa: INP001, EXE002
# LICENSING INFORMATION
####################################################################
"""LICENSE INFORMATION:

Copyright (c) 2020-2030, The Regents of the University of California (Regents).

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
        list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
        this list of conditions and the following disclaimer in the documentation
        and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the authors and should not be interpreted as representing official policies, either expressed or implied, of the FreeBSD Project.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

"""  # noqa: D400
####################################################################
# AUTHOR INFORMATION
####################################################################
# 2020 - 2021: Ajay B Harish (ajaybh@berkeley.edu)

####################################################################
# Import all necessary modules
####################################################################
# Standard python modules
import os

import meshio
import numpy as np

# Other custom modules
from hydroUtils import hydroUtils


####################################################################
# OpenFOAM7 solver class
####################################################################
class of7Building:
    """This class includes the methods related to
    creating the building for openfoam7.

    Methods
    -------
            buildcheck: Checks if all files required for creating the building exists
            createbuilds: Creates the STL files

    """  # noqa: D205, D404

    #############################################################
    def buildcheck(self, data, path):  # noqa: C901, PLR0911
        """Checks if all files required for creating the building exists

        Arguments:
        ---------
                data: all the JSON data
                path: Path to where the dakota.json exists

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()

        # Check if a translate script exists.
        # If so delete it
        if os.path.exists('translate.sh'):  # noqa: PTH110
            os.remove('translate.sh')  # noqa: PTH107

        # Check for STL file
        # Get the type of building definition
        buildeftype = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'BuildData'])
        )
        if buildeftype == 'Manual':
            # Find number of buildings
            numbuild = ', '.join(
                hydroutil.extract_element_from_json(data, ['Events', 'NumBuild'])
            )
            if int(numbuild) > 0:
                # Number of buildings with response
                numbuildres = 0
                # Get data for each building
                for ii in range(int(numbuild)):
                    builddata = ', '.join(
                        hydroutil.extract_element_from_json(
                            data, ['Events', 'BuildingTable' + str(ii)]
                        )
                    )
                    builddata = builddata.replace(',', ' ')
                    nums = [float(n) for n in builddata.split()]
                    buildtype = nums[0]
                    if int(buildtype) == -1 or int(buildtype) == 2:  # noqa: PLR2004
                        stlfile = hydroutil.extract_element_from_json(
                            data, ['Events', 'BuildingSTLFile']
                        )
                        if stlfile == [None]:
                            return -1  # noqa: DOC201, RUF100
                        else:  # noqa: RET505
                            stlfile = ', '.join(
                                hydroutil.extract_element_from_json(
                                    data, ['Events', 'BuildingSTLFile']
                                )
                            )
                            if not os.path.exists(os.path.join(path, stlfile)):  # noqa: PTH110, PTH118
                                return -1

                    if int(buildtype) == -2 or int(buildtype) == -1:  # noqa: PLR2004
                        numbuildres += 1
                        # Check GI
                        depth = hydroutil.extract_element_from_json(
                            data, ['GeneralInformation', 'depth']
                        )
                        if str(depth[0]) == [None]:
                            return -1

                        width = hydroutil.extract_element_from_json(
                            data, ['GeneralInformation', 'width']
                        )
                        if str(width[0]) == [None]:
                            return -1

                        height = hydroutil.extract_element_from_json(
                            data, ['GeneralInformation', 'height']
                        )
                        if str(height[0]) == [None]:
                            return -1

                        geninfo = hydroutil.extract_element_from_json(
                            data, ['GeneralInformation']
                        )
                        geninfo = str(geninfo[0])
                        xbuild = geninfo.partition("'location': {'latitude': ")[
                            1
                        ].partition(", 'longitude':")[0]
                        ybuild = geninfo.partition("'longitude': ")[2].partition(
                            '},'
                        )[0]
                        # if not depth:
                        # 	return -1
                        # # else:
                        # # 	depth = float(depth)
                        # if not width:
                        # 	return -1
                        # # else:
                        # # 	width = float(width)
                        # if not height:
                        # 	return -1
                        # # else:
                        # # 	height = float(height)
                        if not xbuild:
                            return -1
                        # else:
                        # 	xbuild = float(float)
                        if not ybuild:
                            return -1
                        # else:
                        # 	ybuild = float(ybuild)

                    if numbuildres > 1:
                        return -1

        elif buildeftype == 'Parameters':
            buildshape = ', '.join(
                hydroutil.extract_element_from_json(data, ['Events', 'BuildShape'])
            )
            if int(buildshape) == 0:
                return -1
            elif int(buildshape) == 1:  # noqa: RET505
                stlfile = hydroutil.extract_element_from_json(
                    data, ['Events', 'BuildingSTLFile']
                )
                if stlfile == [None]:
                    return -1
                else:  # noqa: RET505
                    stlfile = ', '.join(
                        hydroutil.extract_element_from_json(
                            data, ['Events', 'BuildingSTLFile']
                        )
                    )
                    if not os.path.exists(os.path.join(path, stlfile)):  # noqa: PTH110, PTH118
                        return -1

            # Check if building distribution selected
            buildDist = ', '.join(  # noqa: N806
                hydroutil.extract_element_from_json(data, ['Events', 'BuildDist'])
            )
            if int(buildDist) == 0:
                return -1

        return 0

    #############################################################
    def createbuilds(self, data, path):
        """Creates the STL files for the buildings and move to correct location

        Arguments:
        ---------
                data: all the JSON data
                path: Path to where the dakota.json exists

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()

        # Get the type of building definition
        buildeftype = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'BuildData'])
        )
        if buildeftype == 'Manual':
            self.buildmanual(data, path)

        elif buildeftype == 'Parameters':
            self.buildpara(data, path)

        return 0  # noqa: DOC201, RUF100

    #############################################################
    def buildmanual(self, data, path):
        """Creates the STL files for the buildings using manual data from table

        Arguments:
        ---------
                data: all the JSON data
                path: Path to where the dakota.json exists

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()

        # Number of types of buildings
        numresbuild = 0
        numotherbuild = 0

        # Get the coordinate and dimension data

        # Find number of buildings
        numbuild = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'NumBuild'])
        )
        if int(numbuild) > 0:
            # Get data for each building
            for ii in range(int(numbuild)):
                builddata = ', '.join(
                    hydroutil.extract_element_from_json(
                        data, ['Events', 'BuildingTable' + str(ii)]
                    )
                )
                builddata = builddata.replace(',', ' ')
                nums = [float(n) for n in builddata.split()]
                buildtype = nums[0]

                if int(buildtype) == -2:  # noqa: PLR2004
                    # Create a temporary file using GI information (Response)
                    self.buildcubeGI(data, path)
                    # Increment response buildign number
                    numresbuild += 1
                elif int(buildtype) == -1:
                    # Move the STL file to OF folder and change name to Building (Response)
                    self.readResSTL(data, path, nums[3])
                    # Increment response buildign number
                    numresbuild += 1
                elif int(buildtype) == 1:
                    print('no response + cuboid')  # noqa: T201
                    # Create a temporary file
                    # Call flume to build an STL
                    # Combine all STL to building + number
                    # Increment response buildign number
                    numotherbuild += 1
                elif int(buildtype) == 2:  # noqa: PLR2004
                    print('no response + STL')  # noqa: T201
                    # Check if STL file exists
                    # Increment response buildign number
                    numotherbuild += 1

        # Create other buildings STL if more than one exists (Join buildings)

        # Create the building flag
        self.buildflagadd(numresbuild, numotherbuild)

    #############################################################
    def buildpara(self, data, path):  # noqa: ARG002
        """Creates the STL files for the buildings using parametrized data

        Arguments:
        ---------
                data: all the JSON data
                path: Path to where the dakota.json exists

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()  # noqa: F841

    #############################################################
    def buildcubeGI(self, data, path):  # noqa: ARG002, N802
        """Creates the STL files for the buildings using parametrized data

        Arguments:
        ---------
                data: all the JSON data
                path: Path to where the dakota.json exists

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()  # noqa: F841

        # Create the building STL file
        base_filename = 'Building'
        filename = base_filename + '.stl'
        # Define coordinates
        npa = np.array(
            [
                [-1, -1, -1],
                [+1, -1, -1],
                [+1, +1, -1],
                [-1, +1, -1],
                [-1, -1, +1],
                [+1, -1, +1],
                [+1, +1, +1],
                [-1, +1, +1],
            ]
        )
        npt = np.array(
            [
                [0, 3, 1],
                [1, 3, 2],
                [0, 4, 7],
                [0, 7, 3],
                [4, 5, 6],
                [4, 6, 7],
                [5, 1, 2],
                [5, 2, 6],
                [2, 3, 6],
                [3, 7, 6],
                [0, 1, 5],
                [0, 5, 4],
            ]
        )
        # Scaling
        npa = 0.5 * npa
        npa[:, 2] = 0.5 + npa[:, 2]
        # Temporary
        npa[:, 0] = npa[:, 0] * 3
        npa[:, 2] = npa[:, 2] * 1.2474
        npa[:, 0] = npa[:, 0] + 47
        npa[:, 2] = npa[:, 2] + 1.7526

        # Create the STL file
        cells = [('triangle', npt)]
        meshio.write_points_cells(filename, npa, cells)
        # Modify first and last line
        with open(filename) as f:  # noqa: PTH123
            lines = f.readlines()
            lines[0] = 'solid ' + base_filename + '\n'
            lines[len(lines) - 1] = 'endsolid ' + base_filename + '\n'
        # Write the updated file
        with open(filename, 'w') as f:  # noqa: PTH123
            f.writelines(lines)

        # Create the translation script
        if os.path.exists('translate.sh'):  # noqa: PTH110
            with open('translate.sh', 'a') as f:  # noqa: PTH123
                buildpath = os.path.join(  # noqa: PTH118
                    'constant', 'triSurface', 'Building.stl'
                )
                lines = 'cp Building.stl ' + buildpath + '\n'
                f.writelines(lines)
        else:
            with open('translate.sh', 'w') as f:  # noqa: PTH123
                buildpath = os.path.join(  # noqa: PTH118
                    'constant', 'triSurface', 'Building.stl'
                )
                lines = 'cp Building.stl ' + buildpath + '\n'
                f.writelines(lines)

    #############################################################
    def readResSTL(self, data, path, ztrans):  # noqa: N802
        """Creates the STL files for the buildings using parametrized data

        Arguments:
        ---------
                data: all the JSON data
                path: Path to where the dakota.json exists
                ztrans: Translation distance in z-direction

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()

        # Filename
        stlfile = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'BuildingSTLFile'])
        )

        # Read the stlfile
        stlfilepath = os.path.join(path, stlfile)  # noqa: PTH118
        print(stlfilepath)  # noqa: T201
        mesh = meshio.read(stlfilepath, file_format='stl')

        mesh.points[:, 0] = mesh.points[:, 0] / (max(abs(mesh.points[:, 0])))
        mesh.points[:, 1] = mesh.points[:, 1] / (max(abs(mesh.points[:, 1])))
        mesh.points[:, 2] = mesh.points[:, 2] / (max(abs(mesh.points[:, 2])))

        # Get GI
        geninfo = hydroutil.extract_element_from_json(data, ['GeneralInformation'])
        geninfo = str(geninfo[0])
        # depth = float(geninfo.partition("'depth': ")[2].partition(", 'height':")[0])
        # width = float(geninfo.partition("'width': ")[2].partition("}")[0])
        # height = float(geninfo.partition("'height': ")[2].partition(", 'location':")[0])
        xbuild = float(
            geninfo.partition("'location': {'latitude': ")[2].partition(
                ", 'longitude':"
            )[0]
        )
        ybuild = float(geninfo.partition("'longitude': ")[2].partition('},')[0])
        depth = hydroutil.extract_element_from_json(
            data, ['GeneralInformation', 'depth']
        )
        depth = float(depth[0])
        width = hydroutil.extract_element_from_json(
            data, ['GeneralInformation', 'width']
        )
        width = float(width[0])
        height = hydroutil.extract_element_from_json(
            data, ['GeneralInformation', 'height']
        )
        height = float(height[0])

        # Scale the STL model
        mesh.points[:, 0] = mesh.points[:, 0] * depth
        mesh.points[:, 1] = mesh.points[:, 1] * width
        mesh.points[:, 2] = mesh.points[:, 2] * height

        # Write meshfile
        meshio.write_points_cells('Building.stl', mesh.points, mesh.cells)

        # Modify first and last line
        with open('Building.stl') as f:  # noqa: PTH123
            lines = f.readlines()
            lines[0] = 'solid ' + 'Building' + '\n'
            lines[len(lines) - 1] = 'endsolid ' + 'Building' + '\n'

        # Write the updated file
        with open('Building.stl', 'w') as f:  # noqa: PTH123
            f.writelines(lines)

        # Move the file to constant/triSurface folder
        newfilepath = os.path.join(  # noqa: PTH118
            path, 'constant', 'triSurface', 'Building.stl'
        )
        os.replace('Building.stl', newfilepath)  # noqa: PTH105

        # Create the translation script
        if os.path.exists('translate.sh'):  # noqa: PTH110
            with open('translate.sh', 'a') as f:  # noqa: PTH123
                buildpath = os.path.join(  # noqa: PTH118
                    'constant', 'triSurface', 'Building.stl'
                )
                lines = 'export FILE="' + buildpath + '"\n'
                lines = (
                    lines
                    + 'surfaceTransformPoints -translate "('
                    + str(xbuild)
                    + ' '
                    + str(ybuild)
                    + ' '
                    + str(ztrans)
                    + ')" $FILE $FILE\n'
                )
                f.writelines(lines)
        else:
            with open('translate.sh', 'w') as f:  # noqa: PTH123
                buildpath = os.path.join(  # noqa: PTH118
                    'constant', 'triSurface', 'Building.stl'
                )
                lines = 'export FILE="' + buildpath + '"\n'
                lines = (
                    lines
                    + 'surfaceTransformPoints -translate "('
                    + str(xbuild)
                    + ' '
                    + str(ybuild)
                    + ' '
                    + str(ztrans)
                    + ')" $FILE $FILE\n'
                )
                f.writelines(lines)

    #############################################################
    def buildflagadd(self, numresbuild, numotherbuild):
        """Add building flag to temp_geometry.txt

        Arguments:
        ---------
                numresbuild: Number of building with response
                numotherbuild: NUmber of other buildings

        """  # noqa: D400
        # Get building flag
        if numresbuild == 0 and numotherbuild == 0:
            flag = 0
        elif numresbuild > 0 and numotherbuild == 0:
            flag = 1
        elif numresbuild > 0 and numotherbuild > 0:
            flag = 2
        elif numresbuild == 0 and numotherbuild > 0:
            flag = 3

        # Add building flag to temp file
        with open('temp_geometry.txt', 'a') as f:  # noqa: PTH123
            f.writelines(str(flag) + '\n')
