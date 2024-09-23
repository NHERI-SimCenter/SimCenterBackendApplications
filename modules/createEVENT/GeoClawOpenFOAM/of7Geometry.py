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

import numpy as np
from GeoClaw import GeoClaw
from GeoClawBathy import GeoClawBathy

# Other custom modules
from hydroUtils import hydroUtils
from osuFlume import osuFlume
from userFlume import userFlume


####################################################################
# OpenFOAM7 solver class
####################################################################
class of7Geometry:
    """This class includes the methods related to
    creating the geometry for openfoam7.

    Methods
    -------
            geomcheck: Checks if all files required for creating the geometry exists
            createSTL: Creates the STL files

    """  # noqa: D205, D404

    #############################################################
    def geomcheck(self, data, path):  # noqa: C901, PLR0911
        """Checks if all files required for creating the geometry exists

        Arguments:
        ---------
                data: all the JSON data
                path: Path to where the dakota.json exists

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()

        # Get the simulation type
        simtype = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'SimulationType'])
        )

        # Simtype: Multiscale with SW solutions
        if int(simtype) == 1 or int(simtype) == 2:  # noqa: PLR2004
            # Get the number of bathymetry files
            numbathy = hydroutil.extract_element_from_json(
                data, ['Events', 'NumBathymetryFiles']
            )
            if numbathy == [None]:
                return -1  # noqa: DOC201, RUF100
            else:  # noqa: RET505
                numbathy = ', '.join(
                    hydroutil.extract_element_from_json(
                        data, ['Events', 'NumBathymetryFiles']
                    )
                )

                # Loop to get the name of each bathymetry file
                # Check if it exists. If not, return -1
                for ii in range(int(numbathy)):
                    # Get the file name
                    bathyfilename = hydroutil.extract_element_from_json(
                        data, ['Events', 'BathymetryFile' + str(ii)]
                    )
                    if bathyfilename == [None]:
                        return -1
                    else:  # noqa: RET505
                        bathyfilename = ', '.join(
                            hydroutil.extract_element_from_json(
                                data, ['Events', 'BathymetryFile' + str(ii)]
                            )
                        )
                        bathyfilepath = os.join.path(path, bathyfilename)
                        if not os.path.isfile(bathyfilepath):  # noqa: PTH113
                            return -1

            if int(simtype) == 1:
                # Get the number of solution files
                numsoln = hydroutil.extract_element_from_json(
                    data, ['Events', 'NumSolutionFiles']
                )
                if numsoln == [None]:
                    return -1
                else:  # noqa: RET505
                    numsoln = ', '.join(
                        hydroutil.extract_element_from_json(
                            data, ['Events', 'NumSolutionFiles']
                        )
                    )

                # Loop to get the name of each solution file
                # Check if it exists. If not, return -1
                for ii in range(int(numsoln)):
                    # Get the file name
                    solnfilename = hydroutil.extract_element_from_json(
                        data, ['Events', 'SWSolutionFile' + str(ii)]
                    )
                    if solnfilename == [None]:
                        return -1
                    else:  # noqa: RET505
                        solnfilename = ', '.join(
                            hydroutil.extract_element_from_json(
                                data, ['Events', 'SWSolutionFile' + str(ii)]
                            )
                        )
                        solnfilepath = os.join.path(path, solnfilename)
                        if not os.path.isfile(solnfilepath):  # noqa: PTH113
                            return -1

                # Check the SW-CFD interface file
                swcfdfile = hydroutil.extract_element_from_json(
                    data, ['Events', 'SWCFDInteFile']
                )
                if swcfdfile == [None]:
                    return -1
                else:  # noqa: RET505
                    swcfdfile = ', '.join(
                        hydroutil.extract_element_from_json(
                            data, ['Events', 'SWCFDInteFile']
                        )
                    )
                    swcfdfilepath = os.join.path(path, swcfdfile)
                    if not os.path.isfile(swcfdfilepath):  # noqa: PTH113
                        return -1

        # STL file
        elif int(simtype) == 3:  # noqa: PLR2004
            # Entry.stl
            entrypath = os.join.path(path, 'Entry.stl')
            if not os.path.isfile(entrypath):  # noqa: PTH113
                return -1

            # Exit.stl
            exitpath = os.join.path(path, 'Exit.stl')
            if not os.path.isfile(exitpath):  # noqa: PTH113
                return -1

            # Top.stl
            toppath = os.join.path(path, 'Top.stl')
            if not os.path.isfile(toppath):  # noqa: PTH113
                return -1

            # Bottom.stl
            bottompath = os.join.path(path, 'Bottom.stl')
            if not os.path.isfile(bottompath):  # noqa: PTH113
                return -1

            # Left.stl
            leftpath = os.join.path(path, 'Left.stl')
            if not os.path.isfile(leftpath):  # noqa: PTH113
                return -1

            # Right.stl
            rightpath = os.join.path(path, 'Right.stl')
            if not os.path.isfile(rightpath):  # noqa: PTH113
                return -1

        # Wave flume
        elif int(simtype) == 4:  # noqa: PLR2004
            # Get the flume type
            flumetype = ', '.join(
                hydroutil.extract_element_from_json(
                    data, ['Events', 'FlumeInfoType']
                )
            )

            # Using user coordinates
            if int(flumetype) == 0:
                # Get the number of segments
                numsegs = hydroutil.extract_element_from_json(
                    data, ['Events', 'NumFlumeSegments']
                )
                if numsegs == [None]:
                    return -1
                else:  # noqa: RET505
                    numsegs = ', '.join(
                        hydroutil.extract_element_from_json(
                            data, ['Events', 'NumFlumeSegments']
                        )
                    )
                    if int(numsegs) < 4:  # noqa: PLR2004
                        return -1
                    flumesegs = hydroutil.extract_element_from_json(
                        data, ['Events', 'FlumeSegments']
                    )
                    if flumesegs == [None]:
                        return -1
            # Standard flume
            elif int(flumetype) == 1:
                return 0

        return 0

    #############################################################
    def createOFSTL(self, data, path):  # noqa: C901, N802
        """Creates the STL files

        Arguments:
        ---------
                data: all the JSON data
                path: Path to where the dakota.json exists

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()

        # Get the simulation type
        simtype = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'SimulationType'])
        )

        # Bathymetry + SW solutions
        if int(simtype) == 1:
            finalgeom = GeoClaw()
            # Create geometry (i.e. STL files) and extreme file
            ecode = finalgeom.creategeom(data, path)
            if ecode < 0:
                return -1  # noqa: DOC201, RUF100

        # Bathymetry only
        elif int(simtype) == 2:  # noqa: PLR2004
            print('Bathy')  # noqa: T201
            finalgeom = GeoClawBathy()
            # Create geometry (i.e. STL files) and extreme file
            ecode = finalgeom.creategeom(data, path)
            if ecode < 0:
                return -1

        elif int(simtype) == 3:  # noqa: PLR2004
            return 0

        elif int(simtype) == 4:  # noqa: PLR2004
            # Get the flume type
            flumetype = ', '.join(
                hydroutil.extract_element_from_json(
                    data, ['Events', 'FlumeInfoType']
                )
            )

            # Using user coordinates
            if int(flumetype) == 0:
                finalgeom = userFlume()
                # Create geometry (i.e. STL files) and extreme file
                ecode = finalgeom.creategeom(data, path)
                if ecode < 0:
                    return -1

            # Standard flume
            elif int(flumetype) == 1:
                finalgeom = osuFlume()
                # Create geometry (i.e. STL files) and extreme file
                ecode = finalgeom.creategeom(data, path)
                if ecode < 0:
                    return -1

        return 0

    #############################################################
    def scripts(self, data):
        """Add to caserun.sh

        Arguments:
        ---------
                NONE

        """  # noqa: D400
        # Create a utilities object
        hydroutil = hydroUtils()

        # Get the mesher
        mesher = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'MeshType'])
        )

        # Combine STL files for Hydro mesh or using mesh dict
        if int(mesher[0]) == 0 or int(mesher[0]) == 2:  # noqa: PLR2004
            # Get building flag from temp-geometry file
            geofile = 'temp_geometry.txt'
            data_geoext = np.genfromtxt(geofile, dtype=(float))
            flag = int(data_geoext[6])

            # If translate file exists, use it
            if os.path.exists('translate.sh'):  # noqa: PTH110
                caseruntext = 'echo Translating building STL files...\n'
                caseruntext = caseruntext + 'chmod +x translate.sh\n'
                caseruntext = caseruntext + './translate.sh\n\n'
                caseruntext = caseruntext + 'echo Combining STL files for usage...\n'
            else:
                caseruntext = 'echo Combining STL files for usage...\n'

            # Join all paths
            entryf = os.path.join('constant', 'triSurface', 'Entry.stl')  # noqa: PTH118
            exitf = os.path.join('constant', 'triSurface', 'Exit.stl')  # noqa: PTH118
            topf = os.path.join('constant', 'triSurface', 'Top.stl')  # noqa: PTH118
            bottomf = os.path.join('constant', 'triSurface', 'Bottom.stl')  # noqa: PTH118
            leftf = os.path.join('constant', 'triSurface', 'Left.stl')  # noqa: PTH118
            rightf = os.path.join('constant', 'triSurface', 'Right.stl')  # noqa: PTH118
            buildingf = os.path.join('constant', 'triSurface', 'Building.stl')  # noqa: PTH118
            otherbuildingf = os.path.join(  # noqa: PTH118
                'constant', 'triSurface', 'OtherBuilding.stl'
            )
            all01 = (
                'cat '
                + entryf
                + ' '
                + exitf
                + ' '
                + topf
                + ' '
                + bottomf
                + ' '
                + leftf
                + ' '
                + rightf
            )
            full = os.path.join('constant', 'triSurface', 'Full.stl')  # noqa: PTH118

            # For different building cases
            if flag == 0:
                caseruntext = caseruntext + all01 + ' > ' + full + '\n\n'
            elif flag == 1:
                caseruntext = (
                    caseruntext + all01 + ' ' + buildingf + ' > ' + full + '\n\n'
                )
            elif flag == 2:  # noqa: PLR2004
                caseruntext = (
                    caseruntext
                    + all01
                    + ' '
                    + buildingf
                    + ' '
                    + otherbuildingf
                    + ' > '
                    + full
                    + '\n\n'
                )
            elif flag == 3:  # noqa: PLR2004
                caseruntext = (
                    caseruntext
                    + all01
                    + ' '
                    + otherbuildingf
                    + ' > '
                    + full
                    + '\n\n'
                )
            # Write to caserun file
            scriptfile = open('caserun.sh', 'a')  # noqa: SIM115, PTH123
            scriptfile.write(caseruntext)
            scriptfile.close()
