####################################################################
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

"""
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

    """

    #############################################################
    def geomcheck(self, data, path):
        """Checks if all files required for creating the geometry exists

        Arguments:
        ---------
                data: all the JSON data
                path: Path to where the dakota.json exists

        """
        # Create a utilities object
        hydroutil = hydroUtils()

        # Get the simulation type
        simtype = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'SimulationType'])
        )

        # Simtype: Multiscale with SW solutions
        if int(simtype) == 1 or int(simtype) == 2:
            # Get the number of bathymetry files
            numbathy = hydroutil.extract_element_from_json(
                data, ['Events', 'NumBathymetryFiles']
            )
            if numbathy == [None]:
                return -1
            else:
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
                    else:
                        bathyfilename = ', '.join(
                            hydroutil.extract_element_from_json(
                                data, ['Events', 'BathymetryFile' + str(ii)]
                            )
                        )
                        bathyfilepath = os.join.path(path, bathyfilename)
                        if not os.path.isfile(bathyfilepath):
                            return -1

            if int(simtype) == 1:
                # Get the number of solution files
                numsoln = hydroutil.extract_element_from_json(
                    data, ['Events', 'NumSolutionFiles']
                )
                if numsoln == [None]:
                    return -1
                else:
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
                    else:
                        solnfilename = ', '.join(
                            hydroutil.extract_element_from_json(
                                data, ['Events', 'SWSolutionFile' + str(ii)]
                            )
                        )
                        solnfilepath = os.join.path(path, solnfilename)
                        if not os.path.isfile(solnfilepath):
                            return -1

                # Check the SW-CFD interface file
                swcfdfile = hydroutil.extract_element_from_json(
                    data, ['Events', 'SWCFDInteFile']
                )
                if swcfdfile == [None]:
                    return -1
                else:
                    swcfdfile = ', '.join(
                        hydroutil.extract_element_from_json(
                            data, ['Events', 'SWCFDInteFile']
                        )
                    )
                    swcfdfilepath = os.join.path(path, swcfdfile)
                    if not os.path.isfile(swcfdfilepath):
                        return -1

        # STL file
        elif int(simtype) == 3:
            # Entry.stl
            entrypath = os.join.path(path, 'Entry.stl')
            if not os.path.isfile(entrypath):
                return -1

            # Exit.stl
            exitpath = os.join.path(path, 'Exit.stl')
            if not os.path.isfile(exitpath):
                return -1

            # Top.stl
            toppath = os.join.path(path, 'Top.stl')
            if not os.path.isfile(toppath):
                return -1

            # Bottom.stl
            bottompath = os.join.path(path, 'Bottom.stl')
            if not os.path.isfile(bottompath):
                return -1

            # Left.stl
            leftpath = os.join.path(path, 'Left.stl')
            if not os.path.isfile(leftpath):
                return -1

            # Right.stl
            rightpath = os.join.path(path, 'Right.stl')
            if not os.path.isfile(rightpath):
                return -1

        # Wave flume
        elif int(simtype) == 4:
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
                else:
                    numsegs = ', '.join(
                        hydroutil.extract_element_from_json(
                            data, ['Events', 'NumFlumeSegments']
                        )
                    )
                    if int(numsegs) < 4:
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
    def createOFSTL(self, data, path):
        """Creates the STL files

        Arguments:
        ---------
                data: all the JSON data
                path: Path to where the dakota.json exists

        """
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
                return -1

        # Bathymetry only
        elif int(simtype) == 2:
            print('Bathy')
            finalgeom = GeoClawBathy()
            # Create geometry (i.e. STL files) and extreme file
            ecode = finalgeom.creategeom(data, path)
            if ecode < 0:
                return -1

        elif int(simtype) == 3:
            return 0

        elif int(simtype) == 4:
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

        """
        # Create a utilities object
        hydroutil = hydroUtils()

        # Get the mesher
        mesher = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'MeshType'])
        )

        # Combine STL files for Hydro mesh or using mesh dict
        if int(mesher[0]) == 0 or int(mesher[0]) == 2:
            # Get building flag from temp-geometry file
            geofile = 'temp_geometry.txt'
            data_geoext = np.genfromtxt(geofile, dtype=(float))
            flag = int(data_geoext[6])

            # If translate file exists, use it
            if os.path.exists('translate.sh'):
                caseruntext = 'echo Translating building STL files...\n'
                caseruntext = caseruntext + 'chmod +x translate.sh\n'
                caseruntext = caseruntext + './translate.sh\n\n'
                caseruntext = caseruntext + 'echo Combining STL files for usage...\n'
            else:
                caseruntext = 'echo Combining STL files for usage...\n'

            # Join all paths
            entryf = os.path.join('constant', 'triSurface', 'Entry.stl')
            exitf = os.path.join('constant', 'triSurface', 'Exit.stl')
            topf = os.path.join('constant', 'triSurface', 'Top.stl')
            bottomf = os.path.join('constant', 'triSurface', 'Bottom.stl')
            leftf = os.path.join('constant', 'triSurface', 'Left.stl')
            rightf = os.path.join('constant', 'triSurface', 'Right.stl')
            buildingf = os.path.join('constant', 'triSurface', 'Building.stl')
            otherbuildingf = os.path.join(
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
            full = os.path.join('constant', 'triSurface', 'Full.stl')

            # For different building cases
            if flag == 0:
                caseruntext = caseruntext + all01 + ' > ' + full + '\n\n'
            elif flag == 1:
                caseruntext = (
                    caseruntext + all01 + ' ' + buildingf + ' > ' + full + '\n\n'
                )
            elif flag == 2:
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
            elif flag == 3:
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
            scriptfile = open('caserun.sh', 'a')
            scriptfile.write(caseruntext)
            scriptfile.close()
