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
import math
import os

import numpy as np

# Other custom modules
from hydroUtils import hydroUtils


####################################################################
# OpenFOAM7 solver class
####################################################################
class of7Meshing:
    """This class includes the methods related to
    meshing for openfoam7.

    Methods
    -------
            meshcheck: Check all the meshing

    """

    #############################################################
    def meshcheck(self, data, fipath):
        """Checks for material properties for openfoam7

        Arguments:
        ---------
                data: all the JSON data

        """
        # Create a utilities object
        hydroutil = hydroUtils()

        # Get mesher type
        mesher = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'MeshType'])
        )

        # If hydro mesher - nothing to check
        if int(mesher[0]) == 0:
            return 0

        # Other mesh softwares
        elif int(mesher[0]) == 1:
            meshfile = hydroutil.extract_element_from_json(
                data, ['Events', 'MeshFile']
            )
            if meshfile == [None]:
                return -1
            else:
                meshfile = ', '.join(
                    hydroutil.extract_element_from_json(data, ['Events', 'MeshFile'])
                )
                meshfilepath = os.path.join(fipath, meshfile)
                if not os.path.isfile(meshfilepath):
                    return -1

        # Mesh dictionaries
        elif int(mesher[0]) == 2:
            # Get path of bm and shm
            bmfile = os.path.join(fipath, 'blockMeshDict')
            shmfile = os.path.join(fipath, 'snappyHexMeshDict')

            # Check if both blockmeshdict or SHM do not exist
            if (not os.path.isfile(bmfile)) and (not os.path.isfile(shmfile)):
                return -1

        # Return 0 if all is right
        return 0

    #############################################################
    def meshheader(self, fileobjec):
        """Creates the text for the header

        Variable
        -----------
                header: Header for the solver-files
        """
        header = (
            """/*--------------------------*- NHERI SimCenter -*----------------------------*\\ 
|	   | H |
|	   | Y | HydroUQ: Water-based Natural Hazards Modeling Application
|======| D | Website: simcenter.designsafe-ci.org/research-tools/hydro-uq
|	   | R | Version: 1.00
|	   | O |
\\*---------------------------------------------------------------------------*/ 
FoamFile
{\n\tversion\t2.0;\n\tformat\tascii;\n\tclass\tdictionary;\n\tlocation\t"system";\n\tobject\t"""
            + fileobjec
            + """;\n}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"""
        )

        # Return the header for meshing file
        return header

    #############################################################
    def bmeshtext(self, data):
        """Creates the necessary files for blockMeshDict for openfoam7

        Arguments:
        ---------
                data: all the JSON data

        """
        # Read the geometry data file
        data_geoext = np.genfromtxt('temp_geometry.txt', dtype=(float))

        # Create a utilities object
        hydroutil = hydroUtils()
        meshsize = ''.join(
            hydroutil.extract_element_from_json(data, ['Events', 'MeshSize'])
        )

        # Get the mesh sizes
        nx = 100 * int(meshsize)
        if abs(data_geoext[1] - data_geoext[0]) > 0.000001:
            ny = math.ceil(
                5
                * nx
                * (
                    (data_geoext[3] - data_geoext[2])
                    / (data_geoext[1] - data_geoext[0])
                )
            )
            nz = math.ceil(
                5
                * nx
                * (
                    (data_geoext[5] - data_geoext[4])
                    / (data_geoext[1] - data_geoext[0])
                )
            )

        # Get the header text for the blockMeshDict
        bmeshtext = self.meshheader('blockMeshDict')

        # Convert units
        bmeshtext = bmeshtext + 'convertToMeters\t1;\n\n'

        # Add vertices
        bmeshtext = bmeshtext + 'vertices\n(\n\t'
        bmeshtext = (
            bmeshtext
            + '('
            + str(data_geoext[0])
            + '\t'
            + str(data_geoext[2])
            + '\t'
            + str(data_geoext[4])
            + ')\n\t'
        )
        bmeshtext = (
            bmeshtext
            + '('
            + str(data_geoext[1])
            + '\t'
            + str(data_geoext[2])
            + '\t'
            + str(data_geoext[4])
            + ')\n\t'
        )
        bmeshtext = (
            bmeshtext
            + '('
            + str(data_geoext[1])
            + '\t'
            + str(data_geoext[3])
            + '\t'
            + str(data_geoext[4])
            + ')\n\t'
        )
        bmeshtext = (
            bmeshtext
            + '('
            + str(data_geoext[0])
            + '\t'
            + str(data_geoext[3])
            + '\t'
            + str(data_geoext[4])
            + ')\n\t'
        )
        bmeshtext = (
            bmeshtext
            + '('
            + str(data_geoext[0])
            + '\t'
            + str(data_geoext[2])
            + '\t'
            + str(data_geoext[5])
            + ')\n\t'
        )
        bmeshtext = (
            bmeshtext
            + '('
            + str(data_geoext[1])
            + '\t'
            + str(data_geoext[2])
            + '\t'
            + str(data_geoext[5])
            + ')\n\t'
        )
        bmeshtext = (
            bmeshtext
            + '('
            + str(data_geoext[1])
            + '\t'
            + str(data_geoext[3])
            + '\t'
            + str(data_geoext[5])
            + ')\n\t'
        )
        bmeshtext = (
            bmeshtext
            + '('
            + str(data_geoext[0])
            + '\t'
            + str(data_geoext[3])
            + '\t'
            + str(data_geoext[5])
            + ')\n);\n\n'
        )

        # Add blocks
        bmeshtext = bmeshtext + 'blocks\n(\n\t'
        bmeshtext = (
            bmeshtext
            + 'hex (0 1 2 3 4 5 6 7) ('
            + str(nx)
            + '\t'
            + str(ny)
            + '\t'
            + str(nz)
            + ') simpleGrading (1 1 1)\n);\n\n'
        )

        # Add edges
        bmeshtext = bmeshtext + 'edges\n(\n);\n\n'

        # Add patches
        bmeshtext = bmeshtext + 'patches\n(\n\t'
        bmeshtext = bmeshtext + 'patch maxY\n\t(\n\t\t(3 7 6 2)\n\t)\n\t'
        bmeshtext = bmeshtext + 'patch minX\n\t(\n\t\t(0 4 7 3)\n\t)\n\t'
        bmeshtext = bmeshtext + 'patch maxX\n\t(\n\t\t(2 6 5 1)\n\t)\n\t'
        bmeshtext = bmeshtext + 'patch minY\n\t(\n\t\t(1 5 4 0)\n\t)\n\t'
        bmeshtext = bmeshtext + 'patch minZ\n\t(\n\t\t(0 3 2 1)\n\t)\n\t'
        bmeshtext = bmeshtext + 'patch maxZ\n\t(\n\t\t(4 5 6 7)\n\t)\n'
        bmeshtext = bmeshtext + ');\n\n'

        # Add merge patch pairs
        bmeshtext = bmeshtext + 'mergePatchPairs\n(\n);\n'

        return bmeshtext

    #############################################################
    def sfetext(self):
        """Creates the necessary files for new controldict for post-processing for openfoam7

        Arguments:
        ---------
                data: all the JSON data

        """
        # Read the geometry data file
        data_geoext = np.genfromtxt('temp_geometry.txt', dtype=(float))

        # Get the header text for the blockMeshDict
        sfetext = self.meshheader('surfaceFeatureExtractDict')

        # Rest of text
        stlinfo = '{\n\textractionMethod\textractFromSurface;\n'
        stlinfo = stlinfo + '\textractFromSurfaceCoeffs\n'
        stlinfo = stlinfo + '\t{includedAngle\t150;}\n'
        stlinfo = stlinfo + '\twriteObj\tyes;\n}'
        sfetext = sfetext + 'Entry.stl\n' + stlinfo + '\n\n'
        sfetext = sfetext + 'Exit.stl\n' + stlinfo + '\n\n'
        sfetext = sfetext + 'Top.stl\n' + stlinfo + '\n\n'
        sfetext = sfetext + 'Bottom.stl\n' + stlinfo + '\n\n'
        sfetext = sfetext + 'Left.stl\n' + stlinfo + '\n\n'
        sfetext = sfetext + 'Right.stl\n' + stlinfo + '\n\n'
        if int(data_geoext[6]) == 1:
            sfetext = sfetext + 'Building.stl\n' + stlinfo + '\n\n'
        elif int(data_geoext[6]) == 2:
            sfetext = sfetext + 'Building.stl\n' + stlinfo + '\n\n'
            sfetext = sfetext + 'OtherBuilding.stl\n' + stlinfo + '\n\n'
        elif int(data_geoext[6]) == 3:
            sfetext = sfetext + 'OtherBuilding.stl\n' + stlinfo + '\n\n'

        return sfetext

    #############################################################
    def shmtext(self, data):
        """Creates the necessary files for new controldict for post-processing for openfoam7

        Arguments:
        ---------
                None

        """
        # Read the geometry data file
        data_geoext = np.genfromtxt('temp_geometry.txt', dtype=(float))

        # Create a utilities object
        hydroutil = hydroUtils()
        meshsize = ''.join(
            hydroutil.extract_element_from_json(data, ['Events', 'MeshSize'])
        )

        # Get the header text for the blockMeshDict
        shmtext = self.meshheader('snappyHexMeshDict')

        # Rest of text
        shmtext = shmtext + 'castellatedMesh\ttrue;\n\n'
        shmtext = shmtext + 'snap\ttrue;\n\n'
        shmtext = shmtext + 'addLayers\tfalse;\n\n'

        # Geometry. Definition of all surfaces.
        shmtext = shmtext + 'geometry\n{\n\t'
        shmtext = shmtext + 'Entry.stl {type triSurfaceMesh; name Entry;}\n\t'
        shmtext = shmtext + 'Exit.stl {type triSurfaceMesh; name Exit;}\n\t'
        shmtext = shmtext + 'Top.stl {type triSurfaceMesh; name Top;}\n\t'
        shmtext = shmtext + 'Bottom.stl {type triSurfaceMesh; name Bottom;}\n\t'
        shmtext = shmtext + 'Left.stl {type triSurfaceMesh; name Left;}\n\t'
        shmtext = shmtext + 'Right.stl {type triSurfaceMesh; name Right;}\n'
        if int(data_geoext[6]) == 1:
            shmtext = (
                shmtext + '\tBuilding.stl {type triSurfaceMesh; name Building;}\n'
            )
        elif int(data_geoext[6]) == 2:
            shmtext = (
                shmtext + '\tBuilding.stl {type triSurfaceMesh; name Building;}\n'
            )
            shmtext = (
                shmtext
                + '\tOtherBuilding.stl {type triSurfaceMesh; name OtherBuilding;}\n'
            )
        elif int(data_geoext[6]) == 3:
            shmtext = (
                shmtext
                + '\tOtherBuilding.stl {type triSurfaceMesh; name OtherBuilding;}\n'
            )
        shmtext = shmtext + '\tFull.stl {type triSurfaceMesh; name Full;}\n'
        shmtext = shmtext + '};\n\n'

        # Castellated mesh generation
        maxLocalCells = int(meshsize) * 2000000
        maxGlobalCells = int(meshsize) * 10000000
        shmtext = shmtext + 'castellatedMeshControls\n{\n\t'
        shmtext = shmtext + 'maxLocalCells\t' + str(maxLocalCells) + ';\n\t'
        shmtext = shmtext + 'maxGlobalCells\t' + str(maxGlobalCells) + ';\n\t'
        shmtext = shmtext + 'minRefinementCells\t10;\n\t'
        shmtext = shmtext + 'maxLoadUnbalance\t0.1;\n\t'
        shmtext = shmtext + 'nCellsBetweenLevels\t1;\n\n'

        # Explicit feature edge refinement
        shmtext = shmtext + '\tfeatures\n\t(\n\t\t'
        shmtext = shmtext + '{file "Entry.eMesh"; level 3;}\n\t\t'
        shmtext = shmtext + '{file "Exit.eMesh"; level 3;}\n\t\t'
        shmtext = shmtext + '{file "Top.eMesh"; level 3;}\n\t\t'
        shmtext = shmtext + '{file "Bottom.eMesh"; level 3;}\n\t\t'
        shmtext = shmtext + '{file "Left.eMesh"; level 3;}\n\t\t'
        shmtext = shmtext + '{file "Right.eMesh"; level 3;}\n'
        if int(data_geoext[6]) == 1:
            shmtext = shmtext + '\t\t{file "Building.eMesh"; level 3;}\n'
        elif int(data_geoext[6]) == 2:
            shmtext = shmtext + '\t\t{file "Building.eMesh"; level 3;}\n'
            shmtext = shmtext + '\t\t{file "OtherBuilding.eMesh"; level 3;}\n'
        elif int(data_geoext[6]) == 3:
            shmtext = shmtext + '\t\t{file "OtherBuilding.eMesh"; level 3;}\n'
        shmtext = shmtext + '\t);\n\n'

        # Surface based refinement
        shmtext = shmtext + '\trefinementSurfaces\n\t{\n\t\t'
        shmtext = shmtext + 'Entry {level (0 0);}\n\t\t'
        shmtext = shmtext + 'Exit {level (0 0);}\n\t\t'
        shmtext = shmtext + 'Top {level (0 0);}\n\t\t'
        shmtext = shmtext + 'Bottom {level (2 2);}\n\t\t'
        shmtext = shmtext + 'Left {level (2 2);}\n\t\t'
        shmtext = shmtext + 'Right {level (2 2);}\n'
        if int(data_geoext[6]) == 1:
            shmtext = shmtext + '\t\tBuilding {level (2 2);}\n'
        elif int(data_geoext[6]) == 2:
            shmtext = shmtext + '\t\tBuilding {level (2 2);}\n'
            shmtext = shmtext + '\t\tOtherBuilding {level (2 2);}\n'
        elif int(data_geoext[6]) == 3:
            shmtext = shmtext + '\t\tOtherBuilding {level (2 2);}\n'
        shmtext = shmtext + '\t};\n\n'

        # Resolve sharp angles
        shmtext = shmtext + '\tresolveFeatureAngle 80;\n\n'

        # Regional refinement
        # This needs to be added and corrected
        shmtext = (
            shmtext + '\trefinementRegions\n\t{\n\t\t//Nothing here for now\n\t}\n\n'
        )

        # Get the point inside the body
        px = 0.5 * (data_geoext[1] + data_geoext[0])
        py = 0.5 * (data_geoext[3] + data_geoext[2])
        pz = 0.5 * (data_geoext[5] + data_geoext[4])

        # Mesh selection
        shmtext = (
            shmtext
            + '\tlocationInMesh ('
            + str(px)
            + '\t'
            + str(py)
            + '\t'
            + str(pz)
            + ');\n\n'
        )
        shmtext = shmtext + '\tallowFreeStandingZoneFaces\tfalse;\n'
        shmtext = shmtext + '}\n\n'

        # Snaping settings
        shmtext = shmtext + 'snapControls\n{\n\t'
        shmtext = shmtext + 'nSmoothPatch\t3;\n\t'
        shmtext = shmtext + 'tolerance\t4.0;\n\t'
        shmtext = shmtext + 'nSolveIter\t30;\n\t'
        shmtext = shmtext + 'nRelaxIter\t5;\n'
        shmtext = shmtext + '}\n\n'

        # Settings for layer addition
        # This is presently not being used
        shmtext = shmtext + 'addLayersControls\n{\n\t'
        shmtext = shmtext + 'relativeSizes\ttrue;\n\t'
        shmtext = shmtext + 'layers\n\t{\n\t'
        shmtext = shmtext + 'Bottom\n\t\t{nSurfaceLayers\t3;}\n\t'
        shmtext = shmtext + 'Left\n\t\t{nSurfaceLayers\t3;}\n\t'
        shmtext = shmtext + 'Right\n\t\t{nSurfaceLayers\t3;}\n\t}\n\n\t'
        shmtext = shmtext + 'expansionRatio\t1;\n\t'
        shmtext = shmtext + 'finalLayerThickness\t0.3;\n\t'
        shmtext = shmtext + 'minThickness\t0.1;\n\t'
        shmtext = shmtext + 'nGrow\t0;\n\t'

        # Advanced settings for layer addition
        shmtext = shmtext + 'featureAngle\t80;\n\t'
        shmtext = shmtext + 'nRelaxIter\t3;\n\t'
        shmtext = shmtext + 'nSmoothSurfaceNormals\t1;\n\t'
        shmtext = shmtext + 'nSmoothNormals\t3;\n\t'
        shmtext = shmtext + 'nSmoothThickness\t10;\n\t'
        shmtext = shmtext + 'maxFaceThicknessRatio\t0.5;\n\t'
        shmtext = shmtext + 'maxThicknessToMedialRatio\t0.3;\n\t'
        shmtext = shmtext + 'minMedianAxisAngle\t130;\n\t'
        shmtext = shmtext + 'nBufferCellsNoExtrude\t0;\n\t'
        shmtext = shmtext + 'nLayerIter\t50;\n'
        shmtext = shmtext + '}\n\n'

        # Mesh quality settings
        shmtext = shmtext + 'meshQualityControls\n{\n\t'
        shmtext = shmtext + 'maxNonOrtho\t180;\n\t'
        shmtext = shmtext + 'maxBoundarySkewness\t20;\n\t'
        shmtext = shmtext + 'maxInternalSkewness\t4;\n\t'
        shmtext = shmtext + 'maxConcave\t80;\n\t'
        shmtext = shmtext + 'minFlatness\t0.5;\n\t'
        shmtext = shmtext + 'minVol\t1e-13;\n\t'
        shmtext = shmtext + 'minTetQuality\t1e-30;\n\t'
        shmtext = shmtext + 'minArea\t-1;\n\t'
        shmtext = shmtext + 'minTwist\t0.02;\n\t'
        shmtext = shmtext + 'minDeterminant\t0.001;\n\t'
        shmtext = shmtext + 'minFaceWeight\t0.02;\n\t'
        shmtext = shmtext + 'minVolRatio\t0.01;\n\t'
        shmtext = shmtext + 'minTriangleTwist\t-1;\n\t'
        shmtext = shmtext + 'nSmoothScale\t4;\n\t'
        shmtext = shmtext + 'errorReduction\t0.75;\n'
        shmtext = shmtext + '}\n\n'

        # Advanced
        shmtext = shmtext + 'debug\t0;\n'
        shmtext = shmtext + 'mergeTolerance\t1E-6;\n'

        return shmtext

    #############################################################
    def scripts(self, data, path):
        """Create the scripts for caserun.sh

        Arguments:
        ---------
                data: all the JSON data
                path: Path where dakota.json file is located

        """
        # Create a utilities object
        hydroutil = hydroUtils()

        # Get the mesher
        mesher = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'MeshType'])
        )

        # For the Hydro mesher
        if int(mesher[0]) == 0:
            caseruntext = 'echo blockMesh running...\n'
            caseruntext = caseruntext + 'blockMesh > blockMesh.log\n\n'
            # surfaceFeatureExtract
            caseruntext = caseruntext + 'echo surfaceFeatureExtract running...\n'
            caseruntext = (
                caseruntext + 'surfaceFeatureExtract -force > sFeatureExt.log\n\n'
            )
            # snappyHexMesh
            caseruntext = caseruntext + 'echo snappyHexMesh running...\n'
            caseruntext = caseruntext + 'snappyHexMesh > snappyHexMesh.log\n'
            # Copy polyMesh folder
            path2c = os.path.join('2', 'polyMesh')
            caseruntext = caseruntext + 'cp -r ' + path2c + ' constant\n'
            caseruntext = caseruntext + 'rm -fr 1 2\n\n'

        elif int(mesher[0]) == 1:
            # Get the mesh software
            meshsoftware = hydroutil.extract_element_from_json(
                data, ['Events', 'MeshSoftware']
            )
            # Get the mesh file name
            meshfile = hydroutil.extract_element_from_json(
                data, ['Events', 'MeshFile']
            )
            # Get the mesh file name
            caseruntext = 'Converting the mesh files...\n'
            caseruntext = (
                caseruntext
                + 'MESHFILE=${inputDirectory}/templatedir/'
                + meshfile[0]
                + '\n\n'
            )
            # Write out the appropriate commands
            if int(meshsoftware[0]) == 0:
                caseruntext = (
                    caseruntext
                    + 'fluentMeshToFoam $MESHFILE > fluentMeshToFoam.log\n\n'
                )
            elif int(meshsoftware[0]) == 1:
                caseruntext = (
                    caseruntext + 'ideasToFoam $MESHFILE > ideasToFoam.log\n\n'
                )
            elif int(meshsoftware[0]) == 2:
                caseruntext = (
                    caseruntext + 'cfx4ToFoam $MESHFILE > cfx4ToFoam.log\n\n'
                )
            elif int(meshsoftware[0]) == 3:
                caseruntext = (
                    caseruntext + 'gambitToFoam $MESHFILE > gambitToFoam.log\n\n'
                )
            elif int(meshsoftware[0]) == 4:
                caseruntext = (
                    caseruntext + 'gmshToFoam $MESHFILE > gmshToFoam.log\n\n'
                )

        elif int(mesher[0]) == 2:
            # COPY THE FILES TO THE RIGHT LOCATION
            caseruntext = 'Copying mesh dictionaries...\n'
            # blockMesh
            bmfile = os.path.join(path, 'blockMeshDict')
            if os.path.isfile(bmfile):
                bmfilenew = os.path.join('system', 'blockMeshDict')
                caseruntext = caseruntext + 'cp ' + bmfile + ' ' + bmfilenew + '\n'
                caseruntext = caseruntext + 'echo blockMesh running...\n'
                caseruntext = caseruntext + 'blockMesh > blockMesh.log\n\n'

            # surfaceFeatureExtract
            sfdfile = os.path.join(path, 'surfaceFeatureExtractDict')
            if os.path.isfile(sfdfile):
                sfdfilenew = os.path.join('system', 'surfaceFeatureExtractDict')
                caseruntext = caseruntext + 'cp ' + sfdfile + ' ' + sfdfilenew + '\n'
                caseruntext = caseruntext + 'echo surfaceFeatureExtract running...\n'
                caseruntext = (
                    caseruntext
                    + 'surfaceFeatureExtract -force > sFeatureExt.log\n\n'
                )

            # snappyHexMesh
            shmfile = os.path.join(path, 'snappyHexMeshDict')
            if os.path.isfile(shmfile):
                shmfilenew = os.path.join('system', 'snappyHexMeshDict')
                caseruntext = caseruntext + 'cp ' + shmfile + ' ' + shmfilenew + '\n'
                caseruntext = caseruntext + 'echo snappyHexMesh running...\n'
                caseruntext = caseruntext + 'snappyHexMesh > snappyHexMesh.log\n'
                path2c = os.path.join('2', 'polyMesh')
                caseruntext = caseruntext + 'cp -r ' + path2c + ' constant\n'
                caseruntext = caseruntext + 'rm -fr 1 2\n'

        # All other items
        caseruntext = caseruntext + 'echo Checking mesh...\n'
        caseruntext = caseruntext + 'checkMesh > Meshcheck.log\n\n'

        # Create 0-folder
        caseruntext = caseruntext + 'echo Creating 0-folder...\n'
        caseruntext = caseruntext + 'rm -fr 0\n'
        caseruntext = caseruntext + 'cp -r 0.org 0\n\n'

        # Copy new force-based controldict
        caseruntext = caseruntext + 'echo Copying force-based controlDict...\n'
        caseruntext = (
            caseruntext
            + 'cp cdictforce '
            + os.path.join('system', 'controlDict')
            + '\n\n'
        )

        # Write to caserun file
        scriptfile = open('caserun.sh', 'a')
        scriptfile.write(caseruntext)
        scriptfile.close()
