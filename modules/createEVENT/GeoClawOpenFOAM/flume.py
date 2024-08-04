# # noqa: INP001
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
import triangle as tr
from shapely.geometry import Point, Polygon

# Other custom modules
# from hydroUtils import hydroUtils


####################################################################
# OpenFOAM7 solver class
####################################################################
class flume:
    """This class includes the methods related to wave flume

    Methods
    -------
            generateflume: Create STL files for the flume
            extremedata: Get the extreme values and building information

    """  # noqa: D400, D404

    #############################################################
    def generateflume(self, breadth, path):  # noqa: ANN001, ANN201
        """Creates the STL files for the flume

        Arguments:
        ---------
                breadth: Breadth f the flume
                path: Path where dakota.json exists - where we need to put STL files

        """  # noqa: D400, D401
        # Get the triangulated flume
        extremeval = self.flumedata('FlumeData.txt')
        self.breadth = breadth

        # Right face
        self.right()  # Right vertices
        self.npt_right = self.npt  # Right triangles
        self.writeSTL(
            'Right', self.npa_right, self.npt_right, path
        )  # Write right STL file

        # Left face
        self.left()  # Left vertices
        self.lefttri()  # Left triangles
        self.writeSTL(
            'Left', self.npa_left, self.npt_left, path
        )  # Write left STL file

        # Front face
        self.front()  # Front faces
        self.fronttri()  # Front triangles
        self.writeSTL(
            'Entry', self.npa_front, self.npt_front, path
        )  # Write front STL file

        # Back face
        self.back()  # Back vertices
        self.backtri()  # Back triangles
        self.writeSTL(
            'Exit', self.npa_back, self.npt_back, path
        )  # Write back STL file

        # Top face
        self.top()  # Top vertices
        self.toptri()  # Top triangles
        self.writeSTL('Top', self.npa_top, self.npt_top, path)  # Write top STL file

        # Bottom face
        self.bottom()  # Bottom vertices
        self.bottomtri()  # Bottom triangles
        self.writeSTL(
            'Bottom', self.npa_bottom, self.npt_bottom, path
        )  # Write bottom STL file

        # Return extreme values
        return extremeval

    #############################################################
    def flumedata(self, IpPTFile):  # noqa: ANN001, ANN201, N803
        """Gets information about the flume to create STL files

        Arguments:
        ---------
                IpPTFile: File with points of the flume

        """  # noqa: D400, D401
        # Get the data for the boundary
        data_boun = np.genfromtxt(IpPTFile, delimiter=',', dtype=(float, float))

        # Add extremum to the constants file
        maxvalues = np.max(data_boun, axis=0)
        minvalues = np.min(data_boun, axis=0)
        extremeval = np.array(
            [minvalues[0], maxvalues[0], minvalues[1], maxvalues[1]]
        )

        # Initialize segments for left and right
        segmentLR = []  # noqa: N806

        # Loop over all coordinates and create coordinates
        for ii in range(data_boun.shape[0]):
            # Get each of the user points
            if ii < data_boun.shape[0] - 1:
                segmentLR.extend([(ii, ii + 1)])
            else:
                segmentLR.extend([(ii, 0)])

        # Triangulate the polygon
        ALR = dict(vertices=data_boun, segments=segmentLR)  # noqa: C408, N806
        BLR = tr.triangulate(ALR)  # noqa: N806

        # Get the tringles and vertices
        nm_triangle = BLR['triangles'].tolist()
        self.npt = np.asarray(nm_triangle, dtype=np.int32)
        nm_vertices = BLR['vertices'].tolist()
        self.npa = np.asarray(nm_vertices, dtype=np.float32)

        # Define the polygon
        mypoly = Polygon(data_boun)

        # Loop over all triangles to find if inside polygon
        indexes = []
        noindexes = []
        for ii in range(self.npt.shape[0]):
            n0 = self.npt[ii, 0]
            n1 = self.npt[ii, 1]
            n2 = self.npt[ii, 2]
            centroidX = (1 / 3) * (  # noqa: N806
                self.npa[n0, 0] + self.npa[n1, 0] + self.npa[n2, 0]
            )
            centroidZ = (1 / 3) * (  # noqa: N806
                self.npa[n0, 1] + self.npa[n1, 1] + self.npa[n2, 1]
            )
            po = Point(centroidX, centroidZ)
            if mypoly.contains(po):
                indexes.extend([(ii)])
            else:
                noindexes.extend([(ii)])

        # Delete extra triangles
        self.npt = np.delete(self.npt, noindexes, axis=0)

        # Return extreme values
        return extremeval

    ####################################################################
    def right(self):  # noqa: ANN201
        """Gets information/nodes about to create right face of the flume

        Arguments:
        ---------
                none

        """  # noqa: D400, D401
        self.npa_right = np.zeros(shape=(self.npa.shape[0], 3))
        self.npa_right[:, 0] = self.npa[:, 0]
        self.npa_right[:, 2] = self.npa[:, 1]
        self.npa_right[:, 1] = -self.breadth / 2

    ####################################################################
    def left(self):  # noqa: ANN201
        """Gets information/nodes about to create left face of the flume

        Arguments:
        ---------
                none

        """  # noqa: D400, D401
        self.npa_left = np.zeros(shape=(self.npa.shape[0], 3))
        self.npa_left[:, 0] = self.npa[:, 0]
        self.npa_left[:, 2] = self.npa[:, 1]
        self.npa_left[:, 1] = self.breadth / 2

    ####################################################################
    def lefttri(self):  # noqa: ANN201
        """Define triangles of the left face of the flume

        Arguments:
        ---------
                none

        """  # noqa: D400
        self.npt_left = np.array(self.npt)
        self.npt_left[:, [1, 0]] = self.npt_left[:, [0, 1]]

    ####################################################################
    def front(self):  # noqa: ANN201
        """Define information/nodes of the front face of the flume

        Arguments:
        ---------
                none

        """  # noqa: D400
        self.npa_front = np.zeros(shape=(4, 3))
        self.npa_front[0, :] = self.npa_right[0, :]
        self.npa_front[1, :] = self.npa_right[self.npa_right.shape[0] - 1, :]
        self.npa_front[2, :] = self.npa_left[0, :]
        self.npa_front[3, :] = self.npa_left[self.npa_left.shape[0] - 1, :]

    ####################################################################
    def fronttri(self):  # noqa: ANN201
        """Define triangles of the front face of the flume

        Arguments:
        ---------
                none

        """  # noqa: D400
        self.npt_front = np.array([[0, 1, 2], [1, 3, 2]])

    ####################################################################
    def back(self):  # noqa: ANN201
        """Define information/nodes of the back face of the flume

        Arguments:
        ---------
                none

        """  # noqa: D400
        self.npa_back = np.zeros(shape=(4, 3))
        self.npa_back[0, :] = self.npa_right[self.npa_right.shape[0] - 3, :]
        self.npa_back[1, :] = self.npa_right[self.npa_right.shape[0] - 2, :]
        self.npa_back[2, :] = self.npa_left[self.npa_left.shape[0] - 3, :]
        self.npa_back[3, :] = self.npa_left[self.npa_left.shape[0] - 2, :]

    ####################################################################
    def backtri(self):  # noqa: ANN201
        """Define triangles of the back face of the flume

        Arguments:
        ---------
                none

        """  # noqa: D400
        self.npt_back = np.array([[3, 1, 0], [0, 2, 3]])

    ####################################################################
    def top(self):  # noqa: ANN201
        """Define information/nodes of the top face of the flume

        Arguments:
        ---------
                none

        """  # noqa: D400
        self.npa_top = np.zeros(shape=(4, 3))
        self.npa_top[0, :] = self.npa_right[self.npa_right.shape[0] - 1, :]
        self.npa_top[1, :] = self.npa_right[self.npa_right.shape[0] - 2, :]
        self.npa_top[2, :] = self.npa_left[self.npa_left.shape[0] - 1, :]
        self.npa_top[3, :] = self.npa_left[self.npa_left.shape[0] - 2, :]

    ####################################################################
    def toptri(self):  # noqa: ANN201
        """Define triangles of the top face of the flume

        Arguments:
        ---------
                none

        """  # noqa: D400
        self.npt_top = np.array([[2, 0, 1], [2, 1, 3]])

    ####################################################################
    def bottom(self):  # noqa: ANN201
        """Define information/nodes of the bottom face of the flume

        Arguments:
        ---------
                none

        """  # noqa: D400
        # Create the coordinate vector
        self.npa_bottom = []

        # Loop over all the points
        for ii in range(self.npa_right.shape[0] - 3):
            npa_temp1 = np.zeros(shape=(4, 3))
            npa_temp2 = np.zeros(shape=(2, 3))

            # Get the points
            if ii == 0:
                npa_temp1[0, :] = self.npa_right[ii, :]
                npa_temp1[1, :] = self.npa_left[ii, :]
                npa_temp1[2, :] = self.npa_right[ii + 1, :]
                npa_temp1[3, :] = self.npa_left[ii + 1, :]
            else:
                npa_temp2[0, :] = self.npa_right[ii + 1, :]
                npa_temp2[1, :] = self.npa_left[ii + 1, :]

            # Concatenate as necessary
            if ii == 0:
                self.npa_bottom = npa_temp1
            else:
                self.npa_bottom = np.concatenate(
                    (self.npa_bottom, npa_temp2), axis=0
                )

    ####################################################################
    def bottomtri(self):  # noqa: ANN201
        """Define triangles of the bottom face of the flume

        Arguments:
        ---------
                none

        """  # noqa: D400
        # Create the coordinate vector
        self.npt_bottom = []
        ntri = 2

        # Loop over all the points
        for ii in range(self.npa_right.shape[0] - 3):
            npt_temp = np.zeros(shape=(2, 3))

            # Get the triangles
            npt_temp = np.array([[0, 1, 2], [1, 3, 2]])
            npt_temp = npt_temp + ii * ntri  # noqa: PLR6104

            # Concatenate as necessary
            if ii == 0:
                self.npt_bottom = npt_temp
            else:
                self.npt_bottom = np.concatenate((self.npt_bottom, npt_temp), axis=0)

    #############################################################
    def writeSTL(self, base_filename, npa, npt, path):  # noqa: ANN001, ANN201, N802, PLR6301
        """Write the STL files for each patch

        Arguments:
        ---------
                base_filename: Patchname of the flume
                npa: List of nodes
                npt: List of triangles
                path: Location where dakota.json file exists

        """  # noqa: D400
        # Create a filename
        filename = base_filename + '.stl'
        # Create the STL file
        cells = [('triangle', npt)]
        meshio.write_points_cells(filename, npa, cells)
        # Modify first and last line
        with open(filename) as f:  # noqa: PLW1514, PTH123
            lines = f.readlines()
            lines[0] = 'solid ' + base_filename + '\n'
            lines[len(lines) - 1] = 'endsolid ' + base_filename + '\n'
        # Write the updated file
        with open(filename, 'w') as f:  # noqa: PLW1514, PTH123
            f.writelines(lines)
        # Move the file to constant/triSurface folder
        newfilepath = os.path.join(path, 'constant', 'triSurface', filename)  # noqa: PTH118
        os.replace(filename, newfilepath)  # noqa: PTH105

    #############################################################
    def extremedata(self, extreme, breadth):  # noqa: ANN001, ANN201, PLR6301
        """Creates the STL files for the flume

        Arguments:
        ---------
                data: content of JSON file
                extreme: Maximum limits
                breadth: Breadth of the flume

        """  # noqa: D400, D401
        # Write the Max-Min values for the blockMesh
        BMXmin = extreme[0] - 0.25 * (extreme[1] - extreme[0])  # noqa: N806
        BMXmax = extreme[1] + 0.25 * (extreme[1] - extreme[0])  # noqa: N806
        BMYmin = -0.625 * breadth  # noqa: N806
        BMYmax = 0.625 * breadth  # noqa: N806
        BMZmin = extreme[2] - 0.25 * (extreme[3] - extreme[2])  # noqa: N806
        BMZmax = extreme[3] + 0.25 * (extreme[3] - extreme[2])  # noqa: N806

        # Write the temporary file
        filename = 'temp_geometry.txt'
        if os.path.exists(filename):  # noqa: PTH110
            os.remove(filename)  # noqa: PTH107
        tempfileID = open('temp_geometry.txt', 'w')  # noqa: N806, PLW1514, PTH123, SIM115

        # Write the extreme values to the files
        tempfileID.write(
            str(BMXmin)
            + '\n'
            + str(BMXmax)
            + '\n'
            + str(BMYmin)
            + '\n'
            + str(BMYmax)
            + '\n'
            + str(BMZmin)
            + '\n'
            + str(BMZmax)
            + '\n'
        )
        tempfileID.close  # noqa: B018

        return 0
