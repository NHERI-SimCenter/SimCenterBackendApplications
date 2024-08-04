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

import numpy as np

# Other custom modules
from flume import flume


####################################################################
# OpenFOAM7 solver class
####################################################################
class osuFlume:
    """This class includes the methods related to
    creating a standard OSU flume

    Methods
    -------
            creategeom: Create geometry and STL files

    """  # noqa: D205, D400, D404

    #############################################################
    def creategeom(self, data, path):  # noqa: ANN001, ANN201, ARG002, PLR6301
        """Creates the geometry for OSU flume

        Arguments:
        ---------
                data: all the JSON data

        """  # noqa: D400, D401
        # Number of flume points
        numflumepoints = 9

        # Provide the coordinates in a numpy array
        nums = np.zeros(2 * (numflumepoints))
        nums[0] = -2.085
        nums[1] = 0.0
        nums[2] = 14.2748
        nums[3] = 0.0
        nums[4] = 14.2748
        nums[5] = 0.1524
        nums[6] = 17.9324
        nums[7] = 0.1524
        nums[8] = 28.9052
        nums[9] = 1.15
        nums[10] = 43.5356
        nums[11] = 1.7526
        nums[12] = 80.116
        nums[13] = 1.7526
        nums[14] = 80.116
        nums[15] = 4.572
        nums[16] = -2.085
        nums[17] = 4.572

        # Create temporary file
        filename = 'FlumeData.txt'
        if os.path.exists(filename):  # noqa: PTH110
            os.remove(filename)  # noqa: PTH107
        f = open(filename, 'a')  # noqa: PLW1514, PTH123, SIM115
        for ii in range(int(numflumepoints)):
            f.write(str(nums[2 * ii]) + ',' + str(nums[2 * ii + 1]) + '\n')
        f.close()

        # Add breadth of the flume
        breadth = 3.70

        # Create the STL file and get extreme file (needed for blockmesh and building)
        flumeobj = flume()
        extreme = flumeobj.generateflume(breadth, path)

        # Write extreme values and building data to temporary file for later usage
        flumeobj.extremedata(extreme, breadth)

        return 0
