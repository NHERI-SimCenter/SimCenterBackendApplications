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

from flume import flume

# Other custom modules
from hydroUtils import hydroUtils


####################################################################
# OpenFOAM7 solver class
####################################################################
class userFlume:
    """This class includes the methods related to
    creating a flume as specified by the user

    Methods
    -------
            creategeom: Create geometry and STL files

    """

    #############################################################
    def creategeom(self, data, path):
        """Creates the geometry for user flume

        Arguments:
        ---------
                data: all the JSON data

        """
        # Create a utilities object
        hydroutil = hydroUtils()

        # Read the flume segments
        flumesegs = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'FlumeSegments'])
        )
        # Get the number of flume segments
        numflumesegs = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'NumFlumeSegments'])
        )

        # Replace the comma by spaces in segments list
        flumesegs = flumesegs.replace(',', ' ')
        # Convert the flume segment to list of floats
        nums = [float(n) for n in flumesegs.split()]
        # Remove the first item
        nums.pop(0)

        # Create temporary file
        filename = 'FlumeData.txt'
        if os.path.exists(filename):
            os.remove(filename)
        f = open(filename, 'a')
        for ii in range(int(numflumesegs)):
            f.write(str(nums[2 * ii]) + ',' + str(nums[2 * ii + 1]) + '\n')
        f.close()

        # Get the breadth
        breadthval = ''.join(
            hydroutil.extract_element_from_json(data, ['Events', 'FlumeBreadth'])
        )
        breadth = float(breadthval)

        # Create the STL file and get extreme file (needed for blockmesh and building)
        flumeobj = flume()
        extreme = flumeobj.generateflume(breadth, path)

        # Write extreme values and building data to temporary file for later usage
        flumeobj.extremedata(extreme, breadth)

        return 0
