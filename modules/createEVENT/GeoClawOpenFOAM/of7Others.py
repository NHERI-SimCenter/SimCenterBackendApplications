#
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

# Other custom modules
from hydroUtils import hydroUtils


####################################################################
# OpenFOAM7 solver class
####################################################################
class of7Others:
    """This class includes the methods related to
    auxiliary files for openfoam7.

    Methods
    -------
            gfiletext: Get all the text for the gravity file

    """

    #############################################################
    def othersheader(self, fileclas, fileloc, fileobjec):
        """Creates the text for the header

        Variable
        -----------
                header: Header for the other-files
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
{\n\tversion\t2.0;\n\tformat\tascii;\n\tclass\t"""
            + fileclas
            + """;\n\tlocation\t"""
            + '"'
            + fileloc
            + """";\n\tobject\t"""
            + fileobjec
            + """;\n}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"""
        )

        # Return the header for U file
        return header

    #############################################################
    def gfiletext(self, data):
        """Creates the necessary text for gravity file for openfoam7

        Arguments:
        ---------
                data: all the JSON data

        """
        # Create a utilities object
        hydroutil = hydroUtils()

        # Initialize gravity
        gx = 0.0
        gy = 0.0
        gz = 0.0

        simtype = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'SimulationType'])
        )

        if int(simtype) == 4:
            gz = -9.81
        else:
            # Get the gravity from dakota.json file
            gravity = ', '.join(
                hydroutil.extract_element_from_json(data, ['Events', 'Gravity'])
            )
            # Depending on the inputs, initialize gravity in the right direction
            if int(gravity) == 11:
                gx = 9.81
            elif int(gravity) == 12:
                gy = 9.81
            elif int(gravity) == 13:
                gz = 9.81
            elif int(gravity) == 21:
                gx = -9.81
            elif int(gravity) == 22:
                gy = -9.81
            elif int(gravity) == 23:
                gz = -9.81

        # Get the header text for the gravity-file
        gfiletext = self.othersheader(
            'uniformDimensionedVectorField', 'constant', 'g'
        )

        # All other data
        gfiletext = gfiletext + 'dimensions\t[0 1 -2 0 0 0 0];\n'
        gfiletext = (
            gfiletext
            + 'value\t('
            + str(gx)
            + '\t'
            + str(gy)
            + '\t'
            + str(gz)
            + ');\n'
        )

        return gfiletext
