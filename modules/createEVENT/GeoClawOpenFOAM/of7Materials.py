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

# Other custom modules
from hydroUtils import hydroUtils


####################################################################
# OpenFOAM7 solver class
####################################################################
class of7Materials:
    """This class includes the methods related to
    material properties for openfoam7.

    Methods
    -------
            mattext: Get all the text for the transportProperties

    """  # noqa: D205, D404

    #############################################################
    def mattext(self, data):
        """Creates the necessary files for materials for openfoam7

        Arguments:
        ---------
                data: all the JSON data

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()

        # Get the header text for the U-file
        mattext = self.matheader()

        # Start by stating phases
        mattext = mattext + 'phases (water air);\n\n'  # noqa: PLR6104

        # Water phase
        # Viscosity
        nuwater = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'WaterViscosity'])
        )
        # Exponent
        nuwaterexp = ', '.join(
            hydroutil.extract_element_from_json(
                data, ['Events', 'WaterViscosityExp']
            )
        )
        # Density
        rhowater = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'WaterDensity'])
        )

        mattext = mattext + 'water\n{\n'  # noqa: PLR6104
        mattext = mattext + '\ttransportModel\tNewtonian;\n'  # noqa: PLR6104
        mattext = (
            mattext + '\tnu\t[0 2 -1 0 0 0 0]\t' + nuwater + 'e' + nuwaterexp + ';\n'
        )
        mattext = mattext + '\trho\t[1 -3 0 0 0 0 0]\t' + rhowater + ';\n'
        mattext = mattext + '}\n\n'  # noqa: PLR6104

        # Air properties
        # Viscosity
        nuair = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'AirViscosity'])
        )
        # Exponent
        nuairexp = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'AirViscosityExp'])
        )
        # Density
        rhoair = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'AirDensity'])
        )

        mattext = mattext + 'air\n{\n'  # noqa: PLR6104
        mattext = mattext + '\ttransportModel\tNewtonian;\n'  # noqa: PLR6104
        mattext = (
            mattext + '\tnu\t[0 2 -1 0 0 0 0]\t' + nuair + 'e' + nuairexp + ';\n'
        )
        mattext = mattext + '\trho\t[1 -3 0 0 0 0 0]\t' + rhoair + ';\n'
        mattext = mattext + '}\n\n'  # noqa: PLR6104

        # Surface tension between water and air
        sigma = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'SurfaceTension'])
        )

        mattext = mattext + 'sigma\t[1 0 -2 0 0 0 0]\t' + sigma + ';\n'

        return mattext  # noqa: RET504

    #############################################################
    def matheader(self):  # noqa: PLR6301
        """Creates the text for the header

        Variable
        -----------
                header: Header for the transportProp-file
        """  # noqa: D400, D401
        header = """/*--------------------------*- NHERI SimCenter -*----------------------------*\\ 
|	   | H |
|	   | Y | HydroUQ: Water-based Natural Hazards Modeling Application
|======| D | Website: simcenter.designsafe-ci.org/research-tools/hydro-uq
|	   | R | Version: 1.00
|	   | O |
\\*---------------------------------------------------------------------------*/ 
FoamFile
{\n\tversion\t2.0;\n\tformat\tascii;\n\tclass\tdictionary;\n\tlocation\t"constant";\n\tobject\ttransportProperties;\n}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"""  # noqa: W291

        # Return the header for U file
        return header  # noqa: RET504

    #############################################################
    def matcheck(self, data):  # noqa: PLR6301
        """Checks for material properties for openfoam7

        Arguments:
        ---------
                data: all the JSON data

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()

        # Check water properties
        # Viscosity
        nuwater = hydroutil.extract_element_from_json(
            data, ['Events', 'WaterViscosity']
        )
        if nuwater == [None]:
            return -1
        # Exponent
        nuwaterexp = hydroutil.extract_element_from_json(
            data, ['Events', 'WaterViscosityExp']
        )
        if nuwaterexp == [None]:
            return -1
        # Density
        rhowater = hydroutil.extract_element_from_json(
            data, ['Events', 'WaterDensity']
        )
        if rhowater == [None]:
            return -1

        # Check air properties
        # Viscosity
        nuair = hydroutil.extract_element_from_json(data, ['Events', 'AirViscosity'])
        if nuair == [None]:
            return -1
        # Exponent
        nuairexp = hydroutil.extract_element_from_json(
            data, ['Events', 'AirViscosityExp']
        )
        if nuairexp == [None]:
            return -1
        # Density
        rhoair = hydroutil.extract_element_from_json(data, ['Events', 'AirDensity'])
        if rhoair == [None]:
            return -1

        # Surface tension between water and air
        sigma = hydroutil.extract_element_from_json(
            data, ['Events', 'SurfaceTension']
        )
        if sigma == [None]:
            return -1

        # Return 0 if all is right
        return 0
