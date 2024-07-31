####################################################################
# LICENSING INFORMATION
####################################################################
"""
LICENSE INFORMATION:

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

# Other custom modules
from hydroUtils import hydroUtils


####################################################################
# OpenFOAM7 solver class
####################################################################
class of7PtDboundary:
    """
    This class includes the methods related to
    point displacement boundary conditions for openfoam7.

    Methods
    --------
            PDtext: Get all the text for the pointDisplacement-file
    """

    # #############################################################
    # def PtDtext(self,data,fipath,patches):
    # 	'''
    # 	Creates the necessary text for point displacement for openfoam7

    # 	Arguments
    # 	-----------
    # 		data: all the JSON data
    # 		patches: List of boundary patches
    # 		fipath: Path where the dakota.json file exists
    # 	'''

    # 	# Create a utilities object
    # 	hydroutil = hydroUtils()

    # 	# Number of moving walls
    # 	numMovWall = 0

    # 	# Loop over all patches
    # 	for patchname in patches:
    # 		# Get the type of velocity bc
    # 		patch = hydroutil.extract_element_from_json(data, ["Events","VelocityType_" + patchname])
    # 		if patch == [None]:
    # 			Utype = -1
    # 		else:
    # 			Utype = ', '.join(hydroutil.extract_element_from_json(data, ["Events","VelocityType_" + patchname]))

    # 		# If any moving walls (103 - 104)
    # 		if (int(Utype) == 103) or (int(Utype) == 104):
    # 			print(patchname)

    #############################################################
    def PtDcheck(self, data, patches):
        """
        Checks if a point displacement for openfoam7 is required

        Arguments
        -----------
                data: all the JSON data
                patches: List of boundary patches
        """

        # Create a utilities object
        hydroutil = hydroUtils()

        # Number of moving walls
        numMovWall = 0

        # Loop over all patches
        for patchname in patches:
            # Get the type of velocity bc
            patch = hydroutil.extract_element_from_json(
                data, ['Events', 'VelocityType_' + patchname]
            )
            if patch == [None]:
                Utype = -1
            else:
                Utype = ', '.join(
                    hydroutil.extract_element_from_json(
                        data, ['Events', 'VelocityType_' + patchname]
                    )
                )

            # If any moving walls (103 - 104)
            if (int(Utype) == 103) or (int(Utype) == 104):
                numMovWall += 1
                if numMovWall > 0:
                    return 1

        if numMovWall == 0:
            return 0
        else:
            return 1

    #############################################################
    def PtDtext(self, data, fipath, patches):
        """
        Create text for point displacement for openfoam7

        Arguments
        -----------
                data: all the JSON data
                patches: List of boundary patches
        """

        # Create a utilities object
        hydroutil = hydroUtils()

        # Get the header text for the U-file
        ptdtext = self.PtDheader()

        # Start the outside
        ptdtext = ptdtext + 'boundaryField\n{\n'

        # Loop over all patch
        for patchname in patches:
            ptdtext = ptdtext + '\t' + patchname + '\n'
            # Get the type of velocity bc
            patch = hydroutil.extract_element_from_json(
                data, ['Events', 'VelocityType_' + patchname]
            )
            if patch == [None]:
                Utype = -1
            else:
                Utype = ', '.join(
                    hydroutil.extract_element_from_json(
                        data, ['Events', 'VelocityType_' + patchname]
                    )
                )

            ptdtext = ptdtext + self.PtDpatchtext(data, Utype, patchname, fipath)

        # Check for building and other building
        ptdtext = ptdtext + '\tBuilding\n'
        ptdtext = ptdtext + self.PtDpatchtext(data, '301', 'Building', fipath)
        ptdtext = ptdtext + '\tOtherBuilding\n'
        ptdtext = ptdtext + self.PtDpatchtext(data, '301', 'OtherBuilding', fipath)

        # Close the outside
        ptdtext = ptdtext + '}\n\n'

        # Return the text for pointDisplacement
        return ptdtext

    #############################################################
    def PtDheader(self):
        """
        Creates the text for the header

        Variable
        -----------
                header: Header for the pointDisplacement-file
        """

        header = """/*--------------------------*- NHERI SimCenter -*----------------------------*\ 
|	   | H |
|	   | Y | HydroUQ: Water-based Natural Hazards Modeling Application
|======| D | Website: simcenter.designsafe-ci.org/research-tools/hydro-uq
|	   | R | Version: 1.00
|	   | O |
\*---------------------------------------------------------------------------*/ 
FoamFile
{\n\tversion\t2.0;\n\tformat\tascii;\n\tclass\tpointVectorField;\n\tlocation\t"0.01";\n\tobject\tpointDisplacement;\n}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"""

        header = header + 'dimensions\t[0 1 0 0 0 0 0];\n\n'
        header = header + 'internalField\tuniform (0 0 0);\n\n'

        # Return the header for U file
        return header

    #############################################################
    def PtDpatchtext(self, data, Utype, patchname, fipath):
        """
        Creates the text the pointDisplacement boundary condition

        Arguments
        -----------
                data: All the json data
                Utype: Type of velocity b.c
                patchname: Name of the patch
                fipath: Path to where dakota.json file exists

        Variable
        -----------
                PtDtext: Text for the particular patch
        """

        # Get hte normal of the patch
        normal = self.getNormal(patchname)

        # For each patch / type provide the text
        # Moving walls
        if (int(Utype) == 103) or (int(Utype) == 104):
            PtDtext = '\t{\n\t\t'
            PtDtext = PtDtext + 'type\twavemakerMovement;\n\t\t'
            PtDtext = PtDtext + 'wavemakerDictName\twavemakerMovementDict;\n\t\t'
            PtDtext = PtDtext + 'value\tuniform (0 0 0);\n'
            PtDtext = PtDtext + '\t}\n'

        elif int(Utype) > 300:
            PtDtext = '\t{\n\t\t'
            PtDtext = PtDtext + 'type\tfixedNormalSlip;\n\t\t'
            PtDtext = PtDtext + 'n\t(' + normal + ');\n\t\t'
            PtDtext = PtDtext + 'value\tuniform (0 0 0);\n'
            PtDtext = PtDtext + '\t}\n'

        elif (int(Utype) > 200) and (int(Utype) < 300):
            PtDtext = '\t{\n\t\t'
            PtDtext = PtDtext + 'type\tfixedValue;\n\t\t'
            PtDtext = PtDtext + 'value\tuniform (0 0 0);\n'
            PtDtext = PtDtext + '\t}\n'

        else:
            PtDtext = '\t{\n\t\t'
            PtDtext = PtDtext + 'type\tfixedValue;\n\t\t'
            PtDtext = PtDtext + 'value\tuniform (0 0 0);\n'
            PtDtext = PtDtext + '\t}\n'

        return PtDtext

    #############################################################
    def getNormal(self, patchname):
        """
        Get the normal to the patch

        Arguments
        -----------
                patchname: Name of the patch

        Variable
        -----------
                normal: Normal to the patch
        """

        if (patchname == 'Entry') or (patchname == 'Exit'):
            normal = '1 0 0'
        elif (patchname == 'Left') or (patchname == 'Right'):
            normal = '0 1 0'
        elif (patchname == 'Bottom') or (patchname == 'Top'):
            normal = '0 0 1'
        elif (patchname == 'Building') or (patchname == 'OtherBuilding'):
            normal = '1 0 0'

        return normal
