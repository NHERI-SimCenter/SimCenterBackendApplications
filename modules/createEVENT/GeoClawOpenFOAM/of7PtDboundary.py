####################################################################  # noqa: INP001
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

"""  # noqa: E501, D400, D415
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
class of7PtDboundary:  # noqa: N801
    """This class includes the methods related to
    point displacement boundary conditions for openfoam7.

    Methods
    -------
            PDtext: Get all the text for the pointDisplacement-file

    """  # noqa: D205, D404

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
    # 		patch = hydroutil.extract_element_from_json(data, ["Events","VelocityType_" + patchname])  # noqa: E501
    # 		if patch == [None]:
    # 			Utype = -1
    # 		else:
    # 			Utype = ', '.join(hydroutil.extract_element_from_json(data, ["Events","VelocityType_" + patchname]))  # noqa: E501

    # 		# If any moving walls (103 - 104)  # noqa: ERA001
    # 		if (int(Utype) == 103) or (int(Utype) == 104):
    # 			print(patchname)

    #############################################################
    def PtDcheck(self, data, patches):  # noqa: ANN001, ANN201, N802
        """Checks if a point displacement for openfoam7 is required

        Arguments:
        ---------
                data: all the JSON data
                patches: List of boundary patches

        """  # noqa: D400, D401, D415
        # Create a utilities object
        hydroutil = hydroUtils()

        # Number of moving walls
        numMovWall = 0  # noqa: N806

        # Loop over all patches
        for patchname in patches:
            # Get the type of velocity bc
            patch = hydroutil.extract_element_from_json(
                data, ['Events', 'VelocityType_' + patchname]
            )
            if patch == [None]:
                Utype = -1  # noqa: N806
            else:
                Utype = ', '.join(  # noqa: N806
                    hydroutil.extract_element_from_json(
                        data, ['Events', 'VelocityType_' + patchname]
                    )
                )

            # If any moving walls (103 - 104)
            if (int(Utype) == 103) or (int(Utype) == 104):  # noqa: PLR2004
                numMovWall += 1  # noqa: N806
                if numMovWall > 0:
                    return 1

        if numMovWall == 0:
            return 0
        else:  # noqa: RET505
            return 1

    #############################################################
    def PtDtext(self, data, fipath, patches):  # noqa: ANN001, ANN201, N802, D417
        """Create text for point displacement for openfoam7

        Arguments:
        ---------
                data: all the JSON data
                patches: List of boundary patches

        """  # noqa: D400, D415
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
                Utype = -1  # noqa: N806
            else:
                Utype = ', '.join(  # noqa: N806
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
        return ptdtext  # noqa: RET504

    #############################################################
    def PtDheader(self):  # noqa: ANN201, N802
        """Creates the text for the header

        Variable
        -----------
                header: Header for the pointDisplacement-file
        """  # noqa: D400, D401, D415
        header = """/*--------------------------*- NHERI SimCenter -*----------------------------*\\ 
|	   | H |
|	   | Y | HydroUQ: Water-based Natural Hazards Modeling Application
|======| D | Website: simcenter.designsafe-ci.org/research-tools/hydro-uq
|	   | R | Version: 1.00
|	   | O |
\\*---------------------------------------------------------------------------*/ 
FoamFile
{\n\tversion\t2.0;\n\tformat\tascii;\n\tclass\tpointVectorField;\n\tlocation\t"0.01";\n\tobject\tpointDisplacement;\n}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"""  # noqa: E501, W291

        header = header + 'dimensions\t[0 1 0 0 0 0 0];\n\n'
        header = header + 'internalField\tuniform (0 0 0);\n\n'

        # Return the header for U file
        return header  # noqa: RET504

    #############################################################
    def PtDpatchtext(self, data, Utype, patchname, fipath):  # noqa: ANN001, ANN201, ARG002, N802, N803, D417
        """Creates the text the pointDisplacement boundary condition

        Arguments:
        ---------
                data: All the json data
                Utype: Type of velocity b.c
                patchname: Name of the patch
                fipath: Path to where dakota.json file exists

        Variable
        -----------
                PtDtext: Text for the particular patch

        """  # noqa: D400, D401, D415
        # Get hte normal of the patch
        normal = self.getNormal(patchname)

        # For each patch / type provide the text
        # Moving walls
        if (int(Utype) == 103) or (int(Utype) == 104):  # noqa: PLR2004
            PtDtext = '\t{\n\t\t'  # noqa: N806
            PtDtext = PtDtext + 'type\twavemakerMovement;\n\t\t'  # noqa: N806
            PtDtext = PtDtext + 'wavemakerDictName\twavemakerMovementDict;\n\t\t'  # noqa: N806
            PtDtext = PtDtext + 'value\tuniform (0 0 0);\n'  # noqa: N806
            PtDtext = PtDtext + '\t}\n'  # noqa: N806

        elif int(Utype) > 300:  # noqa: PLR2004
            PtDtext = '\t{\n\t\t'  # noqa: N806
            PtDtext = PtDtext + 'type\tfixedNormalSlip;\n\t\t'  # noqa: N806
            PtDtext = PtDtext + 'n\t(' + normal + ');\n\t\t'  # noqa: N806
            PtDtext = PtDtext + 'value\tuniform (0 0 0);\n'  # noqa: N806
            PtDtext = PtDtext + '\t}\n'  # noqa: N806

        elif (int(Utype) > 200) and (int(Utype) < 300):  # noqa: PLR2004
            PtDtext = '\t{\n\t\t'  # noqa: N806
            PtDtext = PtDtext + 'type\tfixedValue;\n\t\t'  # noqa: N806
            PtDtext = PtDtext + 'value\tuniform (0 0 0);\n'  # noqa: N806
            PtDtext = PtDtext + '\t}\n'  # noqa: N806

        else:
            PtDtext = '\t{\n\t\t'  # noqa: N806
            PtDtext = PtDtext + 'type\tfixedValue;\n\t\t'  # noqa: N806
            PtDtext = PtDtext + 'value\tuniform (0 0 0);\n'  # noqa: N806
            PtDtext = PtDtext + '\t}\n'  # noqa: N806

        return PtDtext

    #############################################################
    def getNormal(self, patchname):  # noqa: ANN001, ANN201, N802
        """Get the normal to the patch

        Arguments:
        ---------
                patchname: Name of the patch

        Variable
        -----------
                normal: Normal to the patch

        """  # noqa: D400, D415
        if (patchname == 'Entry') or (patchname == 'Exit'):  # noqa: PLR1714
            normal = '1 0 0'
        elif (patchname == 'Left') or (patchname == 'Right'):  # noqa: PLR1714
            normal = '0 1 0'
        elif (patchname == 'Bottom') or (patchname == 'Top'):  # noqa: PLR1714
            normal = '0 0 1'
        elif (patchname == 'Building') or (patchname == 'OtherBuilding'):  # noqa: PLR1714
            normal = '1 0 0'

        return normal
