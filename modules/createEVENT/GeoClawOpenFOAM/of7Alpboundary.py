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
class of7Alpboundary:  # noqa: N801
    """This class includes the methods related to
    alpha boundary conditions for openfoam7.

    Methods
    -------
            Alptext: Get all the text for the p_rgh-file

    """  # noqa: D205, D404

    #############################################################
    def Alptext(self, data, patches):  # noqa: ANN001, ANN201, N802, D417
        """Creates the necessary text for pressure bc for openfoam7

        Arguments:
        ---------
                data: all the JSON data

        """  # noqa: D400, D401, D415
        # Create a utilities object
        hydroutil = hydroUtils()

        # Get the header text for the U-file
        Alptext = self.Alpheader()  # noqa: N806

        # Start the outside
        Alptext = Alptext + 'boundaryField\n{\n'  # noqa: N806

        # Loop over all patches
        for patchname in patches:
            Alptext = Alptext + '\t' + patchname + '\n'  # noqa: N806
            patch = hydroutil.extract_element_from_json(
                data, ['Events', 'PressureType_' + patchname]
            )
            if patch == [None]:  # noqa: SIM108
                Alptype = -1  # noqa: N806
            else:
                Alptype = 0  # noqa: N806
            Alptext = Alptext + self.Alppatchtext(Alptype, patchname)  # noqa: N806

        # Check for building and other building
        Alptext = Alptext + '\tBuilding\n'  # noqa: N806
        Alptext = Alptext + self.Alppatchtext(0, 'Building')  # noqa: N806
        Alptext = Alptext + '\tOtherBuilding\n'  # noqa: N806
        Alptext = Alptext + self.Alppatchtext(0, 'OtherBuilding')  # noqa: N806

        # Close the outside
        Alptext = Alptext + '}\n\n'  # noqa: N806

        # Return the text for velocity BC
        return Alptext  # noqa: RET504

    #############################################################
    def Alpheader(self):  # noqa: ANN201, N802
        """Creates the text for the header for pressure file

        Variable
        -----------
                header: Header for the p_rgh-file
        """  # noqa: D400, D401, D415
        header = """/*--------------------------*- NHERI SimCenter -*----------------------------*\\ 
|	   | H |
|	   | Y | HydroUQ: Water-based Natural Hazards Modeling Application
|======| D | Website: simcenter.designsafe-ci.org/research-tools/hydro-uq
|	   | R | Version: 1.00
|	   | O |
\\*---------------------------------------------------------------------------*/ 
FoamFile
{\n\tversion\t2.0;\n\tformat\tascii;\n\tclass\tvolScalarField;\n\tlocation\t"0";\n\tobject\talpha.water;\n}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"""  # noqa: E501, W291

        header = header + 'dimensions\t[0 0 0 0 0 0 0];\n\n'
        header = header + 'internalField\tuniform\t0;\n\n'

        # Return the header for U file
        return header  # noqa: RET504

    #############################################################
    def Alppatchtext(self, Alptype, patchname):  # noqa: ANN001, ANN201, ARG002, N802, N803, D417
        """Creates the text the pressure boundary condition

        Arguments:
        ---------
                patchname: Name of the patch

        Variable
        -----------
                Alptext: Text for the particular patch

        """  # noqa: D400, D401, D415
        if patchname == 'Top':
            Alptext = '\t{\n\t\t'  # noqa: N806
            Alptext = Alptext + 'type\tinletOutlet;\n\t\t'  # noqa: N806
            Alptext = Alptext + 'inletValue\tuniform 0;\n\t\t'  # noqa: N806
            Alptext = Alptext + 'value\tuniform 0;\n\t}\n'  # noqa: N806

        else:
            Alptext = '\t{\n\t\t'  # noqa: N806
            Alptext = Alptext + 'type\tzeroGradient;\n\t}\n'  # noqa: N806

        # Return the header for U file
        return Alptext
