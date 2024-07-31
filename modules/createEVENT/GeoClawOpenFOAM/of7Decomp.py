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

"""  # noqa: D400, D415
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
class of7Decomp:  # noqa: N801
    """This class includes the methods related to
    parallelization for openfoam7.

    Methods
    -------
            decomptext: Get all the text for the decomposeParDict

    """  # noqa: D205, D404

    #############################################################
    def decomptext(self, data):  # noqa: ANN001, ANN201
        """Creates the necessary files for domain decomposition for openfoam7

        Arguments:
        ---------
                data: all the JSON data

        """  # noqa: D400, D401, D415
        # Create a utilities object
        hydroutil = hydroUtils()

        # Get the header text for the U-file
        decomptext = self.decompheader()

        # Get number of subdomains
        subdomains = ', '.join(
            hydroutil.extract_element_from_json(
                data, ['Events', 'DomainDecomposition']
            )
        )

        decomptext = decomptext + 'numberOfSubdomains\t' + subdomains + ';\n\n'

        decomptext = decomptext + 'method\tscotch;\n\n'

        return decomptext  # noqa: RET504

    #############################################################
    def decompheader(self):  # noqa: ANN201
        """Creates the text for the header

        Variable
        -----------
                header: Header for the decomposeparDict-file
        """  # noqa: D400, D401, D415
        header = """/*--------------------------*- NHERI SimCenter -*----------------------------*\\ 
|	   | H |
|	   | Y | HydroUQ: Water-based Natural Hazards Modeling Application
|======| D | Website: simcenter.designsafe-ci.org/research-tools/hydro-uq
|	   | R | Version: 1.00
|	   | O |
\\*---------------------------------------------------------------------------*/ 
FoamFile
{\n\tversion\t2.0;\n\tformat\tascii;\n\tclass\tdictionary;\n\tlocation\t"system";\n\tobject\tdecomposeParDict;\n}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"""  # noqa: W291

        # Return the header for U file
        return header  # noqa: RET504

    #############################################################
    def scripts(self, data, path):  # noqa: ANN001, ANN201, ARG002
        """Create the scripts for caserun.sh

        Arguments:
        ---------
                data: all the JSON data
                path: Path where dakota.json file is located

        """  # noqa: D400, D415
        # Create a utilities object
        hydroutil = hydroUtils()

        # Get number of subdomains
        totalprocs = ', '.join(
            hydroutil.extract_element_from_json(
                data, ['Events', 'DomainDecomposition']
            )
        )

        # Get simulation type
        simtype = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'SimulationType'])
        )

        # Decompose for parallel, else serial
        if int(totalprocs) > 1:
            # Decompose the domain
            caseruntext = 'echo Decomposing domain...\n'
            caseruntext = caseruntext + 'decomposePar > decomposePar.log\n\n'

            # Start the CFD simulation
            caseruntext = (
                caseruntext + 'echo Starting CFD simulation in parallel...\n'
            )
            if int(simtype) == 4:  # noqa: PLR2004
                caseruntext = (
                    caseruntext
                    + 'ibrun -n '
                    + totalprocs
                    + ' -o 0 olaDyMFlow -parallel > olaDyMFlow.log\n\n'
                )
            else:
                caseruntext = (
                    caseruntext
                    + 'ibrun -n '
                    + totalprocs
                    + ' -o 0 olaFlow -parallel > olaFlow.log\n\n'
                )

        else:
            caseruntext = 'echo Starting CFD simulation in serial...\n'
            if int(simtype) == 4:  # noqa: PLR2004
                caseruntext = caseruntext + 'olaDyMFlow > olaDyMFlow.log\n\n'
            else:
                caseruntext = caseruntext + 'olaFlow > olaFlow.log\n\n'

        # Write to caserun file
        scriptfile = open('caserun.sh', 'a')  # noqa: SIM115, PTH123
        scriptfile.write(caseruntext)
        scriptfile.close()
