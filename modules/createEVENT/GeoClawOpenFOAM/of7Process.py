#  # noqa: INP001
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
from hydroUtils import hydroUtils
from of7Solve import of7Solve


####################################################################
# OpenFOAM7 solver class
####################################################################
class of7Process:
    """This class includes the methods related to
    post-processing for openfoam7.

    Methods
    -------
            pprocesstext: Get all the text for the post-processing

    """  # noqa: D205, D404

    #############################################################
    def pprocesstext(self, data, path):
        """Creates the necessary files for post-processing for openfoam7

        Arguments:
        ---------
                data: all the JSON data

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()
        solver = of7Solve()

        # Point data from file
        pprocessfile = ', '.join(
            hydroutil.extract_element_from_json(
                data, ['Events', 'PProcessFile']
            )
        )
        pprocesspath = os.path.join(path, pprocessfile)  # noqa: PTH118
        pp_data = np.genfromtxt(pprocesspath, delimiter=',')
        num_points = np.shape(pp_data)[0]
        ptext = '\t\t(\n'
        for ii in range(num_points):
            ptext = (
                ptext
                + '\t\t\t('
                + str(pp_data[ii, 0])
                + '\t'
                + str(pp_data[ii, 1])
                + '\t'
                + str(pp_data[ii, 2])
                + ')\n'
            )
        ptext = ptext + '\t\t);\n'

        # Fields required
        value = 0
        pprocessV = hydroutil.extract_element_from_json(  # noqa: N806
            data, ['Events', 'PPVelocity']
        )
        if pprocessV != [None]:
            pprocessV = ', '.join(  # noqa: N806
                hydroutil.extract_element_from_json(
                    data, ['Events', 'PPVelocity']
                )
            )
            if pprocessV == 'Yes':
                value += 1
        pprocessP = hydroutil.extract_element_from_json(  # noqa: N806
            data, ['Events', 'PPPressure']
        )
        if pprocessP != [None]:
            pprocessP = ', '.join(  # noqa: N806
                hydroutil.extract_element_from_json(
                    data, ['Events', 'PPPressure']
                )
            )
            if pprocessP == 'Yes':
                value += 2
        if value == 1:
            fieldtext = '(U)'
        elif value == 2:  # noqa: PLR2004
            fieldtext = '(p_rgh)'
        else:
            fieldtext = '(U p_rgh)'

        # Get the header text for the U-file
        sampletext = solver.solverheader('sample')

        # Other information
        sampletext = sampletext + '\ntype sets;\n'
        sampletext = sampletext + 'libs\t("libsampling.so");\n\n'
        sampletext = sampletext + 'interpolationScheme\tcellPoint;\n\n'
        sampletext = sampletext + 'setFormat\traw;\n\n'
        sampletext = sampletext + 'sets\n(\n\tdata\n\t{\n'
        sampletext = sampletext + '\t\ttype\tpoints;\n'
        sampletext = sampletext + '\t\tpoints\n'
        sampletext = sampletext + ptext
        sampletext = sampletext + '\t\tordered\tyes;\n'
        sampletext = sampletext + '\t\taxis\tx;\n'
        sampletext = sampletext + '\t}\n'
        sampletext = sampletext + ');\n\n'
        sampletext = sampletext + 'fields\t' + fieldtext + ';\n'

        return sampletext  # noqa: DOC201, RET504, RUF100

    #############################################################
    def pprocesscdict(self, data, path):  # noqa: C901
        """Creates the necessary files for new controldict for post-processing for openfoam7

        Arguments:
        ---------
                data: all the JSON data

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()
        solver = of7Solve()

        # Get the header text for the U-file
        cdicttext = solver.solverheader('controlDict')

        # Get the simulation type: Solver
        simtype = ', '.join(
            hydroutil.extract_element_from_json(
                data, ['Events', 'SimulationType']
            )
        )
        if int(simtype) == 4:  # noqa: PLR2004
            cdicttext = cdicttext + '\napplication \t olaDyMFlow;\n\n'
        else:
            cdicttext = cdicttext + '\napplication \t olaFlow;\n\n'

        # Check restart situation and give start time
        restart = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'Restart'])
        )
        if restart == 'Yes':
            cdicttext = cdicttext + 'startFrom \t latestTime;\n\n'
        elif restart == 'No':
            # Start time
            startT = ', '.join(  # noqa: N806
                hydroutil.extract_element_from_json(
                    data, ['Events', 'StartTime']
                )
            )
            cdicttext = cdicttext + 'startFrom \t startTime;\n\n'
            cdicttext = cdicttext + 'startTime \t' + startT + ';\n\n'

        # End time
        endT = ', '.join(  # noqa: N806
            hydroutil.extract_element_from_json(data, ['Events', 'EndTime'])
        )
        cdicttext = cdicttext + 'stopAt \t endTime;\n\n'
        cdicttext = cdicttext + 'endTime \t' + endT + ';\n\n'

        # Time interval
        deltaT = ', '.join(  # noqa: N806
            hydroutil.extract_element_from_json(
                data, ['Events', 'TimeInterval']
            )
        )
        cdicttext = cdicttext + 'deltaT \t' + deltaT + ';\n\n'

        # Write control
        cdicttext = cdicttext + 'writeControl \t adjustableRunTime;\n\n'

        # Write interval
        writeT = ', '.join(  # noqa: N806
            hydroutil.extract_element_from_json(
                data, ['Events', 'WriteInterval']
            )
        )
        cdicttext = cdicttext + 'writeInterval \t' + writeT + ';\n\n'

        # All others
        cdicttext = cdicttext + 'purgeWrite \t 0;\n\n'
        cdicttext = cdicttext + 'writeFormat \t ascii;\n\n'
        cdicttext = cdicttext + 'writePrecision \t 6;\n\n'
        cdicttext = cdicttext + 'writeCompression \t uncompressed;\n\n'
        cdicttext = cdicttext + 'timeFormat \t general;\n\n'
        cdicttext = cdicttext + 'timePrecision \t 6;\n\n'
        cdicttext = cdicttext + 'runTimeModifiable \t yes;\n\n'
        cdicttext = cdicttext + 'adjustTimeStep \t yes;\n\n'
        cdicttext = cdicttext + 'maxCo \t 1.0;\n\n'
        cdicttext = cdicttext + 'maxAlphaCo \t 1.0;\n\n'
        cdicttext = cdicttext + 'maxDeltaT \t 1;\n\n'

        # Point data from file
        pprocessfile = ', '.join(
            hydroutil.extract_element_from_json(
                data, ['Events', 'PProcessFile']
            )
        )
        pprocesspath = os.path.join(path, pprocessfile)  # noqa: PTH118
        pp_data = np.genfromtxt(pprocesspath, delimiter=',')
        num_points = np.shape(pp_data)[0]
        ptext = '\t\t\t\t(\n'
        for ii in range(num_points):
            ptext = (
                ptext
                + '\t\t\t\t\t('
                + str(pp_data[ii, 0])
                + '\t'
                + str(pp_data[ii, 1])
                + '\t'
                + str(pp_data[ii, 2])
                + ')\n'
            )
        ptext = ptext + '\t\t\t\t);\n'

        # Fields required
        value = 0
        pprocessV = hydroutil.extract_element_from_json(  # noqa: N806
            data, ['Events', 'PPVelocity']
        )
        if pprocessV != [None]:
            pprocessV = ', '.join(  # noqa: N806
                hydroutil.extract_element_from_json(
                    data, ['Events', 'PPVelocity']
                )
            )
            if pprocessV == 'Yes':
                value += 1
        pprocessP = hydroutil.extract_element_from_json(  # noqa: N806
            data, ['Events', 'PPPressure']
        )
        if pprocessP != [None]:
            pprocessP = ', '.join(  # noqa: N806
                hydroutil.extract_element_from_json(
                    data, ['Events', 'PPPressure']
                )
            )
            if pprocessP == 'Yes':
                value += 2
        if value == 1:
            fieldtext = '(U)'
        elif value == 2:  # noqa: PLR2004
            fieldtext = '(p_rgh)'
        else:
            fieldtext = '(U p_rgh)'

        # Get the library data
        cdicttext = cdicttext + 'function\n{\n\tlinesample\n\t{\n'
        cdicttext = cdicttext + '\t\ttype\tsets;\n'
        cdicttext = cdicttext + '\t\tfunctionObjectLibs\t("libsampling.so");\n'
        cdicttext = cdicttext + '\t\twriteControl\ttimeStep;\n'
        cdicttext = cdicttext + '\t\toutputInterval\t1;\n'
        cdicttext = cdicttext + '\t\tinterpolationScheme\tcellPoint;\n'
        cdicttext = cdicttext + '\t\tsetFormat\traw;\n\n'
        cdicttext = cdicttext + '\t\tsets\n\t\t(\n'
        cdicttext = cdicttext + '\t\t\tdata\n\t\t\t{\n'
        cdicttext = cdicttext + '\t\t\t\ttype\tpoints;\n'
        cdicttext = cdicttext + '\t\t\t\tpoints\n'
        cdicttext = cdicttext + ptext
        cdicttext = cdicttext + '\t\t\t\tordered\tyes;\n'
        cdicttext = cdicttext + '\t\t\t\taxis\tx;\n'
        cdicttext = cdicttext + '\t\t\t}\n\t\t);\n'
        cdicttext = cdicttext + '\t\tfields\t' + fieldtext + ';\n'
        cdicttext = cdicttext + '\t}\n}'

        return cdicttext  # noqa: DOC201, RET504, RUF100

    #############################################################
    def scripts(self, data, path):  # noqa: ARG002
        """Creates the necessary postprocessing in scripts

        Arguments:
        ---------
                data: all the JSON data

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()

        pprocess = hydroutil.extract_element_from_json(
            data, ['Events', 'Postprocessing']
        )
        if pprocess == [None]:
            return 0  # noqa: DOC201, RUF100
        else:  # noqa: RET505
            pprocess = ', '.join(
                hydroutil.extract_element_from_json(
                    data, ['Events', 'Postprocessing']
                )
            )
            if pprocess == 'No':
                caseruntext = 'echo no postprocessing for EVT\n'
            elif pprocess == 'Yes':
                caseruntext = 'echo postprocessing for EVT\n'
                # Reconstruct case
                caseruntext = (
                    caseruntext + 'reconstructPar > reconstruct.log \n'
                )
                # Move new controlDict
                cdictpppath = os.path.join('system', 'controlDict')  # noqa: PTH118
                caseruntext = caseruntext + 'cp cdictpp ' + cdictpppath + '\n'
                # Move the wavemakerfile (if exists)
                if os.path.exists(  # noqa: PTH110
                    os.path.join('constant', 'wavemakerMovement.txt')  # noqa: PTH118
                ):
                    caseruntext = caseruntext + 'mkdir extras\n'
                    wavepath = os.path.join(  # noqa: PTH118
                        'constant', 'wavemakerMovement.txt'
                    )
                    wavepathnew = os.path.join(  # noqa: PTH118
                        'extras', 'wavemakerMovement.txt'
                    )
                    caseruntext = (
                        caseruntext
                        + 'mv '
                        + wavepath
                        + ' '
                        + wavepathnew
                        + '\n'
                    )
                # Copy sample file
                caseruntext = (
                    caseruntext
                    + 'cp sample '
                    + os.path.join('system', 'sample')  # noqa: PTH118
                    + '\n'
                )
                # Start the postprocessing
                caseruntext = caseruntext + 'postProcess -func sample \n\n'

        # Write to caserun file
        scriptfile = open('caserun.sh', 'a')  # noqa: SIM115, PTH123
        scriptfile.write(caseruntext)
        scriptfile.close()  # noqa: RET503

    #############################################################
    def pprocesscheck(self, data, path):
        """Checks for material properties for openfoam7

        Arguments:
        ---------
                data: all the JSON data

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()

        # Find if pprocess is required
        pprocess = ', '.join(
            hydroutil.extract_element_from_json(
                data, ['Events', 'Postprocessing']
            )
        )

        if pprocess == 'No':
            return 0  # noqa: DOC201, RUF100
        else:  # noqa: RET505
            pprocessV = ', '.join(  # noqa: N806
                hydroutil.extract_element_from_json(
                    data, ['Events', 'PPVelocity']
                )
            )
            pprocessP = ', '.join(  # noqa: N806
                hydroutil.extract_element_from_json(
                    data, ['Events', 'PPPressure']
                )
            )
            if pprocessV == 'Yes' or pprocessP == 'Yes':
                pprocessfile = hydroutil.extract_element_from_json(
                    data, ['Events', 'PProcessFile']
                )
                if pprocessfile == [None]:
                    return -1
                else:  # noqa: RET505
                    pprocessfile = ', '.join(
                        hydroutil.extract_element_from_json(
                            data, ['Events', 'PProcessFile']
                        )
                    )
                    if not os.path.exists(os.path.join(path, pprocessfile)):  # noqa: PTH110, PTH118
                        return -1
            else:
                return 0

        # Return 0 if all is right
        return 1
