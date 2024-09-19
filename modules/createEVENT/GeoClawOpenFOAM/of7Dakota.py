#  # noqa: INP001, EXE002
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

# Other custom modules


####################################################################
# OpenFOAM7 solver class
####################################################################
class of7Dakota:
    """This class includes the methods related to
    dakota for openfoam7.

    Methods
    -------
            scripts: Generate relevant scripts

    """  # noqa: D205, D404

    #############################################################
    def dakotascripts(self, args):
        """Create the scripts for caserun.sh

        Arguments:
        ---------
                data: all the JSON data
                path: Path where dakota.json file is located

        """  # noqa: D400
        caseruntext = 'echo Starting Dakota preparation...\n'
        caseruntext = (
            caseruntext
            + 'python3 $HYDROBRAIN/GetOpenFOAMEvent.py -b '
            + args.b
            + '\n'
        )

        # Openfoam cleanup
        caseruntext = caseruntext + 'rm -fr processor*\n'
        caseruntext = caseruntext + 'rm -fr 0\n'
        caseruntext = caseruntext + 'mkdir EVTfiles\n'
        caseruntext = (
            caseruntext
            + 'mv 0.org '
            + os.path.join('EVTfiles', '0.org')  # noqa: PTH118
            + '\n'
        )
        caseruntext = (
            caseruntext
            + 'mv constant '
            + os.path.join('EVTfiles', 'constant')  # noqa: PTH118
            + '\n'
        )
        caseruntext = (
            caseruntext
            + 'mv system '
            + os.path.join('EVTfiles', 'system')  # noqa: PTH118
            + '\n'
        )
        caseruntext = (
            caseruntext
            + 'mv postProcessing '
            + os.path.join('EVTfiles', 'postProcessing')  # noqa: PTH118
            + '\n'
        )
        caseruntext = caseruntext + 'mv *.log EVTfiles\n'
        caseruntext = caseruntext + 'mv *.stl EVTfiles\n'
        caseruntext = caseruntext + 'mv *.sh EVTfiles\n'
        caseruntext = caseruntext + 'mv *.txt EVTfiles\n'
        caseruntext = caseruntext + 'mv cdict* EVTfiles\n'
        caseruntext = caseruntext + 'tar zcBf EVTfiles.tar.gz EVTfiles\n'
        caseruntext = caseruntext + 'rm -fr EVTfiles\n\n'

        # Write to caserun file
        scriptfile = open('caserun.sh', 'a')  # noqa: SIM115, PTH123
        scriptfile.write(caseruntext)
        scriptfile.close()

    #############################################################
    def cleaning(self, args, path):  # noqa: ARG002
        """Create the scripts for cleaning

        Arguments:
        ---------
                args: all the arguments

        """  # noqa: D400
        print('No OF cleaning')  # noqa: T201


# # tar -c -f trial.tar $(readlink -e a b c d)
# # tar -xvf trial.tar

# 		caseruntext = 'echo Starting cleaning...\n'

# 		# Move all log files
# 		caseruntext = caseruntext + 'mkdir ./logfiles\n'
# 		caseruntext = caseruntext + 'find . -name "*.log" -exec mv "{}" ./logfiles \;' + '\n'

# 		# Tar all files and folder
# 		caseruntext = caseruntext + 'tar -c -f Files.tar $(cdictpp cdictforce FlumeData.txt sample temp_geometry.txt translate.sh caserun.sh 0 0.org constant system postProcessing logfiles ' + path + ')\n'

# 		# Remove all folders
# 		caseruntext = caseruntext + 'rm -rf ./*/' + '\n'

# 		# Untar
# 		caseruntext = caseruntext + 'tar -xvf Files.tar\n'

# 		# Tar all other EVT files
# 		caseruntext = caseruntext + 'tar -c -f Files.tar $(cdictpp cdictforce FlumeData.txt sample temp_geometry.txt translate.sh caserun.sh 0 0.org constant system postProcessing logfiles)\n'

# 		# Write to caserun file
# 		scriptfile = open('caserun.sh',"a")
# 		scriptfile.write(caseruntext)
# 		scriptfile.close()
