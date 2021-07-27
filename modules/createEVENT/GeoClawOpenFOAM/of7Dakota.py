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
class of7Dakota():
	"""
	This class includes the methods related to
	dakota for openfoam7.

	Methods
	--------
		scripts: Generate relevant scripts
	"""

	#############################################################
	def dakotascripts(self,args):
		'''
		Create the scripts for caserun.sh

		Arguments
		-----------
			data: all the JSON data
			path: Path where dakota.json file is located
		'''

		caseruntext = 'echo Starting Dakota preparation...\n'
		caseruntext = caseruntext + 'python3 $HYDROBRAIN/GetOpenFOAMEvent.py -b '+args.b+'\n'
		caseruntext = caseruntext + 'cp -f EVENT.json ${inputDirectory}/EVENT.json\n'
		caseruntext = caseruntext + 'cp -f EVENT.json ${inputDirectory}/evt.j\n\n'

		# Load necessary modules
		caseruntext = caseruntext + 'echo Loading necessary modules for Dakota...\n'
		caseruntext = caseruntext + 'module load intel/18.0.2  impi/18.0.2 dakota/6.8.0 python3\n\n'

		# Initialize file names and scripts
		caseruntext = caseruntext + 'echo Initializing file names and scripts...\n'
		caseruntext = caseruntext + 'echo "inputScript is ${inputFile}"\n'
		caseruntext = caseruntext + 'cd ${inputDirectory}\n'
		caseruntext = caseruntext + 'chmod \'a+x\' workflow_driver\n'
		caseruntext = caseruntext + 'cp workflow_driver ../\n'
		caseruntext = caseruntext + 'cd ..\n\n'

		# Run Dakota
		caseruntext = caseruntext + 'echo Running dakota...\n'
		caseruntext = caseruntext + 'ibrun dakota -in dakota.in -out dakota.out -err dakota.err\n\n'

		# Cleanup
		caseruntext = caseruntext + 'if [ -d ./workdir.1 ]\n'
		caseruntext = caseruntext + 'then\n'
		caseruntext = caseruntext + '\tmkdir ./workdir\n'
		caseruntext = caseruntext + '\tmv workdir.* workdir\n'
		caseruntext = caseruntext + '\ttar zcBf workdir.tar.gz workdir\n'
		caseruntext = caseruntext + '\trm -fr workdir\n'
		caseruntext = caseruntext + 'fi\n'
		caseruntext = caseruntext + 'cp templatedir/dakota.json ./\n'
		caseruntext = caseruntext + 'tar zcBf templatedir.tar.gz templatedir\n'
		caseruntext = caseruntext + 'rm -fr templatedir\n\n'
		caseruntext = caseruntext + 'cd ..\n\n'
		caseruntext = caseruntext + 'if [ ! $? ]; then\n'
		caseruntext = caseruntext + '\techo "dakota exited with an error status. $?" >&2\n'
		caseruntext = caseruntext + '\t${AGAVE_JOB_CALLBACK_FAILURE}\n'
		caseruntext = caseruntext + '\texit\n'
		caseruntext = caseruntext + 'fi\n\n'

		# Write to caserun file
		scriptfile = open('caserun.sh',"a")
		scriptfile.write(caseruntext)
		scriptfile.close()

	#############################################################
	def cleaning(self,args,path):
		'''
		Create the scripts for cleaning

		Arguments
		-----------
			args: all the arguments
		'''

		print('No OF cleaning')

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