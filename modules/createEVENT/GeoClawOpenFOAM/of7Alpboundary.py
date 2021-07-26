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
class of7Alpboundary():
	"""
	This class includes the methods related to
	alpha boundary conditions for openfoam7.

	Methods
	--------
		Alptext: Get all the text for the p_rgh-file
	"""

	#############################################################
	def Alptext(self,data,patches):
		'''
		Creates the necessary text for pressure bc for openfoam7

		Arguments
		-----------
			data: all the JSON data
		'''

		# Create a utilities object
		hydroutil = hydroUtils()

		# Get the header text for the U-file
		Alptext = self.Alpheader()

		# Start the outside 
		Alptext = Alptext + "boundaryField\n{\n"

		# Loop over all patches
		for patchname in patches:
			Alptext = Alptext + "\t" + patchname + "\n"
			patch = hydroutil.extract_element_from_json(data, ["Events","PressureType_" + patchname])
			if patch == [None]:
				Alptype = -1
			else:
				Alptype = 0
			Alptext = Alptext + self.Alppatchtext(Alptype,patchname)
			print(patchname)

		# Check for building and other building
		Alptext = Alptext + '\tBuilding\n'
		Alptext = Alptext + self.Alppatchtext(0,'Building')
		Alptext = Alptext + '\tOtherBuilding\n'
		Alptext = Alptext + self.Alppatchtext(0,'OtherBuilding')

		# Close the outside
		Alptext = Alptext + "}\n\n"
		
		# Return the text for velocity BC
		return Alptext
	
	#############################################################
	def Alpheader(self):
		'''
		Creates the text for the header for pressure file

		Variable
		-----------
			header: Header for the p_rgh-file
		'''

		header = """/*--------------------------*- NHERI SimCenter -*----------------------------*\ 
|	   | H |
|	   | Y | HydroUQ: Water-based Natural Hazards Modeling Application
|======| D | Website: simcenter.designsafe-ci.org/research-tools/hydro-uq
|	   | R | Version: 1.00
|	   | O |
\*---------------------------------------------------------------------------*/ 
FoamFile
{\n\tversion\t2.0;\n\tformat\tascii;\n\tclass\tvolScalarField;\n\tlocation\t"0";\n\tobject\talpha.water;\n}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"""

		header = header + "dimensions\t[0 0 0 0 0 0 0];\n\n"
		header = header + "internalField\tuniform\t0;\n\n"
		
		# Return the header for U file
		return header

	#############################################################
	def Alppatchtext(self,Alptype,patchname):
		'''
		Creates the text the pressure boundary condition

		Arguments
		-----------
			patchname: Name of the patch

		Variable
		-----------
			Alptext: Text for the particular patch
		'''

		if patchname == 'Top':
			Alptext = "\t{\n\t\t"
			Alptext = Alptext + "type\tinletOutlet;\n\t\t"
			Alptext = Alptext + "inletValue\tuniform 0;\n\t\t"
			Alptext = Alptext + "value\tuniform 0;\n\t}\n"

		else:
			Alptext = "\t{\n\t\t"
			Alptext = Alptext + "type\tzeroGradient;\n\t}\n"

		
		# Return the header for U file
		return Alptext