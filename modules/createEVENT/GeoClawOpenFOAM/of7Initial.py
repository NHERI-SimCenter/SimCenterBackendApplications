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
class of7Initial():
	"""
	This class includes the methods related to
	initial conditions for openfoam7.

	Methods
	--------
		alphatext: Get all the text for the setFieldsDict
	"""

	#############################################################
	def alphatext(self,data,fipath):
		'''
		Creates the necessary files for alpha - setFields for openfoam7

		Arguments
		-----------
			data: all the JSON data
		'''

		# Create a utilities object
		hydroutil = hydroUtils()

		# Get the simulation type
		simtype = ', '.join(hydroutil.extract_element_from_json(data, ["Events","SimulationType"]))

		# Get the header text for the U-file
		alphatext = self.alphaheader()

		# Read the values
		if int(simtype) == 1:
			fname = "SWAlpha.txt"
			swalphafile = os.path.join(fipath,fname)
			with open(swalphafile) as f:
				gloalpha, localalpha, x1,y1,z1,x2,y2,z2 = [float(x) for x in next(f).split(',')]
			
			alphatext = alphatext + 'defaultFieldValues\n(\n\tvolScalarFieldValue\talpha.water\t' + str(gloalpha) + '\n);\n\n'

			alphatext = alphatext + 'regions\n(\n'
			alphatext = alphatext + '\tboxToCell\n\t{\n\t\t'
			alphatext = alphatext + 'box\t(' + str(x1) + '\t' + str(y1) + '\t' + str(z1) + ')\t(' + str(x2) + '\t' + str(y2) + '\t' + str(z2) +');\n\n\t\t'
			alphatext = alphatext + 'fieldValues\n\t\t(\n\t\t\tvolScalarFieldValue\talpha.water\t' + str(localalpha) + '\n\t\t);\n\t}\n\n'

		else:
			gloalpha = ', '.join(hydroutil.extract_element_from_json(data, ["Events","InitialAlphaGlobal"]))

			numregs = ', '.join(hydroutil.extract_element_from_json(data, ["Events","NumAlphaRegion"]))

			alphatext = alphatext + 'defaultFieldValues\n(\n\tvolScalarFieldValue\talpha.water\t' + str(gloalpha) + '\n);\n\n'

			alphatext = alphatext + 'regions\n(\n'

			# Check for each alpha region
			for ii in range(int(numregs)):

				# Get the region
				# We dont check if region is inside the geometry
				# Should be added later on
				region = ', '.join(hydroutil.extract_element_from_json(data, ["Events","InitialAlphaRegion"+str(ii)]))
				regsegs = region.replace(',', ' ')
				# Convert the regions to list of floats
				nums = [float(n) for n in regsegs.split()]
				
				alphatext = alphatext + '\tboxToCell\n\t{\n\t\t'
				alphatext = alphatext + 'box\t(' + str(nums[0]) + '\t' + str(nums[1]) + '\t' + str(nums[2]) + ')\t(' + str(nums[3]) + '\t' + str(nums[4]) + '\t' + str(nums[5]) +');\n\n\t\t'
				alphatext = alphatext + 'fieldValues\n\t\t(\n\t\t\tvolScalarFieldValue\talpha.water\t' + str(nums[6]) + '\n\t\t);\n\t}\n\n'

		alphatext = alphatext + '\n);'

		return alphatext

	#############################################################
	def alphaheader(self):
		'''
		Creates the text for the header

		Variable
		-----------
			header: Header for the setFields-file
		'''

		header = """/*--------------------------*- NHERI SimCenter -*----------------------------*\ 
|	   | H |
|	   | Y | HydroUQ: Water-based Natural Hazards Modeling Application
|======| D | Website: simcenter.designsafe-ci.org/research-tools/hydro-uq
|	   | R | Version: 1.00
|	   | O |
\*---------------------------------------------------------------------------*/ 
FoamFile
{\n\tversion\t2.0;\n\tformat\tascii;\n\tclass\tdictionary;\n\tlocation\t"system";\n\tobject\tsetFieldsDict;\n}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"""

		# Return the header for U file
		return header

	#############################################################
	def alphacheck(self,data,fipath):
		'''
		Checks for initial conditions for openfoam7

		Arguments
		-----------
			data: all the JSON data
		'''

		# Create a utilities object
		hydroutil = hydroUtils()

		# Get the simulation type
		simtype = ', '.join(hydroutil.extract_element_from_json(data, ["Events","SimulationType"]))

		# For SW-CFD coupling
		if simtype == 1:

			# Check for the file exists
			fname = "SWAlpha.txt"
			swalphafile = os.path.join(fipath,fname)
			if not os.path.exists(swalphafile):
				return -1
		
		# For all types other than the shallow water
		else:

			# Check global alpha value
			alphaglobal = hydroutil.extract_element_from_json(data, ["Events","InitialAlphaGlobal"])
			if alphaglobal == [None]:
				return -1

			# Check number of alpha region
			numreg = hydroutil.extract_element_from_json(data, ["Events","NumAlphaRegion"])
			if numreg == [None]:
				return -1
			else:
				numreg = ', '.join(hydroutil.extract_element_from_json(data, ["Events","NumAlphaRegion"]))
				if int(numreg) < 1:
					return -1
			
			# Check for each alpha region
			for ii in range(int(numreg)):

				# Get the region
				# We dont check if region is inside the geometry
				# Should be added later on
				region = hydroutil.extract_element_from_json(data, ["Events","InitialAlphaRegion"+str(ii)])
				if region == [None]:
					return -1
				else:
					region = ', '.join(hydroutil.extract_element_from_json(data, ["Events","InitialAlphaRegion"+str(ii)]))
					regsegs = region.replace(',', ' ')
					# Convert the regions to list of floats
					nums = [float(n) for n in regsegs.split()]
					# Check if 6 coordinates + 1 alpha number
					if len(nums) != 7:
						return -1

		

		# Return 0 if all is right
		return 0