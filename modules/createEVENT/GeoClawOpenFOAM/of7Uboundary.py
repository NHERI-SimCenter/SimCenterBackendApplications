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
class of7Uboundary():
	"""
	This class includes the methods related to
	velocity boundary conditions for openfoam7.

	Methods
	--------
		Utext: Get s all the text for the U-file
	"""

	#############################################################
	def Utext(self,data,fipath,patches):
		'''
		Creates the necessary folders for openfoam7

		Arguments
		-----------
			data: all the JSON data
			patches: List of boundary patches
			fipath: Path where the dakota.json file exists
		'''

		# Create a utilities object
		hydroutil = hydroUtils()

		# Get the header text for the U-file
		utext = self.Uheader()

		# Start the outside 
		utext = utext + "boundaryField\n{\n"

		# Loop over all patches
		for patchname in patches:
			utext = utext + "\t" + patchname + "\n"
			patch = hydroutil.extract_element_from_json(data, ["Events","VelocityType_" + patchname])
			if patch == [None]:
				Utype = -1
			else:
				Utype = ', '.join(hydroutil.extract_element_from_json(data, ["Events","VelocityType_" + patchname]))
			utext = utext + self.Upatchtext(data,Utype,patchname,fipath)

		# Close the outside
		utext = utext + "}\n\n"
		
		# Return the text for velocity BC
		return utext
	
	#############################################################
	def Uheader(self):
		'''
		Creates the text for the header

		Variable
		-----------
			header: Header for the U-file
		'''

		header = """/*--------------------------*- NHERI SimCenter -*----------------------------*\ 
|	   | H |
|	   | Y | HydroUQ: Water-based Natural Hazards Modeling Application
|======| D | Website: simcenter.designsafe-ci.org/research-tools/hydro-uq
|	   | R | Version: 1.00
|	   | O |
\*---------------------------------------------------------------------------*/ 
FoamFile
{\n\tversion\t2.0;\n\tformat\tascii;\n\tclass\tvolVectorField;\n\tlocation\t"0";\n\tobject\tU;\n}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"""

		header = header + "dimensions\t[0 1 -1 0 0 0 0];\n\n"
		header = header + "internalField\tuniform (0 0 0);\n\n"
		
		# Return the header for U file
		return header

	#############################################################
	def Upatchtext(self,data,Utype,patchname,fipath):
		'''
		Creates the text the velocity boundary condition

		Arguments
		-----------
			data: All the json data
			Utype: Type of velocity b.c
			patchname: Name of the patch
			fipath: Path to where dakota.json file exists

		Variable
		-----------
			Utext: Text for the particular patch
		'''

		# Create a utilities object
		hydroutil = hydroUtils()

		# Inlet types
		# For each type, get the text
		if int(Utype) == 101:
			# SW solutions (1)
			Utext = "\t{\n\t\t"
			Utext = Utext + "type\ttimeVaryingMappedFixedValue;\n\t\t"
			Utext = Utext + "offset\t(0 0 0);\n\t\t"
			Utext = Utext + "setAverage\toff;\n"
			Utext = Utext + "\t}\n"

		elif int(Utype) == 102:
			# Inlet: constant velocity
			# Get the velocity values
			velo = hydroutil.extract_element_from_json(data, ["Events","Velocity_"+patchname])
			if velo == [None]:
				vx = 0.0
				vy = 0.0
				vz = 0.0
			else:
				velvals = ', '.join(hydroutil.extract_element_from_json(data, ["Events","Velocity_"+patchname]))
				velvals = velvals.replace(',',' ')
				vels = [float(vi) for vi in velvals.split()]
				vx = vels[0]
				vy = vels[1]
				vz = vels[2]

			# Get the text
			Utext = "\t{\n\t\t"
			Utext = Utext + "type\tfixedValue;\n\t\t"
			Utext = Utext + "value\t(" + str(vx) +"\t"+ str(vy) +"\t"+ str(vz) +");\n\t}\n"

		elif int(Utype) == 103:
			# Inlet Moving wall (OSU flume)
			Utext = "\t{\n\t\t"
			Utext = Utext + "type\tmovingWallVelocity;\n\t\t"
			Utext = Utext + "value\tuniform (0 0 0);\n\t}\n"
			# Create the files required
			# Moving wall file
			self.OSUwavemakerText(data,fipath)
			# Wavemakermovement dictionary
			self.of7wavemakerdict(fipath)
			# Dynamic mesh dictionary
			self.of7dynamicMeshdict(fipath)

		elif int(Utype) == 104:
			# Inlet Moving wall (Gen flume)
			Utext = "\t{\n\t\t"
			Utext = Utext + "type\tmovingWallVelocity;\n\t\t"
			Utext = Utext + "value\tuniform (0 0 0);\n\t}\n"
			# Create the files required
			# Moving wall file
			# self.GenwavemakerText(data,fipath)
			# Wavemakermovement dictionary
			self.of7wavemakerdict(fipath)
			# Dynamic mesh dictionary
			self.of7dynamicMeshdict(fipath)

		elif int(Utype) == 201:
			# Outlet zero gradient
			Utext = "\t{\n\t\t"
			Utext = Utext + "type\tzeroGradient;\n\t}\n"

		elif int(Utype) == 202:
			# Outlet: inletOutlet
			# Get the velocity values
			velo = hydroutil.extract_element_from_json(data, ["Events","Velocity_"+patchname])
			if velo == [None]:
				vx = 0.0
				vy = 0.0
				vz = 0.0
			else:
				velvals = ', '.join(hydroutil.extract_element_from_json(data, ["Events","Velocity_"+patchname]))
				velvals = velvals.replace(',',' ')
				vels = [float(vi) for vi in velvals.split()]
				vx = vels[0]
				vy = vels[1]
				vz = vels[2]

			# Get the text
			Utext = "\t{\n\t\t"
			Utext = Utext + "type\tinletOutlet;\n\t\t"
			Utext = Utext + "inletValue\tuniform (" + str(vx) +"\t"+ str(vy) +"\t"+ str(vz) +");\n\t\t"
			Utext = Utext + "value\tuniform (" + str(vx) +"\t"+ str(vy) +"\t"+ str(vz) +");\n"
			Utext = Utext + "\t}\n"
			
		elif int(Utype) == 301:
			# Wall: noSlip
			Utext = "\t{\n\t\t"
			Utext = Utext + "type\tnoSlip;\n\t}\n"

		else:
			# Default: Empty
			Utext = "\t{\n\t\t"
			Utext = Utext + "type\tempty;\n\t}\n"
		
		# Return the header for U file
		return Utext

	#############################################################
	def Uchecks(self,data,fipath,patches):
		'''
		Creates the data files required for the OSU moving wall

		Arguments
		-----------
			data: All the data from JSON file
			fipath: Path to the dakota.json file location
			patches: List of patches
		'''

		# Create a utilities object
		hydroutil = hydroUtils()

		# Number of moving walls
		numMovWall = 0

		# Loop over all patches
		for patchname in patches:
			# Get the type of velocity bc
			patch = hydroutil.extract_element_from_json(data, ["Events","VelocityType_" + patchname])
			if patch == [None]:
				Utype = -1
			else:
				Utype = ', '.join(hydroutil.extract_element_from_json(data, ["Events","VelocityType_" + patchname]))
			
			# Checks for different U-types
			if int(Utype) == 103:
				# Checking for multiple moving walls
				numMovWall += 1
				if numMovWall > 1:
					return -1
				
				# Check for existing moving wall files
				dispfilename = hydroutil.extract_element_from_json(data, ["Events","OSUMovingWallDisp_" + patchname])
				if dispfilename == [None]:
					return -1
				else:
					dispfilename = ', '.join(hydroutil.extract_element_from_json(data, ["Events","OSUMovingWallDisp_" + patchname]))
				pathF = os.path.join(fipath,dispfilename)
				if(os.path.exists(pathF)):
					print('Displacement file exists')
				else: 
					return -1

			elif int(Utype) == 104:
				# Checking for multiple moving walls
				numMovWall += 1
				if numMovWall > 1:
					return -1
				
				# Check for existing moving wall files
				dispfilename = hydroutil.extract_element_from_json(data, ["Events","MovingWallDisp_" + patchname])
				pathF = os.path.join(fipath,dispfilename)
				if(os.path.exists(pathF)):
					print('Displacement file exists')
				else: 
					return -1

		# If all checks passes
		return 0

	#############################################################
	def of7wavemakerdict(self,fipath):
		'''
		Creates the wavemaker dictionary for the moving wall

		Arguments
		-----------
			fipath: Path to the dakota.json file location
		'''

		# Create a utilities object
		hydroutil = hydroUtils()

		# Get the file ID
		filepath = os.path.join(fipath,"constant","wavemakerMovementDict")
		fileID = open(filepath, "w")
		# Header
		header = hydroutil.of7header("dictionary","constant","wavemakerMovementDict")
		fileID.write(header)
		# Other data
		fileID.write('\nreread\tfalse;\n\n')
		fileID.write('#include\t"wavemakerMovement.txt"\n')
		# Close the file
		fileID.close()

	#############################################################
	def of7dynamicMeshdict(self,fipath):
		'''
		Creates the dynamic mesh dictionary for the moving wall

		Arguments
		-----------
			fipath: Path to the dakota.json file location
		'''

		# Create a utilities object
		hydroutil = hydroUtils()

		# Get the file ID
		filepath = os.path.join(fipath,"constant","dynamicMeshDict")
		fileID = open(filepath, "w")
		# Header
		header = hydroutil.of7header("dictionary","constant","dynamicMeshDict")
		fileID.write(header)
		# Other data
		fileID.write('\ndynamicFvMesh\tdynamicMotionSolverFvMesh;\n\n')
		fileID.write('motionSolverLibs\t("libfvMotionSolvers.so");\n\n')
		fileID.write('solver\tdisplacementLaplacian;\n\n')
		fileID.write('displacementLaplacianCoeffs\n{\n\tdiffusivity uniform;\n}\n');
		# Close the file
		fileID.close()

	#############################################################
	def OSUwavemakerText(self,fipath):
		'''
		Creates the wavemaker text file for the OSU moving wall

		Arguments
		-----------
			fipath: Path to the dakota.json file location
		'''

		# Create a utilities object
		hydroutil = hydroUtils()