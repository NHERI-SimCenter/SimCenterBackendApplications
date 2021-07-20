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
import shutil

# Other custom modules
from hydroUtils import hydroUtils
from of7Materials import of7Materials
from of7Initial import of7Initial
from of7Uboundary import of7Uboundary
from of7Prboundary import of7Prboundary
from of7PtDboundary import of7PtDboundary
from of7Decomp import of7Decomp
from of7Solve import of7Solve
from of7Others import of7Others

####################################################################
# OpenFOAM7 solver class
####################################################################
class openfoam7():
	"""
	This class includes the methods related to openfoam7.

	Methods
	--------
		extract: 
	"""

	#############################################################
	def createfolder(self,data,path):
		'''
		Creates the necessary folders for openfoam7

		Arguments
		-----------
			data: all the JSON data
			path: Path where the new folder needs to be created
		'''

		# Create directories for openfoam dictionaries
		# Access: Only owner can read and write
		access_rights = 0o700

		# Create 0-directory
		pathF = os.path.join(path,"0.org")
		if(os.path.exists(pathF)):
			shutil.rmtree(pathF)
			os.mkdir(pathF,access_rights)
		else:
			os.mkdir(pathF,access_rights)
	
		#Create constant-directory
		pathF = os.path.join(path,"constant")
		if(os.path.exists(pathF)):
			shutil.rmtree(pathF)
			os.mkdir(pathF,access_rights)
		else:
			os.mkdir(pathF,access_rights)

		# Create the triSurface directory
		pathF = os.path.join(path,"constant/triSurface")
		if(os.path.exists(pathF)):
			shutil.rmtree(pathF)
			os.mkdir(pathF,access_rights)
		else:
			os.mkdir(pathF,access_rights)

		#Create system-directory
		pathF = os.path.join(path,"system")
		if(os.path.exists(pathF)):
			shutil.rmtree(pathF)
			os.mkdir(pathF,access_rights)
		else:
			os.mkdir(pathF,access_rights) 
		
		# Return completion flag
		return 0

	#############################################################
	def creategeometry(self,data,path):
		'''
		Creates the necessary folders for openfoam7

		Arguments
		-----------
			data: all the JSON data
			path: Path where the geometry files (STL) needs to be created
		'''

		print("Geometry and building files are created here")

		# Create directories for openfoam dictionaries
		# Access: Only owner can read and write

		return 0

	#############################################################
	def createmesh(self,data,path):
		'''
		Creates the mesh dictionaries for openfoam7

		Arguments
		-----------
			data: all the JSON data
			path: Path where the geometry files (STL) needs to be created
		'''

		print("Mesher files are created here")
		# Create directories for openfoam dictionaries
		# Access: Only owner can read and write

		return 0

	#############################################################
	def materials(self,data,path):
		'''
		Creates the material files for openfoam7

		Arguments
		-----------
			data: all the JSON data
			path: Path where the geometry files (STL) needs to be created
		'''

		# Create the transportProperties file
		Materials = of7Materials()
		matcode = Materials.matcheck(data)
		if matcode == -1:
			return -1
		else:
			mattext = Materials.mattext(data)
			fname = "transportProperties"
			filepath = os.path.join(path, "constant", fname)
			matfile = open(filepath, "w")
			matfile.write(mattext)
			matfile.close()

		return 0

	#############################################################
	def initial(self,data,path):
		'''
		Creates the initial condition files for openfoam7

		Arguments
		-----------
			data: all the JSON data
			path: Path where the geometry files dakota.json lies
		'''

		# Create the setFields file
		Inicond = of7Initial()
		initcode = Inicond.alphacheck(data,path)
		if initcode == -1:
			return -1
		else:
			alphatext = Inicond.alphatext(data,path)
			fname = "setFieldsDict"
			filepath = os.path.join(path, "system", fname)
			alphafile = open(filepath, "w")
			alphafile.write(alphatext)
			alphafile.close()

		return 0

	#############################################################
	def boundary(self,data,path):
		'''
		Creates the bc condition files for openfoam7

		Arguments
		-----------
			data: all the JSON data
			path: Path where the geometry files (STL) needs to be created
		'''

		# Initialize the patches
		patches = ['Entry', 'Exit', 'Top', 'Bottom', 'Right', 'Left']

		# Create object for velocity boundary condition
		# Get the text for the velocity boundary
		# Write the U-file in 0.org
		Uboundary = of7Uboundary()
		utext = Uboundary.Utext(data,path,patches)
		# Check for boundary conditions here
		ecode = Uboundary.Uchecks(data,path,patches)
		if ecode == -1:
			return -1
		else:
			# Write the U-file if no errors
			# Path to the file
			fname = "U"
			filepath = os.path.join(path, "0.org", fname)
			Ufile = open(filepath, "w")
			Ufile.write(utext)
			Ufile.close()

		# Create object for pressure boundary condition
		# Get the text for the pressure boundary
		# Write the p_rgh-file in 0.org
		Prboundary = of7Prboundary()
		prtext = Prboundary.Prtext(data,patches)
		fname = "p_rgh"
		filepath = os.path.join(path, "0.org", fname)
		prfile = open(filepath, "w")
		prfile.write(prtext)
		prfile.close()

		# Loop over all the velocity type to see if any 
		# has a moving wall. If so initialize the 
		# pointDisplacement file
		PtDboundary = of7PtDboundary()
		ptDcode = PtDboundary.PtDcheck(data,patches)
		if ptDcode == 1:
			pdtext = PtDboundary.PtDtext(data,path,patches)
			fname = "pointDisplacement"
			filepath = os.path.join(path, "0.org", fname)
			ptDfile = open(filepath, "w")
			ptDfile.write(pdtext)
			ptDfile.close()

		return 0

	#############################################################
	def parallelize(self,data,path):
		'''
		Creates the domain decomposition files for openfoam7

		Arguments
		-----------
			data: all the JSON data
			path: Path where the geometry files (STL) needs to be created
		'''

		# Create the domain decomposition file
		Decomp = of7Decomp()
		decomptext = Decomp.decomptext(data)
		fname = "decomposeParDict"
		filepath = os.path.join(path, "system", fname)
		decompfile = open(filepath, "w")
		decompfile.write(decomptext)
		decompfile.close()

		return 0

	#############################################################
	def solve(self,data,path):
		'''
		Creates the solver related files for openfoam7

		Arguments
		-----------
			data: all the JSON data
			path: Path where the geometry files (STL) needs to be created
		'''

		# Create the solver files
		Solve = of7Solve()
		# fvSchemes
		fvschemetext = Solve.fvSchemetext(data)
		fname = "fvSchemes"
		filepath = os.path.join(path, "system", fname)
		fvschemefile = open(filepath,"w")
		fvschemefile.write(fvschemetext)
		fvschemefile.close()

		#fvSolutions
		fvsolntext = Solve.fvSolntext(data)
		fname = "fvSolution"
		filepath = os.path.join(path, "system", fname)
		fvsolnfile = open(filepath,"w")
		fvsolnfile.write(fvsolntext)
		fvsolnfile.close()

		# controlDict
		ecode = Solve.cdictcheck(data)
		if ecode == -1:
			return -1
		else:
			cdicttext = Solve.cdicttext(data)
			fname = "controlDict"
			filepath = os.path.join(path, "system", fname)
			cdictfile = open(filepath,"w")
			cdictfile.write(cdicttext)
			cdictfile.close()

		return 0

	#############################################################
	def others(self,data,path):
		'''
		Creates the other auxillary files for openfoam7

		Arguments
		-----------
			data: all the JSON data
			path: Path where the geometry files (STL) needs to be created
		'''

		# Create the auxillary files
		Others = of7Others()
		# g-file
		gfiletext = Others.gfiletext(data)
		fname = "g"
		filepath = os.path.join(path, "constant", fname)
		gfile = open(filepath,"w")
		gfile.write(gfiletext)
		gfile.close()

		return 0

	#############################################################
	def scripts(self,args,path):
		'''
		Creates the scripts required to run for openfoam7

		Arguments
		-----------
			args: User input arguments
			path: Path where the geometry files (STL) needs to be created
		'''

		# Create the auxillary files
		hpc = TACCdesignsafe()
		# # g-file
		# gfiletext = Others.gfiletext(data)
		# fname = "g"
		# filepath = os.path.join(path, "constant", fname)
		# gfile = open(filepath,"w")
		# gfile.write(gfiletext)
		# gfile.close()

		return 0