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
import meshio
import numpy as np
import re

# Other custom modules
from hydroUtils import hydroUtils


####################################################################
# OpenFOAM7 solver class
####################################################################
class of7Building():
	"""
	This class includes the methods related to
	creating the building for openfoam7.

	Methods
	--------
		buildcheck: Checks if all files required for creating the building exists
		createbuilds: Creates the STL files
	"""

	#############################################################
	def buildcheck(self,data,path):
		'''
		Checks if all files required for creating the building exists

		Arguments
		-----------
			data: all the JSON data
			path: Path to where the dakota.json exists
		'''

		# Create a utilities object
		hydroutil = hydroUtils()

		# Check if a translate script exists. 
		# If so delete it
		if os.path.exists('translate.sh'):
			os.remove('translate.sh')

		# Check for STL file
		# Get the type of building definition
		buildeftype = ', '.join(hydroutil.extract_element_from_json(data, ["Events","BuildData"]))
		if buildeftype == 'Manual':
			# Find number of buildings
			numbuild = ', '.join(hydroutil.extract_element_from_json(data, ["Events","NumBuild"]))
			if int(numbuild) > 0:
				# Number of buildings with response
				numbuildres = 0
				# Get data for each building
				for ii in range(int(numbuild)):
					builddata = ', '.join(hydroutil.extract_element_from_json(data, ["Events","BuildingTable"+str(ii)]))
					builddata = builddata.replace(',',' ')
					nums = [float(n) for n in builddata.split()]
					buildtype = nums[0]
					if int(buildtype) == -1 or int(buildtype) == 2:
						stlfile = hydroutil.extract_element_from_json(data, ["Events","BuildingSTLFile"])
						if stlfile == [None]:
							return -1
						else:
							stlfile = ', '.join(hydroutil.extract_element_from_json(data, ["Events","BuildingSTLFile"]))
							if not os.path.exists(os.path.join(path,stlfile)):
								return -1
							
					if int(buildtype) == -2 or int(buildtype) == -1:
						numbuildres += 1
						# Check GI
						depth = hydroutil.extract_element_from_json(data, ["GeneralInformation","depth"])
						if str(depth[0]) == [None]:
							return -1

						width = hydroutil.extract_element_from_json(data, ["GeneralInformation","width"])
						if str(width[0]) == [None]:
							return -1

						height = hydroutil.extract_element_from_json(data, ["GeneralInformation","height"])
						if str(height[0]) == [None]:
							return -1

						# xbuild = hydroutil.extract_element_from_json(data, ["GeneralInformation","xbuild"])
						# if str(xbuild[0]) == [None]:
						# 	return -1

						# ybuild = hydroutil.extract_element_from_json(data, ["GeneralInformation","ybuild"])
						# if str(ybuild[0]) == [None]:
						# 	return -1
						geninfo = hydroutil.extract_element_from_json(data, ["GeneralInformation"])
						geninfo = str(geninfo[0])
						# depth = geninfo.partition("'depth': ")[2].partition(", 'height':")[0]
						# width = geninfo.partition("'width': ")[2].partition("}")[0]
						# height = geninfo.partition("'height': ")[2].partition(", 'location':")[0]
						xbuild = geninfo.partition("'location': {'latitude': ")[1].partition(", 'longitude':")[0]
						ybuild = geninfo.partition("'longitude': ")[2].partition("},")[0]
						# if not depth:
						# 	return -1
						# # else:
						# # 	depth = float(depth)
						# if not width:
						# 	return -1
						# # else:
						# # 	width = float(width)
						# if not height:
						# 	return -1
						# # else:
						# # 	height = float(height)
						if not xbuild:
							return -1
						# else:
						# 	xbuild = float(float)
						if not ybuild:
							return -1
						# else:
						# 	ybuild = float(ybuild)

					if numbuildres > 1:
						return -1

		elif buildeftype == 'Parameters':
			buildshape = ', '.join(hydroutil.extract_element_from_json(data, ["Events","BuildShape"]))
			if int(buildshape) == 0:
				return -1
			elif int(buildshape) == 1:
				stlfile = hydroutil.extract_element_from_json(data, ["Events","BuildingSTLFile"])
				if stlfile == [None]:
					return -1
				else:
					stlfile = ', '.join(hydroutil.extract_element_from_json(data, ["Events","BuildingSTLFile"]))
					if not os.path.exists(os.path.join(path,stlfile)):
						return -1

			# Check if building distribution selected
			buildDist = ', '.join(hydroutil.extract_element_from_json(data, ["Events","BuildDist"]))
			if int(buildDist) == 0:
				return -1
			

		return 0

	#############################################################
	def createbuilds(self,data,path):
		'''
		Creates the STL files for the buildings and move to correct location

		Arguments
		-----------
			data: all the JSON data
			path: Path to where the dakota.json exists
		'''

		# Create a utilities object
		hydroutil = hydroUtils()

		# Get the type of building definition
		buildeftype = ', '.join(hydroutil.extract_element_from_json(data, ["Events","BuildData"]))
		if buildeftype == 'Manual':
			self.buildmanual(data,path)
					
		elif buildeftype == 'Parameters':
			self.buildpara(data,path)
	
		return 0

	#############################################################
	def buildmanual(self,data,path):
		'''
		Creates the STL files for the buildings using manual data from table

		Arguments
		-----------
			data: all the JSON data
			path: Path to where the dakota.json exists
		'''

		# Create a utilities object
		hydroutil = hydroUtils()

		# Number of types of buildings
		numresbuild = 0
		numotherbuild = 0

		# Find number of buildings
		numbuild = ', '.join(hydroutil.extract_element_from_json(data, ["Events","NumBuild"]))
		if int(numbuild) > 0:
			# Get data for each building
			for ii in range(int(numbuild)):
				builddata = ', '.join(hydroutil.extract_element_from_json(data, ["Events","BuildingTable"+str(ii)]))
				builddata = builddata.replace(',',' ')
				nums = [float(n) for n in builddata.split()]
				buildtype = nums[0]

				if int(buildtype) == -2:
					print('response + cuboid + GI')
					# Create a temporary file using GI information (Response)
					self.buildcubeGI(data,path)
					# Call flume to build an STL
					# Increment response buildign number
					numresbuild += 1
				elif int(buildtype) == -1:
					# Check if STL file exists (Response)
					self.readResSTL(data,path,nums[3])
					# Increment response buildign number
					numresbuild += 1
				elif int(buildtype) == 1:
					print('no response + cuboid')
					# Create a temporary file 
					# Call flume to build an STL
					# Combine all STL to building + number
					# Increment response buildign number
					numotherbuild += 1
				elif int(buildtype) == 2:
					print('no response + STL')
					# Check if STL file exists
					# Increment response buildign number
					numotherbuild += 1

		# Create other buildings STL if more than one exists (Join buildings)

		# Create the building flag
		self.buildflagadd(numresbuild,numotherbuild)

	#############################################################
	def buildpara(self,data,path):
		'''
		Creates the STL files for the buildings using parametrized data

		Arguments
		-----------
			data: all the JSON data
			path: Path to where the dakota.json exists
		'''

		# Create a utilities object
		hydroutil = hydroUtils()

	#############################################################
	def buildcubeGI(self,data,path):
		'''
		Creates the STL files for the buildings using parametrized data

		Arguments
		-----------
			data: all the JSON data
			path: Path to where the dakota.json exists
		'''

		# Create a utilities object
		hydroutil = hydroUtils()

	#############################################################
	def readResSTL(self,data,path,ztrans):
		'''
		Creates the STL files for the buildings using parametrized data

		Arguments
		-----------
			data: all the JSON data
			path: Path to where the dakota.json exists
			ztrans: Translation distance in z-direction
		'''

		# Create a utilities object
		hydroutil = hydroUtils()

		# Filename
		stlfile = ', '.join(hydroutil.extract_element_from_json(data, ["Events","BuildingSTLFile"]))

		# Read the stlfile
		stlfilepath = os.path.join(path,stlfile)
		mesh = meshio.read(stlfilepath,file_format="stl")
		mesh.points[:,0] = mesh.points[:,0]/(max(abs(mesh.points[:,0])))
		mesh.points[:,1] = mesh.points[:,1]/(max(abs(mesh.points[:,1])))
		mesh.points[:,2] = mesh.points[:,2]/(max(abs(mesh.points[:,2])))

		# Get GI
		geninfo = hydroutil.extract_element_from_json(data, ["GeneralInformation"])
		geninfo = str(geninfo[0])
		# depth = float(geninfo.partition("'depth': ")[2].partition(", 'height':")[0])
		# width = float(geninfo.partition("'width': ")[2].partition("}")[0])
		# height = float(geninfo.partition("'height': ")[2].partition(", 'location':")[0])
		xbuild = float(geninfo.partition("'location': {'latitude': ")[2].partition(", 'longitude':")[0])
		ybuild = float(geninfo.partition("'longitude': ")[2].partition("},")[0])
		depth = hydroutil.extract_element_from_json(data, ["GeneralInformation","depth"])
		depth = int(depth[0])
		width = hydroutil.extract_element_from_json(data, ["GeneralInformation","width"])
		width = int(width[0])
		height = hydroutil.extract_element_from_json(data, ["GeneralInformation","height"])
		height = int(height[0])

		# Scale the STL model
		mesh.points[:,0] = mesh.points[:,0]*depth
		mesh.points[:,1] = mesh.points[:,1]*width
		mesh.points[:,2] = mesh.points[:,2]*height

		# Write meshfile
		meshio.write_points_cells('Building.stl', mesh.points, mesh.cells)

		# Modify first and last line
		with open("Building.stl") as f:
			lines = f.readlines()
			lines[0] = 'solid '+ 'Building' + '\n'
			lines[len(lines)-1] = 'endsolid ' + 'Building' + '\n'

		# Write the updated file	
		with open('Building.stl', "w") as f:
			f.writelines(lines)
		
		# Move the file to constant/triSurface folder
		newfilepath = os.path.join(path,'constant','triSurface','Building.stl')
		os.replace('Building.stl',newfilepath)

		# Create the translation script
		if os.path.exists('translate.sh'):
			with open('translate.sh','a') as f:
				buildpath = os.path.join('constant','triSurface','Building.stl')
				lines = 'export FILE="' + buildpath + '"\n'
				lines = lines + 'surfaceTransformPoints -translate "(' + str(xbuild) + ' ' + str(ybuild) + ' ' + str(ztrans) +')" $FILE $FILE\n'
				f.writelines(lines)
		else:
			with open('translate.sh','w') as f:
				buildpath = os.path.join('constant','triSurface','Building.stl')
				lines = 'export FILE="' + buildpath + '"\n'
				lines = lines + 'surfaceTransformPoints -translate "(' + str(xbuild) + ' ' + str(ybuild) + ' ' + str(ztrans) + ')" $FILE $FILE\n'
				f.writelines(lines)
		
	#############################################################
	def buildflagadd(self,numresbuild,numotherbuild):
		'''
		Add building flag to temp_geometry.txt

		Arguments
		-----------
			numresbuild: Number of building with response
			numotherbuild: NUmber of other buildings
		'''

		# Get building flag
		if numresbuild == 0 and numotherbuild == 0:
			flag = 0
		elif numresbuild > 0 and numotherbuild == 0:
			flag = 1
		elif numresbuild > 0 and numotherbuild > 0:
			flag = 2
		elif numresbuild == 0 and numotherbuild > 0:
			flag = 3

		# Add building flag to temp file
		with open('temp_geometry.txt', "a") as f:
			f.writelines(str(flag)+'\n')
