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

####################################################################
# Hydro-UQ utilities class
####################################################################
class hydroUtils():
	"""
	This class includes all the general utilities that are
	required for the Hydro-UQ.

	Methods
	--------
		extract: Extracts an element from a nested json 
		extract_element_from_json: Extracts an element from a nested json
		hydrolog: Initializes the log file
		general_header: Creates the header for the Hydro-UQ files
	"""

	#############################################################
	def extract(self,obj,path,ind,arr):
		'''
		Extracts an element from a nested dictionary
		along a specified path and returns a list.

		Arguments
		-----------
			obj: A dict - input dictionary
			path: A list - list of strings that form the JSON path
			ind: An int - starting index
			arr: A list - output list
		'''
		
		key = path[ind]
		if ind + 1 < len(path):
			if isinstance(obj, dict):
				if key in obj.keys():
					self.extract(obj.get(key), path, ind + 1, arr)
				else:
					arr.append(None)
			elif isinstance(obj, list):
				if not obj:
					arr.append(None)
				else:
					for item in obj:
						self.extract(item, path, ind, arr)
			else:
				arr.append(None)
		if ind + 1 == len(path):
			if isinstance(obj, list):
				if not obj:
					arr.append(None)
				else:
					for item in obj:
						arr.append(item.get(key, None))
			elif isinstance(obj, dict):
				arr.append(obj.get(key, None))
			else:
				arr.append(None)
		
		return arr

	#############################################################
	def extract_element_from_json(self,obj,path):
		'''
		Extracts an element from a nested dictionary or
		a list of nested dictionaries along a specified path.
		If the input is a dictionary, a list is returned.
		If the input is a list of dictionary, a list of lists is returned.

		Arguments
		-----------
			obj: A list or dict - input dictionary or list of dictionaries
			path: A list - list of strings that form the path to the desired element
		'''
		
		if isinstance(obj, dict):
			return self.extract(obj, path, 0, [])
		elif isinstance(obj, list):
			outer_arr = []
			for item in obj:
				outer_arr.append(self.extract(item, path, 0, []))
			return outer_arr
	
	#############################################################
	def general_header(self):
		'''
		Used to create a general header for Hydro-UQ related files

		Variables
		-----------
			header: Stores the general header for the Hydro-UQ files
		'''

		header = """/*--------------------------*- NHERI SimCenter -*----------------------------*\ 
|	   | H  |
|	   | Y  | HydroUQ: Water-based Natural Hazards Modeling Application
|======| D  | Website: simcenter.designsafe-ci.org/research-tools/hydro-uq
|	   | R  | Version: 1.00
|	   | O  |
\*---------------------------------------------------------------------------*/ \n\n"""
		
		return header

	####################################################################
	def of7header(self,OFclass,location,filename):
		'''
		Method to create a header for the input dictionaries.

		Variables
		-----------
			header: FileID for the file being created
		'''

		header = """/*--------------------------*- NHERI SimCenter -*----------------------------*\ 
|	   | H |
|	   | Y | HydroUQ: Water-based Natural Hazards Modeling Application
|======| D | Website: simcenter.designsafe-ci.org/research-tools/hydro-uq
|	   | R | Version: 1.00
|	   | O |
\*---------------------------------------------------------------------------*/ 
FoamFile
{{
	version   2.0;
	format	ascii;
	class	 {};
	location  "{}";
	object	{};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n""".format(OFclass,location,filename)

		return header

	#############################################################
	def hydrolog(self,projname,fipath):
		'''
		Used to initialize the log file for the Hydro-UQ program

		Arguments
		-----------
			projname: Name of the project as given by the user
			fipath: Path where the log file needs to be created

		Variables
		-----------
			flog: File pointer to the log file
		'''

		# Open a log file to write the outputs
		# Use project name for the log file
		# If no project name is specified, call it Untitled
		if projname != "":
			fname = ''.join(projname.split())+".h20log"
		else:
			fname = "Untitled.h20log"

		# Path to the file
		filepath = os.path.join(fipath, fname)
		print(filepath)
		self.flog = open(filepath, "w")