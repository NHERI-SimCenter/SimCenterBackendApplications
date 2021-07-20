####################################################################
# LICENSING INFORMATION
####################################################################
"""
	LICENSE INFORMATION:

	Copyright (c) 2020-2030, The Regents of the University of California (Regents).

	All rights reserved.

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
	2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

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
import argparse
import json
import sys
import datetime

# Other custom modules
from hydroUtils import hydroUtils
from openfoam7 import openfoam7

####################################################################
# Main function
####################################################################
def main():
	"""
	This is the primary function

	Objects:
		h2oparser: Parser for CLI arguments

	Functions:
		main(): All necessary calls are made from this routine

	Variables:
		fipath: Path to dakota.json
	"""

	# Get the system argument
	# Create a parser Object
	h2oparser = argparse.ArgumentParser(description='Get the Dakota.json file')

	# Add the arguments
	# Path to dakota.json input file
	h2oparser.add_argument(
		'-b',
		metavar='path to input file',
		type=str,
		help='the path to Dakota.json file',
		required=True)
	# # Input directory - templateDir
	# h2oparser.add_argument(
	# 	'-I',
	# 	metavar='path to input directory',
	# 	type=str,
	# 	help='the path to input directory',
	# 	required=True)
	# # Library
	# h2oparser.add_argument(
	# 	'-L',
	# 	metavar='path to library',
	# 	type=str,
	# 	help='the path to library',
	# 	required=True)
	# # User bin
	# h2oparser.add_argument(
	# 	'-P',
	# 	metavar='path to user bin',
	# 	type=str,
	# 	help='the path to user app bin',
	# 	required=True)
	# # Input file
	# h2oparser.add_argument(
	# 	'-i',
	# 	metavar='input file',
	# 	type=str,
	# 	help='input file',
	# 	required=True)
	# # Driver file
	# h2oparser.add_argument(
	# 	'-d',
	# 	metavar='driver file',
	# 	type=str,
	# 	help='driver file',
	# 	required=True)
		
	# Execute the parse_args() method
	args = h2oparser.parse_args()

	# Get the path
	fipath = args.b.replace('/dakota.json','')
	
	# Open the JSON file and load all objects
	with open(args.b) as f:
		data = json.load(f)

	# Create a utilities object
	hydroutil = hydroUtils()

	# Get the project name
	projname = hydroutil.extract_element_from_json(data, ["Events","ProjectName"])
	projname = ', '.join(projname)

	# Initialize a log ID number
	logID = 0

	# Initialize the log
	hydroutil.hydrolog(projname,fipath)

	# Start the log file with header and time and date
	logfiletext = hydroutil.general_header()
	hydroutil.flog.write(logfiletext)
	logID += 1
	hydroutil.flog.write('%d (%s): This log has started.\n' % (logID,datetime.datetime.now()))

	# Get the solver type from the dakota file
	hydrosolver = ', '.join(hydroutil.extract_element_from_json(data, ["Events","SolverChoice"]))
		
	# Find the solver
	# 0 - OpenFoam7 (+ olaFlow)
	# 1 - OpenFoam 8 (+ olaFlow)
	# default - OpenFoam7 (+ olaFlow)
	# Create related object
	if int(hydrosolver) == 0:
		# OpenFoam 7 + olaFlow
		solver = openfoam7()

	elif int(hydrosolver) == 1:
		print('This is not yet available')
		# OpenFoam 8 + olaFlow
		# solver = openfoam8()
	
	else:
		# Default is Openfoam7 + olaFlow
		solver = openfoam7()

	# Call the important routines
	# Create folders
	ecode = solver.createfolder(data,fipath)
	logID += 1
	if ecode == 0:
		hydroutil.flog.write('%d (%s): Folders required for EVT solver created.\n' % (logID,datetime.datetime.now()))
	else:
		hydroutil.flog.write('%d (%s): Error creating folders required for EVT solver.\n' % (logID,datetime.datetime.now()))

	# Create Geometry (only is hydro mesher is used)
	mesher = hydroutil.extract_element_from_json(data, ["Events","MeshType"])
	if int(mesher[0]) == 0:
		print("Creating geometry and mesh")
		# Create the geometry
		#ecode = solver.geometry(data,fipath)

		# Create mesh
		#ecode = solver.meshing(data,fipath)

	# Create initial condition

	# Create boundary condition
	ecode = solver.boundary(data,fipath)
	logID += 1
	if ecode < 0:
		hydroutil.flog.write('%d (%s): Error with boundary condition definition in EVT.\n' % (logID,datetime.datetime.now()))
		sys.exit('Issues with definition of boundary condition in EVT')
	else:
		hydroutil.flog.write('%d (%s): Files required for boundary condition definition successfully created.\n' % (logID,datetime.datetime.now()))
		
	# Material
	ecode = solver.materials(data,fipath)
	logID += 1
	if ecode < 0:
		hydroutil.flog.write('%d (%s): Error with material parameters in EVT.\n' % (logID,datetime.datetime.now()))
		sys.exit('Error with material parameters in EVT.')
	else:
		hydroutil.flog.write('%d (%s): Files required for materials definition successfully created.\n' % (logID,datetime.datetime.now()))

	# Turbulence
	# ecode = solver.turbulence(data,fipath)
	# logID += 1
	# if ecode < 0:
	# 	hydroutil.flog.write('%d (%s): Error with turbulence parameters in EVT.\n' % (logID,datetime.datetime.now()))
	# 	sys.exit('Error with turbulence parameters in EVT.')
	# else:
	# 	hydroutil.flog.write('%d (%s): Files required for turbulence definition successfully created.\n' % (logID,datetime.datetime.now()))

	# Solver

	# Create run script


####################################################################
# Primary function call
####################################################################
if __name__ == "__main__":

	# Call the main routine
	main()

