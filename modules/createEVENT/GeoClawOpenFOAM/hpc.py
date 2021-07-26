####################################################################
# Import all necessary modules
####################################################################
# Standard python modules
import numpy as np
import os

# Other custom modules
from hydroUtils import hydroUtils

####################################################################
# Hydro-UQ utilities class
####################################################################
class OlaFlowDakota():
	"""
	This class includes all the utilities that are
	required for script generation for TACC.

	Methods
	--------
		extract: 
	"""

	#############################################################
	def caseruntext(self,args,data,fipath):
		'''
		This generates the first script (caserun.sh) to be run before completion of the job.

		Arguments
		-----------
			args: User input arguments
			data: Entire json file
			path: Path where the dakota.json file exists
		'''

		# Create a utilities object
		hydroutil = hydroUtils()

		# Get the information from json file
		hydrobrain = ', '.join(hydroutil.extract_element_from_json(data, ["remoteAppDir"]))
		mesher = ', '.join(hydroutil.extract_element_from_json(data, ["Events","MeshType"]))
		simtype = ', '.join(hydroutil.extract_element_from_json(data, ["Events","SimulationType"]))

		# Add all variables
		# caseruntext = 'echo Setting up variables\n\n'
		# caseruntext = caseruntext + 'export BIM='+args.b+'\n\n'
		# caseruntext = caseruntext + 'export HYDROPATH='+fipath+'\n\n'
		# # caseruntext = caseruntext + 'export LD_LIBRARY_PATH='+args.L+'\n\n'
		# # caseruntext = caseruntext + 'export PATH='+args.P+'\n\n'
		# # caseruntext = caseruntext + 'export inputFile='+args.i+'\n\n'
		# # caseruntext = caseruntext + 'export driverFile='+args.d+'\n\n'
		# caseruntext = caseruntext + 'export inputDirectory='+fipath+'\n\n'	
		# caseruntext = caseruntext + 'export HYDROBRAIN='+os.path.join(hydrobrain,'applications','createEVENT','GeoClawOpenFOAM')+'\n\n'
	
		# Load all modules
		# caseruntext = caseruntext + 'echo Loading modules on Stampede2\n'
		# caseruntext = caseruntext + 'module load intel/18.0.2\n'
		# caseruntext = caseruntext + 'module load impi/18.0.2\n'
		# caseruntext = caseruntext + 'module load openfoam/7.0\n'
		# caseruntext = caseruntext + 'module load dakota/6.8.0\n'
		# caseruntext = caseruntext + 'module load python3\n\n'

		# Move the case files to the present folder
		# zerofldr = os.path.join(fipath, '0.org')
		# zero2fldr = os.path.join(fipath, '0')
		# cstfldr = os.path.join(fipath, 'constant')
		# systfldr = os.path.join(fipath, 'system')
		# caseruntext = caseruntext + 'cp -r ' + zerofldr + ' .\n'
		# caseruntext = caseruntext + 'cp -r ' + zerofldr + ' ' + zero2fldr + ' .\n'
		# caseruntext = caseruntext + 'cp -r ' + cstfldr + ' .\n'
		# caseruntext = caseruntext + 'cp -r ' + systfldr + ' .\n'

		# Start with meshing
		# Start with meshing
		# if int(mesher[0]) == 0:
			# Join all the STL files
			# Read the temporary geometry file with extreme values
			# geofile = os.path.join(fipath,'temp_geometry')
			# data_geoext = np.genfromtxt(geofile, dtype=(float))
			# flag = int(data_geoext[6])
			
			# entryf = os.path.join('constant' + 'triSurface' + 'Entry.stl')
			# exitf = os.path.join('constant' + 'triSurface' + 'Exit.stl')
			# topf = os.path.join('constant' + 'triSurface' + 'Top.stl')
			# bottomf = os.path.join('constant' + 'triSurface' + 'Bottom.stl')
			# leftf = os.path.join('constant' + 'triSurface' + 'Left.stl')
			# rightf = os.path.join('constant' + 'triSurface' + 'Right.stl')
			# buildingf = os.path.join('constant' + 'triSurface' + 'Building.stl')
			# otherbuildingf = os.path.join('constant' + 'triSurface' + 'OtherBuilding.stl')
			# all01 = 'cat '+ entryf + ' ' + exitf + ' ' + topf + ' ' + bottomf + ' ' + leftf + ' ' + rightf 
			# full = fipath + os.path.join('constant' + 'triSurface' + 'Full.stl')

			# caseruntext = caseruntext + 'echo Combining STL files for usage...\n'
			# if flag == 0:
			# 	caseruntext = caseruntext + all01 + ' > ' + full + '\n\n'
			# elif flag == 1:
			# 	caseruntext = caseruntext + all01 + ' ' + buildingf + ' > ' + full + '\n\n'
			# elif flag == 2:
			# 	caseruntext = caseruntext + all01 + ' ' + buildingf + ' ' + otherbuildingf + ' > ' + full + '\n\n'

			# blockmesh
			# caseruntext = caseruntext + 'echo blockMesh running...\n'
			# caseruntext = caseruntext + 'blockMesh > blockMesh.log\n\n'
			# # surfaceFeatureExtract
			# caseruntext = caseruntext + 'echo surfaceFeatureExtract running...\n'
			# caseruntext = caseruntext + 'surfaceFeatureExtract -force > sFeatureExt.log\n\n'
			# # snappyHexMesh
			# caseruntext = caseruntext + 'echo snappyHexMesh running...\n'
			# caseruntext = caseruntext + 'snappyHexMesh > snappyHexMesh.log\n'
			# # Copy polyMesh folder
			# path2c = os.path.join('2','polyMesh')
			# caseruntext = caseruntext + 'cp -r ' + path2c + 'constant\n'
			# caseruntext = caseruntext + 'rm -fr 1 2\n'

		# elif int(mesher[0]) == 1:

		# 	# Get the mesh software
		# 	meshsoftware = hydroutil.extract_element_from_json(data, ["Events","MeshSoftware"])
		# 	# Get the mesh file name
		# 	meshfile = hydroutil.extract_element_from_json(data, ["Events","MeshFile"])
		# 	# Get the mesh file name
		# 	caseruntext = caseruntext + 'MESHFILE=${inputDirectory}/templatedir/'+meshfile[0]+'\n'
		# 	# Write out the appropriate commands
		# 	if int(meshsoftware[0]) == 0:
		# 		caseruntext = caseruntext + 'fluentMeshToFoam $MESHFILE > fluentMeshToFoam.log\n\n'
		# 	elif int(meshsoftware[0]) == 1:
		# 		caseruntext = caseruntext + 'ideasToFoam $MESHFILE > ideasToFoam.log\n\n'
		# 	elif int(meshsoftware[0]) == 2:
		# 		caseruntext = caseruntext + 'cfx4ToFoam $MESHFILE > cfx4ToFoam.log\n\n'
		# 	elif int(meshsoftware[0]) == 3:
		# 		caseruntext = caseruntext + 'gambitToFoam $MESHFILE > gambitToFoam.log\n\n'
		# 	elif int(meshsoftware[0]) == 4:
		# 		caseruntext = caseruntext + 'gmshToFoam $MESHFILE > gmshToFoam.log\n\n'
		
		# elif int(mesher[0]) == 2:

		# 	# COPY THE FILES TO THE RIGHT LOCATION
		# 	# blockMesh
		# 	bmfile = os.path.join(fipath,'blockMeshDict')
		# 	if os.path.isfile(bmfile):
		# 		bmfilenew = os.path.join('system','blockMeshDict')
		# 		caseruntext = caseruntext + 'cp ' + bmfile + ' ' + bmfilenew + '\n'
		# 		caseruntext = caseruntext + 'echo blockMesh running...\n'
		# 		caseruntext = caseruntext + 'blockMesh > blockMesh.log\n\n'

		# 	#surfaceFeatureExtract
		# 	sfdfile = os.path.join(fipath,'surfaceFeatureExtractDict')
		# 	if os.path.isfile(sfdfile):
		# 		sfdfilenew = os.path.join('system','surfaceFeatureExtractDict')
		# 		caseruntext = caseruntext + 'cp ' + sfdfile + ' ' + sfdfilenew + '\n'
		# 		caseruntext = caseruntext + 'echo surfaceFeatureExtract running...\n'
		# 		caseruntext = caseruntext + 'surfaceFeatureExtract -force > sFeatureExt.log\n\n'

		# 	# snappyHexMesh
		# 	shmfile = os.path.join(fipath,'snappyHexMeshDict')
		# 	if os.path.isfile(shmfile):
		# 		shmfilenew = os.path.join('system','snappyHexMeshDict')
		# 		caseruntext = caseruntext + 'cp ' + shmfile + ' ' + shmfilenew + '\n'
		# 		caseruntext = caseruntext + 'echo snappyHexMesh running...\n'
		# 		caseruntext = caseruntext + 'snappyHexMesh > snappyHexMesh.log\n'
		# 		path2c = os.path.join('2','polyMesh')
		# 		caseruntext = caseruntext + 'cp -r ' + path2c + 'constant\n'
		# 		caseruntext = caseruntext + 'rm -fr 1 2\n'
			
		# # Copy the new controlDict with actual deltaT and writeT
		# newcdictpath = os.path.join('system','controlDict')
		# caseruntext = caseruntext + ' mv cdictreal ' + newcdictpath + '\n\n'

		# # Check the mesh
		# caseruntext = caseruntext + 'echo Checking mesh...\n'
		# caseruntext = caseruntext + 'checkMesh > Meshcheck.log\n\n'

		# # Create the 0-folder
		# caseruntext = caseruntext + 'echo Creating 0-folder...\n'
		# caseruntext = caseruntext + 'rm -fr 0\n'
		# caseruntext = caseruntext + 'cp -r 0.org 0\n\n'

		# caseruntext = caseruntext + 'echo Setting fields...\n'
		# caseruntext = caseruntext + 'setFields > setFields.log\n\n'

		# # Get the number of processors required
		# totalprocs = ', '.join(hydroutil.extract_element_from_json(data, ["Events","DomainDecomposition"]))

		# # Start the CFD run with n-processors
		# if int(totalprocs) > 1:
		# 	# Decompose the domain
		# 	caseruntext = caseruntext + 'echo Decomposing domain...\n'
		# 	caseruntext = caseruntext + 'decomposePar > decomposePar.log\n\n'

		# 	# Start the CFD simulation
		# 	caseruntext = caseruntext + 'echo Starting CFD simulation in parallel...\n'
		# 	if int(simtype) == 4:
		# 		caseruntext = caseruntext + 'ibrun -n ' + totalprocs + ' -o 0 olaDyMFlow -parallel > olaDyMFlow.log\n\n'
		# 	else:
		# 		caseruntext = caseruntext + 'ibrun -n ' + totalprocs + ' -o 0 olaFlow -parallel > olaFlow.log\n\n'
						
		# else:
		# 	caseruntext = caseruntext + 'echo Starting CFD simulation in serial...\n'
		# 	if int(simtype) == 4:
		# 		caseruntext = caseruntext + 'olaDyMFlow > olaDyMFlow.log\n\n'
		# 	else:
		# 		caseruntext = caseruntext + 'olaFlow > olaFlow.log\n\n'

		# Call building forces to run Dakota
		caseruntext = caseruntext + 'echo Starting Dakota preparation...\n'
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

		# Post-processing and cleanup
		caseruntext = caseruntext + 'chmod +x pprocessrun.sh \n'
		caseruntext = caseruntext + './pprocessrun.sh \n\n'
		caseruntext = caseruntext + 'chmod +x cleanrun.sh \n'
		caseruntext = caseruntext + './cleanrun.sh\n\n'

		# Complete HydroUQ
		caseruntext = caseruntext + 'echo HydroUQ complete'

		# Return the text
		return caseruntext

	
	# #############################################################
	# def pprocesstext(self,data):
	# 	'''
	# 	This generates the script to postprocess.

	# 	Arguments
	# 	-----------
	# 		args: User input arguments
	# 		data: Entire json file
	# 		path: Path where the dakota.json file exists
	# 	'''

	# 	# Create a utilities object
	# 	hydroutil = hydroUtils()

	# 	# Get the information from json file
	# 	pprocess = hydroutil.extract_element_from_json(data, ["Events","Postprocessing"])
	# 	pprocess = ', '.join(pprocess)
	# 	print(pprocess)
	# 	if pprocess == 'No':
	# 		pproruntext = 'echo no postprocessing for EVT'
	# 	elif pprocess == 'Yes':
	# 		pproruntext = 'echo postprocessing starting for EVT'
	# 		pproruntext = pproruntext + 'reconstructPar \n\n'
	# 		cdictpppath = os.path.join('system','controlDict')
	# 		pproruntext = pproruntext + 'mv cdictpp ' + cdictpppath 
	# 		samplepath = os.path.join('system','sample')
	# 		pproruntext = pproruntext + 'mv sample ' + samplepath 
	# 		pproruntext = pproruntext + 'postProcess -func sample'

	# 	return pproruntext

	