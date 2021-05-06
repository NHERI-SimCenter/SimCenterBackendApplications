####################################################################
# OpenFOAM class for Hydro-UQ
####################################################################
import numpy as np
import os
import shutil
import math
from zipfile36 import ZipFile
from GenUtilities import genUtilities # General utilities
from Flume import OSUWaveFlume # Wave flume utilities

class solver(object):
    '''
    This class includes all the general utilities that are
    required for the Hydro-UQ.
    
    Primary methods
    ----------------
        dircreate: Method to create necessary directories for the solver
        filecreate: Method to create supplementary files
        matmodel: Method for material models
        solvecontrol: Method for solver control
        parallel: Method for parallelization
        geometry: Method for creating STL files
        meshing: Method for creating relevant mesh files
        boundary: Method for creating boundary condition files
        initial: Method for creating initial condition files

    Secondary methods
    -------------------
        gfileOF: Method creates the gravity file for the OpenFOAM
        fvSchemesOF: Method to create the fvSchemes file
        fvSolutionOF: Method to create the fvSolution file
        transportProperties: Method for material properties of water & air
        turbProperties: Method for turbulence properties
        solvecontrol: Method for creation of control dictionary
        decomposepar: Method for decomposparDict / Parallelization dictionary
        bMeshDictOF: Creates the blockMesh dictionary
        surffeatureextDictOF: Creates the surfaceFeatureExtract dictionary
        snappyhexmeshDictOF: Create the snappyHexMesh dictionary
    
    Tertiary methods
    --------------------
        openfileOF: Method to initiate a file and return file pointer
        headerOF: Method to create the header for the input dictionaries
        constvarfileOF: Method to add information into the constant file
    '''

    ####################################################################
    def openfileOF(self,filename,permission):
        '''
        Method initiates and opens a file with particular permission
        and returns a file pointer.

        Variables
        -----------
            fileID: FileID for the file being created
        '''

        fileID = open(filename,permission)
        return fileID

    ####################################################################
    def headerOF(self,OFclass,location,filename):
        '''
        Method to create a header for the input dictionaries.

        Variables
        -----------
            header: FileID for the file being created
        '''

        header = """/*--------------------------*- NHERI SimCenter -*----------------------------*\ 
|       | H |
|       | Y | HydroUQ: Water-based Natural Hazards Modeling Application
|=======| D | Website: simcenter.designsafe-ci.org/research-tools/hydro-uq
|       | R | Version: 1.00
|       | O |
\*---------------------------------------------------------------------------*/ 
FoamFile
{{
    version   2.0;
    format    ascii;
    class     {};
    location  "{}";
    object    {};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n""".format(OFclass,location,filename)

        return header

    ####################################################################
    def header2OF(self,OFclass,filename):
        '''
        Method to create a header for the input dictionaries.

        Variables
        -----------
            header: FileID for the file being created
        '''

        header = """/*--------------------------*- NHERI SimCenter -*----------------------------*\ 
|       | H |
|       | Y | HydroUQ: Water-based Natural Hazards Modeling Application
|=======| D | Website: simcenter.designsafe-ci.org/research-tools/hydro-uq
|       | R | Version: 1.00
|       | O |
\*---------------------------------------------------------------------------*/ 
FoamFile
{{
    version   2.0;
    format    ascii;
    class     {};
    object    {};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n""".format(OFclass,filename)

        return header


    ####################################################################
    def constvarfileOF(self,var,file):
        '''
        Method adds information into the constants file

        Arguments
        -----------
            file: File name of the constants file
            var: Variable numpy list

        Variables
        -----------
            cnstfile: FileID for the constants file
        '''
        self.cnstfile.write('//%s\n' % (file))
        for ii in range(0,var.shape[0]):
            self.cnstfile.write('%s\t%s;\n' % (var[ii,0],var[ii,1]))
        self.cnstfile.write('\n')

    ####################################################################
    def gfileOF(self,data):
        '''
        Method creates the gravity file for the OpenFOAM.

        Variables
        -----------
            fileID: FileID for the gravity file
            hydroutil: General utilities object
            gx: Gravity in x-direction
            gy: Gravity in y-direction
            gz: Gravity in z-direction
            var: Temporary variable for constant
        '''
    
        # Utilities object
        hydroutil = genUtilities()
        # Get the file ID
        fileID = self.openfileOF("constant/g","w")
        # Initialize gravity
        gx = 0.0
        gy = 0.0
        gz = 0.0

        # Get the gravity from dakota.json file
        gravity = ', '.join(hydroutil.extract_element_from_json(data, ["Events","Gravity"]))
        # Depending on the inputs, initialize gravity in the right direction
        if int(gravity) == 11:
            gx = 9.81
        elif int(gravity) == 12:
            gy = 9.81
        elif int(gravity) == 13:
            gz = 9.81
        elif int(gravity) == 21:
            gx = -9.81
        elif int(gravity) == 22:
            gy = -9.81
        elif int(gravity) == 23:
            gz = -9.81

        # Write to constants file
        var =  np.array([['gx', str(gx)], ['gy', str(gy)], ['gz', str(gz)]])
        self.constvarfileOF(var,"g-file")
        
        # Write the gravity file
        # Header
        ofheader = self.headerOF("uniformDimensionedVectorField","constant","g")
        fileID.write(ofheader)
        # Add the constants file
        fileID.write('#include\t"../constantsFile"\n\n')
        # Other content
        fileID.write('dimensions\t[0 1 -2 0 0 0 0];\n')
        fileID.write('value\t($gx $gy $gz);\n')
        # Close the g-file
        fileID.close()

    ####################################################################
    def fvSchemesOF(self,data):
        '''
        Method creates the fvSchemes file for the OpenFOAM.

        Variables
        -----------
            fileID: FileID for the fvSchemes file
            ofheader: Header for the Hydro-UQ input dictionary
        '''

        # Get the file ID
        fileID = self.openfileOF("system/fvSchemes","w")
        # Header
        ofheader = self.headerOF("dictionary","system","fvSchemes")
        fileID.write(ofheader)
        # Other data
        # ddt 
        fileID.write('ddtSchemes\n{\n\tdefault\tEuler;\n}\n\n')
        # grad 
        fileID.write('gradSchemes\n{\n\tdefault\tGauss linear;\n}\n')
        # div 
        fileID.write('\ndivSchemes\n{\n\t')
        fileID.write('div(rhoPhi,U)\tGauss limitedLinearV 1;\n\t')
        fileID.write('div(U)\tGauss linear;\n\t')
        fileID.write('div((rhoPhi|interpolate(porosity)),U)\tGauss limitedLinearV 1;\n\t')
        fileID.write('div(rhoPhiPor,UPor)\tGauss limitedLinearV 1;\n\t')
        fileID.write('div(rhoPhi,UPor)\tGauss limitedLinearV 1;\n\t')
        fileID.write('div(rhoPhiPor,U)\tGauss limitedLinearV 1;\n\t') 
        fileID.write('div(phi,alpha)\tGauss vanLeer;\n\t')
        fileID.write('div(phirb,alpha)\tGauss interfaceCompression;\n\t')
        fileID.write('div((muEff*dev(T(grad(U)))))\tGauss linear;\n\t')
        fileID.write('div(phi,k)\tGauss upwind;\n\t')
        fileID.write('div(phi,epsilon)\tGauss upwind;\n\t')
        fileID.write('div((phi|interpolate(porosity)),k)\tGauss upwind;\n\t')
        fileID.write('div((phi*interpolate(rho)),k)\tGauss upwind;\n\t')
        fileID.write('div((phi|interpolate(porosity)),epsilon)\tGauss upwind;\n\t')
        fileID.write('div(phi,omega)\tGauss upwind;\n\t')
        fileID.write('div((phi|interpolate(porosity)),omega)\tGauss upwind;\n\t')
        fileID.write('div((phi*interpolate(rho)),omega)\tGauss upwind;\n\t')
        fileID.write('div((phi*interpolate(rho)),epsilon)\tGauss upwind;\n')
        fileID.write('}\n\n')
        # laplacian
        fileID.write('laplacianSchemes\n{\n\tdefault\tGauss linear corrected;\n}\n\n')
        # interpolation
        fileID.write('interpolationSchemes\n{\n\tdefault\tlinear;\n}\n\n')
        # snGrad
        fileID.write('snGradSchemes\n{\n\tdefault\tcorrected;\n}\n\n')
        # flux
        fileID.write('fluxRequired\n{\n\t')
        fileID.write('default\tno;\n\t')
        fileID.write('p_rgh;\n\t')
        fileID.write('pcorr;\n\t')
        fileID.write('alpha.water;\n')
        fileID.write('}\n')
        # Close the fvSchemes file
        fileID.close()

    ####################################################################
    def fvSolutionOF(self,data):
        '''
        Method creates the fvSolution file for the OpenFOAM. There can be potential issues when turbulence is used

        Variables
        -----------
            fileID: FileID for the fvSchemes file
            ofheader: Header for the Hydro-UQ input dictionary
            turb: Turbulence model used
            simtype: Type of simulation
        '''

        # Find turbulence model to be used
        hydroutil = genUtilities()
        turb = ', '.join(hydroutil.extract_element_from_json(data, ["Events","TurbulenceModel"]))
        simtype = ', '.join(hydroutil.extract_element_from_json(data, ["Events","SimulationType"]))

        # Get the file ID
        fileID = self.openfileOF("system/fvSolution","w")
        # Header
        ofheader = self.headerOF("dictionary","system","fvSolution")
        fileID.write(ofheader)
        # Other data
        fileID.write('solver\n{\n\t')
        # solvers: alpha
        fileID.write('"alpha.water.*"\n\t{\n\t\t')
        fileID.write('nAlphaCorr\t1;\n\t\t')
        fileID.write('nAlphaSubCycles\t2;\n\t\t')
        fileID.write('alphaOuterCorrectors\tyes;\n\t\t')
        fileID.write('cAlpha\tyes;\n\t\t')
        fileID.write('MULESCorr\tno;\n\t\t')
        fileID.write('nLimiterIter\t3;\n\t\t')
        fileID.write('solver\tsmoothSolver;\n\t\t')
        fileID.write('smoother\tsymGaussSeidel;\n\t\t')
        fileID.write('tolerance\t1e-08;\n\t\t')
        fileID.write('relTol\t0;\n\t}\n\n\t')
        # solvers: pcorr
        fileID.write('"pcorr.*"\n\t{\n\t\t')
        fileID.write('solver\tPCG;\n\t\t')
        fileID.write('preconditioner\tDIC;\n\t\t')
        fileID.write('tolerance\t1e-05;\n\t\t')
        fileID.write('relTol\t0;\n\t}\n\n\t')
        # solvers: pcorrFinal
        fileID.write('pcorrFinal\n\t{\n\t\t')
        fileID.write('$pcorr;\n\t\t')
        fileID.write('relTol\t0;\n\t}\n\n\t')
        # solvers: p_rgh
        fileID.write('p_rgh\n\t{\n\t\t')
        fileID.write('solver\tPCG;\n\t\t')
        fileID.write('preconditioner\tDIC;\n\t\t')
        fileID.write('tolerance\t1e-07;\n\t\t')
        fileID.write('relTol\t0.05;\n\t}\n\n\t')
        # solvers: p_rghFinal
        fileID.write('p_rghFinal\n\t{\n\t\t')
        fileID.write('$p_rgh;\n\t\t')
        fileID.write('relTol\t0;\n\t}\n\n\t')
        # solvers: U
        fileID.write('U\n\t{\n\t\t')
        fileID.write('solver\tsmoothSolver;\n\t\t')
        fileID.write('smoother\tsymGaussSeidel;\n\t\t')
        fileID.write('tolerance\t1e-06;\n\t\t')
        fileID.write('relTol\t0;\n\t}\n')
        # Turbulece variables (if exist)
        if (int(turb) == 1) or (int(turb) == 2):
            fileID.write('\n\t')
            fileID.write('"(k|epsilon|omega|B|nuTilda).*"\n\t{\n\t\t')
            fileID.write('solver\tsmoothSolver;\n\t\t')
            fileID.write('smoother\tsymGaussSeidel;\n\t\t')
            fileID.write('tolerance\t1e-08;\n\t\t')
            fileID.write('relTol\t0;\n\t}\n')
        # solvers: cellDisplacement (for flume)
        if int(simtype) == 4:
            # solvers: cellDisplacement (for flume)
            fileID.write('\n\t')
            fileID.write('cellDisplacement\n\t{\n\t\t')
            fileID.write('solver\tGAMG;\n\t\t')
            fileID.write('tolerance\t1e-05;\n\t\t')
            fileID.write('relTol\t0;\n\t\t')
            fileID.write('smoother\tGaussSeidel;\n\t\t')
            fileID.write('cacheAgglomeration\tfalse;\n\t\t')
            fileID.write('nCellsInCoarsestLevel\t10;\n\t\t')
            fileID.write('agglomerator\tfaceAreaPair;\n\t\t')
            fileID.write('mergeLevels\t1;\n\t}\n\n\t')
            # solvers: cellDisplacementFinal(for flume)
            fileID.write('cellDisplacementFinal\n\t{\n\t\t')
            fileID.write('$cellDisplacement;\n\t\t')
            fileID.write('relTol\t0;\n\t}\n')
        # Close solvers
        fileID.write('}\n\n')
        # PIMPLE
        fileID.write('PIMPLE\n{\n\t')
        fileID.write('momentumPredictor\tno;\n\t')
        fileID.write('nOuterCorrectors\t1;\n\t')
        fileID.write('nCorrectors\t3;\n\t')
        fileID.write('nNonOrthogonalCorrectors\t0;\n}\n\n')
        # Relaxation factors
        fileID.write('relaxationFactors\n{\n\t')
        fileID.write('fields\n\t{\n\t}\n\t')
        fileID.write('equations\n\t{\n\t\t".*"\t1;\n\t}\n}')
        fileID.close()

    ####################################################################
    def OSUwavemakerfileunzip(self,data,fpath):

        '''
        Method is used to unzip the wavemaker files if exist

        Variables
        -----------
            simtype: Type of simulation
            fpath: Path to dakota.json file folder
        '''

        # Get the simulation type
        hydroutil = genUtilities()
        simtype = ', '.join(hydroutil.extract_element_from_json(data, ["Events","SimulationType"]))

        # If the simulation type is digital wave flume
        # Then get the filename
        # Unzip the file
        zipfilename = ', '.join(hydroutil.extract_element_from_json(data, ["Events","MovingWall_Entry"]))

        # Need to unzip the file
        #zip = ZipFile('templatedir/wm.zip')
        zip = ZipFile(os.path.join(fpath,"wm.zip"))
        zip.extractall()

    ####################################################################
    def transportProperties(self,data):
        '''
        Method creates the transportProperties file for the OpenFOAM. There can be potential issues when turbulence is used

        Variables
        -----------
            fileID: FileID for the fvSchemes file
            ofheader: Header for the Hydro-UQ input dictionary
            nuwater: Kinematic viscosity of water
            nuair: Exponent in kinematic viscosity of water
            rhoair: Density of water
            nuair: Kinematic viscosity of air
            nuair: Exponent in kinematic viscosity of air
            rhoair: Density of air
            sigma: Surface tension between water and air
        '''

        # Create the transportProperties file
        fileID = open("constant/transportProperties","w")

        # Get the turbulence model
        hydroutil = genUtilities()

        # Get the properties necessary to print
        nuwater = ', '.join(hydroutil.extract_element_from_json(data, ["Events","WaterViscosity"]))
        nuwaterexp = ', '.join(hydroutil.extract_element_from_json(data, ["Events","WaterViscosityExp"]))
        rhowater = ', '.join(hydroutil.extract_element_from_json(data, ["Events","WaterDensity"]))
        nuair = ', '.join(hydroutil.extract_element_from_json(data, ["Events","AirViscosity"]))
        nuairexp = ', '.join(hydroutil.extract_element_from_json(data, ["Events","AirViscosityExp"]))
        rhoair = ', '.join(hydroutil.extract_element_from_json(data, ["Events","AirDensity"]))
        sigma = ', '.join(hydroutil.extract_element_from_json(data, ["Events","SurfaceTension"]))

        # Write the constants to the file
        var = np.array([['nuwater', nuwater+'e'+nuwaterexp], ['rhowater', rhowater], ['nuair', nuair+'e'+nuairexp], ['rhoair', rhoair], ['sigma',sigma]])
        self.constvarfileOF(var,"transportProperties")

        # Write the dictionary file
        # Header
        ofheader = self.headerOF("dictionary","constant","transportProperties")
        fileID.write(ofheader)
        # Include the constant file
        fileID.write('#include\t"../constantsFile"\n')
        # Water properties
        fileID.write('\nphases (water air);\n\n')
        fileID.write('water\n{\n\ttransportModel\tNewtonian;\n')
        fileID.write('\tnu\t[0 2 -1 0 0 0 0]\t$nuwater;\n')
        fileID.write('\trho\t[1 -3 0 0 0 0 0]\t$rhowater;\n}\n\n')
        # Air properties
        fileID.write('air\n{\n\ttransportModel\tNewtonian;\n')
        fileID.write('\tnu\t[0 2 -1 0 0 0 0]\t$nuair;\n')
        fileID.write('\trho\t[1 -3 0 0 0 0 0]\t$rhoair;\n}\n\n')
        # Surface tension
        fileID.write('sigma\t[1 0 -2 0 0 0 0]\t$sigma;\n')

        # Close the transportProperties file
        fileID.close()

    ####################################################################
    def turbProperties(self,data):
        '''
        Method is used to create the necessary files for turbulence properties

        Variables
        ------------
            fileID: FileID for the fvSchemes file
            ofheader: Header for the Hydro-UQ input dictionary
            turb: Turbulence model
        '''

        # Create the turbulenceProperties file
        fileID = open("constant/turbulenceProperties","w")

        # Get the turbulence model
        hydroutil = genUtilities()
        turb = ', '.join(hydroutil.extract_element_from_json(data, ["Events","TurbulenceModel"]))

        # Write the dictionary file
        # Header
        ofheader = self.headerOF("dictionary","constant","turbulenceProperties")
        fileID.write(ofheader)
        # Add the constants file
        fileID.write('#include\t"../constantsFile"\n\n')
        # Other content
        if int(turb) == 0:
            fileID.write('\nsimulationType\tlaminar;\n\n')
        else:
            fileID.write('\nsimulationType\tRAS;\n\n')
            fileID.write('RAS\n{\n\tRASModel\t')
            if int(turb) == 1:
                fileID.write('kEpsilon;\n\n')
            elif int(turb) == 2:
                fileID.write('kOmegaSST;\n\n')
            fileID.write('\tturbulence\ton;\n\n')
            fileID.write('\tprintCoeffs\ton;\n}')
            
        # Close the turbulenceProperties file
        fileID.close()

    ####################################################################
    def controlDict(self,data):
        '''
        Method to create the control dictionary for the solver

        Variables
        ------------
            fileID: FileID for the fvSchemes file
            ofheader: Header for the Hydro-UQ input dictionary
            startT: Start time of simulation
            endT: End of simulation
            deltaT: Time interval for simulation
            writeT: Write interval
            simtype: Type of simulation
            solver: Solver used for CFD simulation
        '''

        # Create the transportProperties file
        fileID = open("system/controlDict","w")

        # Get the turbulence model
        hydroutil = genUtilities()

        # Get required data
        startT = ''.join(hydroutil.extract_element_from_json(data, ["Events","StartTime"]))
        endT = ''.join(hydroutil.extract_element_from_json(data, ["Events","EndTime"]))
        deltaT = ''.join(hydroutil.extract_element_from_json(data, ["Events","TimeInterval"]))
        writeT = ''.join(hydroutil.extract_element_from_json(data, ["Events","WriteInterval"]))
        simtype = ', '.join(hydroutil.extract_element_from_json(data, ["Events","SimulationType"]))
        if(int(simtype) == 4):
            solver = "olaDyMFlow"
        else:
            solver = "olaFlow"

        # Write the constants to the file
        var = np.array([['startT', startT], ['endT', endT], ['deltaT', deltaT],['deltaT2', 1], ['writeT', writeT], ['solver', '"'+solver+'"']])
        self.constvarfileOF(var,"ControlDict")

        # Write the dictionary file
        ofheader = self.headerOF("dictionary","system","controlDict")
        fileID.write(ofheader)
        # Add the constants file
        fileID.write('#include\t"../constantsFile"\n\n')
        fileID.write('\napplication \t $solver;\n\n')
        fileID.write('startFrom \t latestTime;\n\n')
        fileID.write('startTime \t $startT;\n\n')
        fileID.write('stopAt \t endTime;\n\n')
        fileID.write('endTime \t $endT;\n\n')
        fileID.write('deltaT \t $deltaT2;\n\n')
        fileID.write('writeControl \t adjustableRunTime;\n\n')
        fileID.write('writeInterval \t $writeT;\n\n')
        fileID.write('purgeWrite \t 0;\n\n')
        fileID.write('writeFormat \t ascii;\n\n')
        fileID.write('writePrecision \t 6;\n\n')
        fileID.write('writeCompression \t uncompressed;\n\n')
        fileID.write('timeFormat \t general;\n\n')
        fileID.write('timePrecision \t 6;\n\n')
        fileID.write('runTimeModifiable \t yes;\n\n')
        fileID.write('adjustTimeStep \t yes;\n\n')
        fileID.write('maxCo \t 1.0;\n\n')
        fileID.write('maxAlphaCo \t 1.0;\n\n')
        fileID.write('maxDeltaT \t 1;\n\n')
        #fileID.write('libs\n(\n\t"libwaves.so"\n)\n')

        # Add post-processing stuff

        # Close the controlDict file
        fileID.close()

    ####################################################################
    def decomposepar(self,data):
        '''
        Method to create the decomposeparDict dictionary

        Variables
        -----------
            fileID: FileID for the fvSchemes file
            ofheader: Header for the Hydro-UQ input dictionary
            procs: Array of number of processors
            nums: Number of processors along x, y, z-directions
            totalprocs: Total number of processors
        '''

        # Create the transportProperties file
        fileID = open("system/decomposeParDict","w")

        # Get the turbulence model
        hydroutil = genUtilities()

        # Get required data
        procs = ', '.join(hydroutil.extract_element_from_json(data, ["Events","DomainDecomposition"]))
        method = ''.join(hydroutil.extract_element_from_json(data, ["Events","DecompositionMethod"]))

        # Find total number of processors
        procs = procs.replace(',', ' ')
        nums = [int(n) for n in procs.split()]
        totalprocs = nums[0]*nums[1]*nums[2]

        # Write the constants to the file
        var = np.array([['procX', str(nums[0])], ['procX', str(nums[1])], ['procX', str(nums[2])], ['procTotal', str(totalprocs)], ['decMeth', '"'+method+'"']])
        self.constvarfileOF(var,"decomposeParDict")

        # Write the dictionary file
        ofheader = self.headerOF("dictionary","system","decomposeParDict")
        fileID.write(ofheader)
        # Add the constants file
        fileID.write('#include\t"../constantsFile"\n\n')
        # Write the dictionary file
        fileID.write('\nnumberOfSubdomains \t $procTotal;\n\n')
        fileID.write('method \t $decMeth;\n\n')
        fileID.write('simpleCoeffs \n{\n\tn\t($procX\t$procY\t$procZ); \n\tdelta\t0.001;\n}\n\n')
        fileID.write('hierarchicalCoeffs \n{\n\tn\t($procX\t$procY\t$procZ); \n\tdelta\t0.001;\n\torder\txyz;\n}\n')

        # Close the controlDict file
        fileID.close()

    ####################################################################
    def bMeshDictOF(self):
        '''
        Method to create the blockmesh dictionary 

        Variables
        -----------
            fileID: FileID for the fvSchemes file
            ofheader: Header for the Hydro-UQ input dictionary
        '''

        # Open the blockmeshDict file
        fileID = open("system/blockMeshDict","w")

        # Add the header
        ofheader = self.headerOF("dictionary","system","blockMeshDict")
        fileID.write(ofheader)

        # Include the constant file
        fileID.write('#include\t"../constantsFile"\n\n')
        # Add the units
        fileID.write('convertToMeters\t1;\n\n')
        # Add vertices
        fileID.write('vertices\n(\n\t')
        fileID.write('($BMXmin\t$BMYmin\t$BMZmin)\n\t')
        fileID.write('($BMXmax\t$BMYmin\t$BMZmin)\n\t')
        fileID.write('($BMXmax\t$BMYmax\t$BMZmin)\n\t')
        fileID.write('($BMXmin\t$BMYmax\t$BMZmin)\n\t')
        fileID.write('($BMXmin\t$BMYmin\t$BMZmax)\n\t')
        fileID.write('($BMXmax\t$BMYmin\t$BMZmax)\n\t')
        fileID.write('($BMXmax\t$BMYmax\t$BMZmax)\n\t')
        fileID.write('($BMXmin\t$BMYmax\t$BMZmax)\n);\n\n')
        # Add blocks
        fileID.write('blocks\n(\n\t')
        fileID.write('hex (0 1 2 3 4 5 6 7) ($nx $ny $nz) simpleGrading (1 1 1)\n);\n\n')
        # Add edges
        fileID.write('edges\n(\n);\n\n')
        # Add patches
        fileID.write('patches\n(\n\t')
        fileID.write('patch maxY\n\t(\n\t\t(3 7 6 2)\n\t)\n\t')
        fileID.write('patch minX\n\t(\n\t\t(0 4 7 3)\n\t)\n\t')
        fileID.write('patch maxX\n\t(\n\t\t(2 6 5 1)\n\t)\n\t')
        fileID.write('patch minY\n\t(\n\t\t(1 5 4 0)\n\t)\n\t')
        fileID.write('patch minZ\n\t(\n\t\t(0 3 2 1)\n\t)\n\t')
        fileID.write('patch maxZ\n\t(\n\t\t(4 5 6 7)\n\t)\n')
        fileID.write(');\n\n')
        # Add merge patch pairs
        fileID.write('mergePatchPairs\n(\n);\n')

    ####################################################################
    def surffeatureextDictOF(self,flag):
        '''
        Method to create the surfacefeature extract dictionary 

        Variables
        -----------
            fileID: FileID for the fvSchemes file
            ofheader: Header for the Hydro-UQ input dictionary
        '''

        # Open the blockmeshDict file
        fileID = open("system/surfaceFeatureExtractDict","w")

        # Add the header
        ofheader = self.headerOF("dictionary","system","surfaceFeatureExtractDict")
        fileID.write(ofheader)
        stlinfo = '{\n\textractionMethod\textractFromSurface;\n'
        stlinfo = stlinfo + '\textractFromSurfaceCoeffs\n'
        stlinfo = stlinfo + '\t{includedAngle\t150;}\n'
        stlinfo = stlinfo + '\twriteObj\tyes;\n}'
        fileID.write('Front.stl\n%s\n\n' % (stlinfo))
        fileID.write('Back.stl\n%s\n\n' % (stlinfo))
        fileID.write('Right.stl\n%s\n\n' % (stlinfo))
        fileID.write('Left.stl\n%s\n\n' % (stlinfo))
        fileID.write('Top.stl\n%s\n\n' % (stlinfo))
        fileID.write('Bottom.stl\n%s\n\n' % (stlinfo))
        if flag == 1:
            fileID.write('Building.stl\n%s\n\n' % (stlinfo))
        elif flag == 2:
            fileID.write('Building.stl\n%s\n\n' % (stlinfo))
            fileID.write('OtherBuilding.stl\n%s\n\n' % (stlinfo))

    ####################################################################
    def snappyhexmeshDictOF(self,flag):
        '''
        Method to create the snappyHexMesh dictionary 

        Variables
        -----------
            fileID: FileID for the fvSchemes file
            ofheader: Header for the Hydro-UQ input dictionary
        '''

        # Open the snappyHexMeshDict file
        fileID = open("system/snappyHexMeshDict","w")

        # Add the header
        ofheader = self.headerOF("dictionary","system","snappyHexMeshDict")
        fileID.write(ofheader)

        # Include the constant file
        fileID.write('#include\t"../constantsFile"\n\n')
        # Which of the steps to run
        fileID.write('castellatedMesh\ttrue;\n\n')
        fileID.write('snap\ttrue;\n\n')
        fileID.write('addLayers\tfalse;\n\n')
        # Geometry. Definition of all surfaces. 
        fileID.write('geometry\n{\n\t')
        fileID.write('Front.stl {type triSurfaceMesh; name Front;}\n\t')
        fileID.write('Back.stl {type triSurfaceMesh; name Back;}\n\t')
        fileID.write('Top.stl {type triSurfaceMesh; name Top;}\n\t')
        fileID.write('Bottom.stl {type triSurfaceMesh; name Bottom;}\n\t')
        fileID.write('Left.stl {type triSurfaceMesh; name Left;}\n\t')
        fileID.write('Right.stl {type triSurfaceMesh; name Right;}\n')
        if flag == 1:
            fileID.write('\tBuilding.stl {type triSurfaceMesh; name Building;}\n')
        elif flag == 2:
            fileID.write('\tBuilding.stl {type triSurfaceMesh; name Building;}\n')
            fileID.write('\tOtherBuilding.stl {type triSurfaceMesh; name OtherBuilding;}\n')
        fileID.write('\tFull.stl {type triSurfaceMesh; name Full;}\n')
        fileID.write('};\n\n')

        # Castellated mesh generation
        fileID.write('castellatedMeshControls\n{\n\t')
        fileID.write('maxLocalCells\t$maxLocalCells;\n\t')
        fileID.write('maxGlobalCells\t$maxGlobalCells;\n\t')
        fileID.write('minRefinementCells\t10;\n\t')
        fileID.write('maxLoadUnbalance\t0.1;\n\t')
        fileID.write('nCellsBetweenLevels\t1;\n\n')

        # Explicit feature edge refinement
        fileID.write('\tfeatures\n\t(\n\t\t')
        fileID.write('{file "Front.eMesh"; level 3;}\n\t\t')
        fileID.write('{file "Back.eMesh"; level 3;}\n\t\t')
        fileID.write('{file "Top.eMesh"; level 3;}\n\t\t')
        fileID.write('{file "Bottom.eMesh"; level 3;}\n\t\t')
        fileID.write('{file "Left.eMesh"; level 3;}\n\t\t')
        fileID.write('{file "Right.eMesh"; level 3;}\n')
        if flag == 1:
            fileID.write('\t\t{file "Building.eMesh"; level 3;}\n')
        elif flag == 2:
            fileID.write('\t\t{file "Building.eMesh"; level 3;}\n')
            fileID.write('\t\t{file "OtherBuilding.eMesh"; level 3;}\n')
        fileID.write('\t);\n\n')

        # Surface based refinement
        fileID.write('\trefinementSurfaces\n\t{\n\t\t')
        fileID.write('Front {level (0 0);}\n\t\t')
        fileID.write('Back {level (0 0);}\n\t\t')
        fileID.write('Top {level (0 0);}\n\t\t')
        fileID.write('Bottom {level (2 2);}\n\t\t')
        fileID.write('Left {level (2 2);}\n\t\t')
        fileID.write('Right {level (2 2);}\n')
        if flag == 1:
            fileID.write('\t\tBuilding {level (2 2);}\n')
        elif flag == 2:
            fileID.write('\t\tBuilding {level (2 2);}\n')
            fileID.write('\t\tOtherBuilding {level (2 2);}\n')
        fileID.write('\t};\n\n')

        # Resolve sharp angles
        fileID.write('\tresolveFeatureAngle 80;\n\n')

        # Regional refinement 
        # This needs to be added and corrected
        fileID.write('\trefinementRegions\n\t{\n\t\t//Nothing here for now\n\t}\n\n')

        # Mesh selection
        fileID.write('\tlocationInMesh ($Inposx $Inposy $Inposz);\n\n')

        # Free-standring zone faces
        fileID.write('\tallowFreeStandingZoneFaces\tfalse;\n')
        fileID.write('}\n\n')

        # Snapping settings
        fileID.write('snapControls\n{\n\t')
        fileID.write('nSmoothPatch\t3;\n\t')
        fileID.write('tolerance\t4.0;\n\t')
        fileID.write('nSolveIter\t30;\n\t')
        fileID.write('nRelaxIter\t5;\n')
        fileID.write('}\n\n')

        # Settings for layer addition 
        # This is presently not being used
        fileID.write('addLayersControls\n{\n\t')
        fileID.write('relativeSizes\ttrue;\n\t')
        fileID.write('layers\n\t{\n\t')
        fileID.write('Bottom\n\t\t{nSurfaceLayers\t3;}\n\t')
        fileID.write('Left\n\t\t{nSurfaceLayers\t3;}\n\t')
        fileID.write('Right\n\t\t{nSurfaceLayers\t3;}\n\t}\n\n\t')
        fileID.write('expansionRatio\t1;\n\t')
        fileID.write('finalLayerThickness\t0.3;\n\t')
        fileID.write('minThickness\t0.1;\n\t')
        fileID.write('nGrow\t0;\n\t')

        # Advanced settings for layer addition
        fileID.write('featureAngle\t80;\n\t')
        fileID.write('nRelaxIter\t3;\n\t')
        fileID.write('nSmoothSurfaceNormals\t1;\n\t')
        fileID.write('nSmoothNormals\t3;\n\t')
        fileID.write('nSmoothThickness\t10;\n\t')
        fileID.write('maxFaceThicknessRatio\t0.5;\n\t')
        fileID.write('maxThicknessToMedialRatio\t0.3;\n\t')
        fileID.write('minMedianAxisAngle\t130;\n\t')
        fileID.write('nBufferCellsNoExtrude\t0;\n\t')
        fileID.write('nLayerIter\t50;\n')
        fileID.write('}\n\n')

        # Mesh quality settings
        fileID.write('meshQualityControls\n{\n\t')
        fileID.write('maxNonOrtho\t180;\n\t')
        fileID.write('maxBoundarySkewness\t20;\n\t')
        fileID.write('maxInternalSkewness\t4;\n\t')
        fileID.write('maxConcave\t80;\n\t')
        fileID.write('minFlatness\t0.5;\n\t')
        fileID.write('minVol\t1e-13;\n\t')
        fileID.write('minTetQuality\t1e-30;\n\t')
        fileID.write('minArea\t-1;\n\t')
        fileID.write('minTwist\t0.02;\n\t')
        fileID.write('minDeterminant\t0.001;\n\t')
        fileID.write('minFaceWeight\t0.02;\n\t')
        fileID.write('minVolRatio\t0.01;\n\t')
        fileID.write('minTriangleTwist\t-1;\n\t')
        fileID.write('nSmoothScale\t4;\n\t')
        fileID.write('errorReduction\t0.75;\n')
        fileID.write('}\n\n')

        # Advanced
        fileID.write('debug\t0;\n')
        fileID.write('mergeTolerance\t1E-6;\n')

    ####################################################################
    def setFieldsDictOF(self,numalpharegion):
        '''
        Method to create the setFields dictionary 

        Arguments
        -------------
            numalpharegion: Number of regions for alpha

        Variables
        -----------
            fileID: FileID for the fvSchemes file
            ofheader: Header for the Hydro-UQ input dictionary
        '''

        # Open the blockmeshDict file
        fileID = open("system/setFieldsDict","w")

        # Add the header
        ofheader = self.headerOF("dictionary","system","setFieldsDict")
        fileID.write(ofheader)

        # Include the constant file
        fileID.write('#include\t"../constantsFile"\n\n')

        # Initialize the global field value of alpha
        fileID.write('defaultFieldValues\n(\n\tvolScalarFieldValue\talpha.water\t$alphaglobal\n);\n\n')

        # Initialize the local field value of alpha
        fileinf = 'regions\n(\n'
        for ii in range(numalpharegion):
            fileinf = fileinf + '\tboxToCell\n\t{\n\t\t'
            fileinf = fileinf + 'box\t($alpha'+str(ii)+'x1\t$alpha'+str(ii)+'y1\t$alpha'+str(ii)+'z1)\t($alpha'+str(ii)+'x2\t$alpha'+str(ii)+'y2\t$alpha'+str(ii)+'z2);\n\n\t\t'
            fileinf = fileinf + 'fieldValues\n\t\t(\n\t\t\tvolScalarFieldValue\talpha.water\t$alpha' + str(ii) + '\n\t\t);\n\t}\n\n'
        fileID.write('%s' % (fileinf))
        fileID.write('\n);')

    ####################################################################
    def dynmeshFlumeOF(self):
        '''
        This method is used to write the dynamic mesh dictionary
        particularly aimed at the moving wall in the flume
        '''
        # Get the file ID
        fileID = self.openfileOF("constant/dynamicMeshDict","w")
        # Header
        ofheader = self.headerOF("dictionary","constant","dynamicMeshDict")
        fileID.write(ofheader)
        # Other data
        fileID.write('\ndynamicFvMesh\tdynamicMotionSolverFvMesh;\n\n')
        fileID.write('motionSolverLibs\t("libfvMotionSolvers.so");\n\n')
        fileID.write('solver\tdisplacementLaplacian;\n\n')
        fileID.write('displacementLaplacianCoeffs\n{\n\tdiffusivity uniform;\n}\n');
        # Close the file
        fileID.close()

    ####################################################################
    def wavemakerOF(self):
        '''
        This method is used to write the wavemaker dictionary
        particularly aimed at the moving wall in the flume
        '''
        # Get the file ID
        fileID = self.openfileOF("constant/wavemakerMovementDict","w")
        # Header
        ofheader = self.headerOF("dictionary","constant","wavemakerMovementDict")
        fileID.write(ofheader)
        # Other data
        fileID.write('\nreread\tfalse;\n\n')
        fileID.write('#include\t"wavemakerMovement.txt"\n')
        # Close the file
        fileID.close()

    ####################################################################
    def alpha0OF(self,flag):
        '''
        This method is used to write the alpha file for the 0-folder
        '''
        # Open the blockmeshDict file
        fileID = open("0.org/alpha.water","w")

        # Add the header
        ofheader = self.header2OF("volScalarField","alpha.water")
        fileID.write(ofheader)
        stlinfo = '{\ndimensions\t[0 0 0 0 0 0 0];\n\n'
        stlinfo = stlinfo + 'internalField\tuniform\t0;\n\n'
        stlinfo = stlinfo + 'boundaryField\n{\n\t'
        stlinfo = stlinfo + 'Front\n\t{\n\t\ttype\tzeroGradient;\n\t}\n\t'
        stlinfo = stlinfo + 'Top\n\t{\n\t\ttype\tinletOutlet;\n\t\t'
        stlinfo = stlinfo + 'inletValue\tuniform\t0;\n\t\t'
        stlinfo = stlinfo + 'value\tuniform\t0;\n\t}\n\t'
        stlinfo = stlinfo + 'Bottom\n\t{\n\t\ttype\tzeroGradient;\n\t}\n\t'
        stlinfo = stlinfo + 'Back\n\t{\n\t\ttype\tzeroGradient;\n\t}\n\t'
        stlinfo = stlinfo + 'Right\n\t{\n\t\ttype\tzeroGradient;\n\t}\n\t'
        stlinfo = stlinfo + 'Left\n\t{\n\t\ttype\tzeroGradient;\n\t}\n'
        if flag == 1:
            stlinfo = stlinfo + '\tBuilding\n\t{\n\t\ttype\tzeroGradient;\n\t}\n'
        elif flag == 2:
            stlinfo = stlinfo + '\tBuilding\n\t{\n\t\ttype\tzeroGradient;\n\t}\n'
            stlinfo = stlinfo + '\tOtherBuilding\n\t{\n\t\ttype\tzeroGradient;\n\t}\n'
        stlinfo = stlinfo + '\tdefault\n\t{\n\t\ttype\tnoSlip;\n\t}\n'
        stlinfo = stlinfo + '}\n'
        fileID.write('%s' % (stlinfo))

    ####################################################################
    def U0OF(self,flag):
        '''
        This method is used to write the U file for the 0-folder
        '''
        # Open the U-dof file
        fileID = open("0.org/U","w")

        # Add the header
        ofheader = self.header2OF("volVectorField","U")
        fileID.write(ofheader)
        stlinfo = '{\ndimensions\t[0 1 -1 0 0 0 0];\n\n'
        stlinfo = stlinfo + 'internalField\tuniform\t(0 0 0);\n\n'
        stlinfo = stlinfo + 'boundaryField\n{\n\t'
        stlinfo = stlinfo + 'Front\n\t{\n\t\ttype\tmovingWallVelocity;\n\t\t'
        stlinfo = stlinfo + 'value\tuniform\t(0 0 0);\n\t}\n\t'
        stlinfo = stlinfo + 'Top\n\t{\n\t\ttype\tpressureInletOutletVelocity;\n\t\t'
        stlinfo = stlinfo + 'value\tuniform\t(0 0 0);\n\t}\n\t'
        stlinfo = stlinfo + 'Bottom\n\t{\n\t\ttype\tnoSlip;\n\t}\n\t'
        stlinfo = stlinfo + 'Back\n\t{\n\t\ttype\tnoSlip;\n\t}\n\t'
        stlinfo = stlinfo + 'Right\n\t{\n\t\ttype\tnoSlip;\n\t}\n\t'
        stlinfo = stlinfo + 'Left\n\t{\n\t\ttype\tnoSlip;\n\t}\n'
        if flag == 1:
            stlinfo = stlinfo + '\tBuilding\n\t{\n\t\ttype\tnoSlip;\n\t}\n'
        elif flag == 2:
            stlinfo = stlinfo + '\tBuilding\n\t{\n\t\ttype\tnoSlip;\n\t}\n'
            stlinfo = stlinfo + '\tOtherBuilding\n\t{\n\t\ttype\tnoSlip;\n\t}\n'
        stlinfo = stlinfo + '\tdefault\n\t{\n\t\ttype\tnoSlip;\n\t}\n'
        stlinfo = stlinfo + '}\n'
        fileID.write('%s' % (stlinfo))


    ####################################################################
    def prgh0OF(self,flag):
        '''
        This method is used to write the p_rgh file for the 0-folder
        '''
        # Open the pressure-dof file
        fileID = open("0.org/p_rgh","w")

        # Add the header
        ofheader = self.header2OF("volScalarField","p_rgh")
        fileID.write(ofheader)
        stlinfo = '{\ndimensions\t[1 -1 -2 0 0 0 0];\n\n'
        stlinfo = stlinfo + 'internalField\tuniform\t0;\n\n'
        stlinfo = stlinfo + 'boundaryField\n{\n\t'
        stlinfo = stlinfo + 'Front\n\t{\n\t\ttype\tfixedFluxPressure;\n\t\t'
        stlinfo = stlinfo + 'value\tuniform\t0;\n\t}\n\t'
        stlinfo = stlinfo + 'Top\n\t{\n\t\ttype\ttotalPressure;\n\t\t'
        stlinfo = stlinfo + 'U\tU;\n\t\t'
        stlinfo = stlinfo + 'phi\tphi;\n\t\t'
        stlinfo = stlinfo + 'rho\trho;\n\t\t'
        stlinfo = stlinfo + 'psi\tnone;\n\t\t'
        stlinfo = stlinfo + 'gamma\t1;\n\t\t'
        stlinfo = stlinfo + 'p0\tuniform\t0;\n\t\t'
        stlinfo = stlinfo + 'value\tuniform\t0;\n\t}\n\t'
        stlinfo = stlinfo + 'Bottom\n\t{\n\t\ttype\tfixedFluxPressure;\n\t\t'
        stlinfo = stlinfo + 'value\tuniform\t0;\n\t}\n\t'
        stlinfo = stlinfo + 'Back\n\t{\n\t\ttype\tfixedFluxPressure;\n\t\t'
        stlinfo = stlinfo + 'value\tuniform\t0;\n\t}\n\t'
        stlinfo = stlinfo + 'Right\n\t{\n\t\ttype\tfixedFluxPressure;\n\t\t'
        stlinfo = stlinfo + 'value\tuniform\t0;\n\t}\n\t'
        stlinfo = stlinfo + 'Left\n\t{\n\t\ttype\tfixedFluxPressure;\n\t\t'
        stlinfo = stlinfo + 'value\tuniform\t0;\n\t}\n\t'
        if flag == 1:
            stlinfo = stlinfo + '\tBuilding\n\t{\n\t\ttype\tnoSlip;\n'
        elif flag == 2:
            stlinfo = stlinfo + '\tBuilding\n\t{\n\t\ttype\tnoSlip;\n'
            stlinfo = stlinfo + '\tOtherBuilding\n\t{\n\t\ttype\tnoSlip;\n'
        stlinfo = stlinfo + '\tdefault\n\t{\n\t\ttype\tfixedFluxPressure;\n\t\t'
        stlinfo = stlinfo + 'value\tuniform\t0;\n\t}\n'
        stlinfo = stlinfo + '}\n'
        fileID.write('%s' % (stlinfo))

    ####################################################################
    def pdisp0OF(self,flag):
        '''
        This method is used to write the U file for the 0-folder
        '''
        # Open the pointDisplacement-dof file
        fileID = open("0.org/pointDisplacement","w")

        # Add the header
        ofheader = self.headerOF("volVectorField","0.01","U")
        fileID.write(ofheader)
        stlinfo = '{\ndimensions\t[0 1 0 0 0 0 0];\n\n'
        stlinfo = stlinfo + 'internalField\tuniform\t(0 0 0);\n\n'
        stlinfo = stlinfo + 'boundaryField\n{\n\t'
        stlinfo = stlinfo + 'Front\n\t{\n\t\ttype\twavemakerMovement;\n\t\t'
        stlinfo = stlinfo + 'wavemakerDictName\twavemakerMovementDict;\n\t\t'
        stlinfo = stlinfo + 'value\tuniform\t(0 0 0);\n\t}\n\t'

        stlinfo = stlinfo + 'Top\n\t{\n\t\ttype\tfixedNormalSlip;\n\t\t'
        stlinfo = stlinfo + 'n\t(0 0 1);\n\t\t'
        stlinfo = stlinfo + 'value\tuniform\t(0 0 0);\n\t}\n\t'

        stlinfo = stlinfo + 'Bottom\n\t{\n\t\ttype\tfixedValue;\n\t\t'
        stlinfo = stlinfo + 'value\tuniform\t(0 0 0);\n\t}\n\t'

        stlinfo = stlinfo + 'Back\n\t{\n\t\ttype\tfixedValue;\n\t\t'
        stlinfo = stlinfo + 'value\tuniform\t(0 0 0);\n\t}\n\t'
        stlinfo = stlinfo + 'Right\n\t{\n\t\ttype\tfixedValue;\n\t\t'
        stlinfo = stlinfo + 'value\tuniform\t(0 0 0);\n\t}\n\t'
        stlinfo = stlinfo + 'Left\n\t{\n\t\ttype\tfixedValue;\n\t\t'
        stlinfo = stlinfo + 'value\tuniform\t(0 0 0);\n\t}\n\t'
        if flag == 1:
            stlinfo = stlinfo + 'Building\n\t{\n\t\ttype\tfixedValue;\n\t\t'
            stlinfo = stlinfo + 'value\tuniform\t(0 0 0);\n\t}\n\t'
        elif flag == 2:
            stlinfo = stlinfo + 'Building\n\t{\n\t\ttype\tfixedValue;\n\t\t'
            stlinfo = stlinfo + 'value\tuniform\t(0 0 0);\n\t}\n\t'
            stlinfo = stlinfo + 'OtherBuilding\n\t{\n\t\ttype\tfixedValue;\n\t\t'
            stlinfo = stlinfo + 'value\tuniform\t(0 0 0);\n\t}\n\t'
        stlinfo = stlinfo + '\tdefault\n\t{\n\t\ttype\tfixedValue;\n\t\t'
        stlinfo = stlinfo + 'value\tuniform\t(0 0 0);\n\t}\n'
        stlinfo = stlinfo + '}\n'
        fileID.write('%s' % (stlinfo))

    ####################################################################
    def wavemakerfile(self):
        '''
        This method is used to write the wavemaker movement file
        '''
        # Open the wavemaker movement file
        fileID = open("constant/wavemakerMovement.txt","w")
        fileID.write('wavemakerType\tPiston;\n')
        fileID.write('tSmooth\t1.5;\n')
        fileID.write('genAbs\t0;\n\n')

        # Create the wavemaker movement file
        # Get the frequency of the wavemaker
        frequency = 0
        waterdepth = 0
        filewm = open('wmdisp.txt','r')
        Lines = filewm.readlines()
        count = 0
        for line in Lines:
            count += 1
            if count == 37:
                stra=line.replace('% SampleRate: ','')
                stra2=stra.replace(' Hz','')
                frequency = 1/float(stra2)
                break
        count = 0
        for line in Lines:
            count += 1
            if count == 61:
                stra=line.replace('% StillWaterDepth: ','')
                waterdepth = float(stra)
                break

        # Count the number of lines
        countlines = sum(1 for line in open('wmdisp.txt')) - 72

        # Create the timeseries
        time = 0
        fileID.write('timeSeries\t'+str(countlines)+'(\n')
        for ii in range(countlines):
            fileID.write(str(time)+'\n')
            time = time + frequency
        fileID.write(');\n\n')

        # Create the paddle position
        fileID.write('paddlePosition 1(\n'+str(countlines)+'(\n')
        count = 0
        for line in Lines:
            count += 1
            if count > 72:
                data = float(line)
                fileID.write(str(data)+'\n')
        fileID.write(')\n);\n\n')

        # Write the paddle Eta
        fileID.write('paddleEta 1(\n'+str(countlines)+'(\n')
        filewmg = open('wmwg.txt','r')
        Lines2 = filewmg.readlines()
        count = 0
        for line in Lines2:
            count += 1
            if count > 72:
                data = float(line)+waterdepth
                fileID.write(str(data)+'\n')
        fileID.write(')\n);')

    ####################################################################
    def dircreate(self):
        '''
        Method creates all the necessary directories for the solver being used.

        Variables
        -----------
            foldwritten: Folders being created
        '''
        # Create directories for openfoam dictionaries
        # Access: Only owner can read and write
        access_rights = 0o700

        # Create 0-directory
        if(os.path.exists("0.org")):
            shutil.rmtree("0.org")
            os.mkdir("0.org",access_rights)
        else:
            os.mkdir("0.org",access_rights)

        #Create constant-directory
        if(os.path.exists("constant")):
            shutil.rmtree("constant")
            os.mkdir("constant",access_rights)
        else:
            os.mkdir("constant",access_rights)

        # Create the triSurface directory
        if(os.path.exists("constant/triSurface")):
            shutil.rmtree("constant/triSurface")
            os.mkdir("constant/triSurface",access_rights)
        else:
            os.mkdir("constant/triSurface",access_rights)

        #Create system-directory
        if(os.path.exists("system")):
            shutil.rmtree("system")
            os.mkdir("system",access_rights)
        else:
            os.mkdir("system",access_rights)

        # Folders written: required for log files
        foldwritten = np.array(['0.org','system','constant','constant/triSurface'])
        return foldwritten

    ####################################################################
    def filecreate(self,data,fpath):
        '''
        Method creates all necessary supplementary files required by the solver.

        Variables
        -----------
            filewritten: Files being created
            cnstfile: File pointer for the constants file
            fpath: Path to the dakota.json folder
        '''

        # Initialize the constant file
        self.cnstfile = self.openfileOF("constantsFile","w")
      
        # Write the gravity-file
        self.gfileOF(data)

        # Write the fvSchemes (interpolation schemes)
        self.fvSchemesOF(data)

        # Write the fvSolution (variable solvers)
        self.fvSolutionOF(data)

        # Unzip wavemaker file if required
        self.OSUwavemakerfileunzip(data,fpath)

        # Files written: required for log files
        filewritten = np.array(['g','fvScheme','fvSolution'])

        return filewritten

    ####################################################################
    def matmodel(self,data):
        '''
        Method creates all necessary files related to the material model that is used

        Variables
        ------------
            filewritten: Files being created
        '''

        # Write the transportProperties file
        self.transportProperties(data)

        # Write the turbulenceProperties file
        self.turbProperties(data)

        # Files written: required for log files
        filewritten = np.array(['transportProperties','turbulenceProperties'])

        return filewritten

    ####################################################################
    def solvecontrol(self,data):
        '''
        Method creates all necessary files for the control of program - here OpenFOAM

        Variables
        ------------
            filewritten: Files being created
        '''
        
        # Write the controlDict file
        self.controlDict(data)

        # Files written: required for log files
        filewritten = np.array(['controlDict'])

        return filewritten

    ####################################################################
    def parallel(self,data):
        '''
        Method creates the necessary file for decomposition of the domain

        Variables
        ------------
            filewritten: Files being created
        '''
        
        # Write the controlDict file
        self.decomposepar(data)

        # Files written: required for log files
        filewritten = np.array(['decomposeParDict'])

        return filewritten

    ####################################################################
    def geometry(self,data):
        '''
        Method creates the relevant STL files from the bathymetry
        definitions provided by the user

        Variables
        ------------
            filewritten: Files being created
            hydroutil: Utilities object
            simtype: Type of simulation
            bathyfiletype: Type of bathymetry file (if applicable)
            flumedeftype: Definition of wave flume (if applicable)
            breadth: Breadth of flume (if applicable)
        '''
        
        # Get an object for utilities
        hydroutil = genUtilities()

        # Get the simulation type
        simtype = ', '.join(hydroutil.extract_element_from_json(data, ["Events","SimulationType"]))

        # If SW-CFD coupling
        if (int(simtype) == 1) or (int(simtype) == 3):
            # Get the type of bathymetry
            bathyfiletype = ', '.join(hydroutil.extract_element_from_json(data, ["Events","BathymetryFileType"]))

            # Create an object for the bathymetry 
            if int(bathyfiletype) == 0:
                # Create object for simcenter format
                print(0)

            elif ((int(bathyfiletype) == 1) or (int(bathyfiletype) == 2)) \
                or (int(bathyfiletype) == 3):
                # Create the GeoClaw object
                print(0)

            elif int(bathyfiletype) == 4:
                # Create the AdCirc object
                # Files written: required for log files
                #filewritten = np.array(['ERROR: AdCirc not supported. Contact developer.'])
                #return filewritten
                print(0)

        elif int(simtype) == 4:
            # Get the way the flume is defined
            flumedeftype = ', '.join(hydroutil.extract_element_from_json(data, ["Events","FlumeInfoType"]))

            if int(flumedeftype) == 0:
                # Create the object for the Flume
                flume = OSUWaveFlume()

                # Read flume data from the JSON file and
                # Create a FlumeData.txt file
                flumesegs = ', '.join(hydroutil.extract_element_from_json(data, ["Events","FlumeSegments"]))

                # Separate to numbers
                # Find total number of processors
                flumesegs = flumesegs.replace(',', ' ')
                nums = [int(n) for n in flumesegs.split()]
                #totalprocs = nums[0]*nums[1]*nums[2]
                for ii in range(nums[0]):
                    f = open("FlumeData.txt", "a")
                    f.write(str(nums[2*ii+1]) + ',' + str(nums[2*ii+2]) + '\n' )
                    f.close()

                # Initialize the input file
                IpPTFile = 'FlumeData.txt'
                # Breadth of the flume
                breadth = ''.join(hydroutil.extract_element_from_json(data, ["Events","FlumeBreadth"]))
                flume.breadth = float(breadth)
                # Generate the flume STL files
                extreme = flume.generateflume(IpPTFile)
                # Write the Max-Min values for the blockMesh
                # into the constants file
                BMXmin = extreme[0] - 0.25*(extreme[1] - extreme[0])
                BMXmax = extreme[1] + 0.25*(extreme[1] - extreme[0])
                BMYmin = -0.625*flume.breadth
                BMYmax = 0.625*flume.breadth
                BMZmin = extreme[2] - 0.25*(extreme[3] - extreme[2])
                BMZmax = extreme[3] + 0.25*(extreme[3] - extreme[2])
                var = np.array([['BMXmin', BMXmin], ['BMXmax', BMXmax], ['BMYmin', BMYmin],['BMYmax', BMYmax], ['BMZmin', BMZmin], ['BMZmax', BMZmax]])
                self.constvarfileOF(var,"blockMeshDict")
                # Move the STL files
                shutil.move("Front.stl", "constant/triSurface/Front.stl")
                shutil.move("Back.stl", "constant/triSurface/Back.stl")
                shutil.move("Left.stl", "constant/triSurface/Left.stl")
                shutil.move("Right.stl", "constant/triSurface/Right.stl")
                shutil.move("Top.stl", "constant/triSurface/Top.stl")
                shutil.move("Bottom.stl", "constant/triSurface/Bottom.stl")

                # Get information about building
                flag = 0

                # Write extreme values to temporary file for later usage
                tempfileID = open("temp_geometry","w")
                tempfileID.write(str(BMXmin)+"\n"+str(BMXmax)+"\n"+str(BMYmin)+"\n"+str(BMYmax)+"\n"+str(BMZmin)+"\n"+str(BMZmax)+"\n"+str(flag)+"\n")
                tempfileID.close

                # Files written: required for log files
                filewritten = np.array(['STL files'])
                return filewritten

            else:
                # Error in simulation type. Return error
                filewritten = np.array(['ERROR: Flume info type not supported. Contact developer.'])
                return filewritten
        
        else:
            filewritten = np.array(['ERROR: Simulation type not supported. Contact developer.'])
            return filewritten

        # Files written: required for log files
        filewritten = np.array(['None'])
        return filewritten

    ####################################################################
    def meshing(self,data):
        '''
        Method creates the relevant mesh files / dictionaries 

        Variables
        ------------
            filewritten: Files being created
            meshsize: Goes from 1 - 5 (Points to slider on UI)
            data_geoext: Data from temporary geometry file
            nx, ny, nz: Number of elements in x-, y- and z- directions
            flag: Indicates presence of one or more buildings
            maxLocalCells: Number of local cells for SHM
            maxGlobalCells: Number of global cells for SHM
            px,py,pz: A point inside the domain
        '''
        # Get an object for utilities
        hydroutil = genUtilities()

        # Get required data
        meshsize = ''.join(hydroutil.extract_element_from_json(data, ["Events","MeshSize"]))

        # Read the temporary geometry file with extreme values
        data_geoext = np.genfromtxt("temp_geometry", dtype=(float))

        # Get the mesh sizes
        nx = 100*int(meshsize)
        if( abs(data_geoext[1] - data_geoext[0]) > 0.000001):
            ny = math.ceil(5*nx*((data_geoext[3]-data_geoext[2])/(data_geoext[1]-data_geoext[0])))
            nz = math.ceil(5*nx*((data_geoext[5]-data_geoext[4])/(data_geoext[1]-data_geoext[0])))

        else:
            filewritten = np.array(['Error in length definition of geometry (max = min). Please check geometry'])
            return filewritten

        # Write the constants to the file
        var = np.array([['nx', nx], ['ny', ny], ['nz', nz]])
        self.constvarfileOF(var,"blockMeshDict")

        # Create the blockMeshDict
        self.bMeshDictOF()

        # Create the surface feature extract
        flag = int(data_geoext[6])
        self.surffeatureextDictOF(flag)

        # Read the refinement regions
        # For now this is not considered
        # This should be added in next update

        # Get required data related to mesh
        maxLocalCells = int(meshsize)*2000000
        maxGlobalCells = int(meshsize)*10000000
        px = 0.5*(data_geoext[1]+data_geoext[0])
        py = 0.5*(data_geoext[3]+data_geoext[2])
        pz = 0.5*(data_geoext[5]+data_geoext[4])

        # Write the constants to the file
        var = np.array([['maxLocalCells', str(maxLocalCells)], ['maxGlobalCells', str(maxGlobalCells)],['Inposx', str(px)],['Inposy', str(py)],['Inposz', str(pz)]])
        self.constvarfileOF(var,"snappyHexMeshDict")

        # Create the snappyHexMesh
        self.snappyhexmeshDictOF(flag)

        # Files written: required for log files
        filewritten = np.array(['blockMeshDict','surfaceFeatureExtractDict','snappyHexMeshDict'])
        return filewritten

    ####################################################################
    def initcond(self,data):
        '''
        Method creates the relevant files for the initial condition

        Variables
        ------------
            filewritten: Files being created
            
        '''

        # Get an object for utilities
        hydroutil = genUtilities()

        # Get the simulation type
        simtype = ', '.join(hydroutil.extract_element_from_json(data, ["Events","SimulationType"]))

        if int(simtype) == 1:
            filewritten = np.array(['ERROR: Initialization from SW solutions. Contact developer for help.'])
            return filewritten

        elif int(simtype) == 3:
            # Get the global value of alpha
            alphaglobal = ', '.join(hydroutil.extract_element_from_json(data, ["Events","InitialAlphaGlobal"]))
            var = np.array([['alphaglobal',str(alphaglobal)]])

            # Get each local value 
            numalpharegion = ', '.join(hydroutil.extract_element_from_json(data, ["Events","NumAlphaRegion"]))

            # Get each local region
            for ii in range(int(numalpharegion)):
                regionalpha = ', '.join(hydroutil.extract_element_from_json(data, ["Events","InitialAlphaRegion"+str(ii)]))
                regions = regionalpha.replace(',', ' ')
                nums = [int(n) for n in regions.split()]
                var = np.append(var,[['alpha'+str(ii)+'x1',str(nums[0])]],axis=0)
                var = np.append(var,[['alpha'+str(ii)+'y1',str(nums[1])]],axis=0)
                var = np.append(var,[['alpha'+str(ii)+'z1',str(nums[2])]],axis=0)
                var = np.append(var,[['alpha'+str(ii)+'x2',str(nums[3])]],axis=0)
                var = np.append(var,[['alpha'+str(ii)+'y2',str(nums[4])]],axis=0)
                var = np.append(var,[['alpha'+str(ii)+'z2',str(nums[5])]],axis=0)
                var = np.append(var,[['alpha'+str(ii),str(nums[6])]],axis=0)

            # Write the constants file
            self.constvarfileOF(var,"setFieldsDict")

            # Create the setFields dictionary
            self.setFieldsDictOF(int(numalpharegion))
            filewritten = np.array(['setFieldsDict'])
            return filewritten

        elif int(simtype) == 4:
            # Initialize water depth
            waterdepth = 0
            # Check first if flume file exists.
            if os.path.isfile("wmwg.txt"):
                # Get water height from this
                # Read the temporary geometry file with extreme values
                filewm = open('wmwg.txt','r')
                Lines = filewm.readlines()
                count = 0
                for line in Lines:
                    count += 1
                    if count == 61:
                        stra=line.replace('% StillWaterDepth: ','')
                        waterdepth = float(stra)
                        break

            elif os.path.isfile("wmdisp.txt"):
                # Read the temporary geometry file with extreme values
                filewm = open('wmdisp.txt','r')
                Lines = filewm.readlines()
                count = 0
                for line in Lines:
                    count += 1
                    if count == 61:
                        stra=line.replace('% StillWaterDepth: ','')
                        waterdepth = float(stra)
                        break

            # If water depth has been read, then 
            if waterdepth > 0:
                # Initialize global value to zero
                var = np.array([['alphaglobal',str(0)]])
                numalpharegion = '1'

                # Read the temp geometry file
                data_geoext = np.genfromtxt("temp_geometry", dtype=(float))

                # Set the variables for one region
                var = np.append(var,[['alpha'+str(0)+'x1',str(data_geoext[0])]],axis=0)
                var = np.append(var,[['alpha'+str(0)+'y1',str(data_geoext[2])]],axis=0)
                var = np.append(var,[['alpha'+str(0)+'z1',str(data_geoext[4])]],axis=0)
                var = np.append(var,[['alpha'+str(0)+'x2',str(data_geoext[1])]],axis=0)
                var = np.append(var,[['alpha'+str(0)+'y2',str(data_geoext[3])]],axis=0)
                var = np.append(var,[['alpha'+str(0)+'z2',str(data_geoext[5])]],axis=0)
                var = np.append(var,[['alpha'+str(0),str(1)]],axis=0)

            else:
                # Get the global value of alpha
                alphaglobal = ', '.join(hydroutil.extract_element_from_json(data, ["Events","InitialAlphaGlobal"]))
                var = np.array([['alphaglobal',str(alphaglobal)]])

                # Get each local value 
                numalpharegion = ', '.join(hydroutil.extract_element_from_json(data, ["Events","NumAlphaRegion"]))

                # Get each local region
                for ii in range(int(numalpharegion)):
                    regionalpha = ', '.join(hydroutil.extract_element_from_json(data, ["Events","InitialAlphaRegion"+str(ii)]))
                    regions = regionalpha.replace(',', ' ')
                    nums = [int(n) for n in regions.split()]
                    var = np.append(var,[['alpha'+str(ii)+'x1',str(nums[0])]],axis=0)
                    var = np.append(var,[['alpha'+str(ii)+'y1',str(nums[1])]],axis=0)
                    var = np.append(var,[['alpha'+str(ii)+'z1',str(nums[2])]],axis=0)
                    var = np.append(var,[['alpha'+str(ii)+'x2',str(nums[3])]],axis=0)
                    var = np.append(var,[['alpha'+str(ii)+'y2',str(nums[4])]],axis=0)
                    var = np.append(var,[['alpha'+str(ii)+'z2',str(nums[5])]],axis=0)
                    var = np.append(var,[['alpha'+str(ii),str(nums[6])]],axis=0)

            # Write the constants file
            self.constvarfileOF(var,"setFieldsDict")

            # Create the setFields dictionary
            self.setFieldsDictOF(int(numalpharegion))
            filewritten = np.array(['setFieldsDict'])
            return filewritten

        # Files written: required for log files
        filewritten = np.array(['ERROR: Simulation type likely not supported. Contact developer'])
        return filewritten

    ####################################################################
    def bouncond(self,data):
        '''
        Method creates the relevant files for the initial condition

        Variables
        ------------
            filewritten: Files being created
            
        '''

        # Get an object for utilities
        hydroutil = genUtilities()

        # Get the simulation type
        simtype = ', '.join(hydroutil.extract_element_from_json(data, ["Events","SimulationType"]))

        if int(simtype) == 4:

            # Get the building flag
            data_geoext = np.genfromtxt("temp_geometry", dtype=(float))
            flag = int(data_geoext[6])

            # Write the dynamic mesh dictionary for the moving wall
            self.dynmeshFlumeOF()

            # Write the dynamic mesh dictionary for the moving wall
            self.wavemakerOF()

            # Write the  files for the 0-folder
            self.alpha0OF(flag)
            self.U0OF(flag)
            self.prgh0OF(flag)
            self.pdisp0OF(flag)

            # Write the wavemaker file
            self.wavemakerfile()

            # Give the confirmation that the files have been created
            filewritten = np.array(['0.org/alpha.water','0.org/U','0.org/p_rgh','0.org/point_Displacement','wavemakerMovementDict','dynamicMeshDict','waveMakerMovement.txt'])
            return filewritten

        else:
            filewritten = np.array(['ERROR: Initialization from boundary conditions. Contact developer for help.'])
            return filewritten

        # Files written: required for log files
        filewritten = np.array(['ERROR: Likely your boundary condition is not yet supported. Contact developer'])
        return filewritten