####################################################################
# OpenFOAM class for Hydro-UQ
####################################################################
import numpy as np
import os
import shutil
from GenUtilities import genUtilities # General utilities

class solver(object):
    '''
    This class includes all the general utilities that are
    required for the Hydro-UQ.
    
    Primary methods
    ----------------
        dircreate: Method to create necessary directories for the solver
        filecreate: Method to create supplementary files

    Secondary methods
    -------------------
        gfileOF: Method creates the gravity file for the OpenFOAM
        fvSchemesOF: Method to create the fvSchemes file
        fvSolutionOF: Method to create the fvSolution file
    
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
    def filecreate(self,data):
        '''
        Method creates all necessary supplementary files required by the solver.

        Variables
        -----------
            filewritten: Files being created
            cnstfile: File pointer for the constants file
        '''

        # Initialize the constant file
        self.cnstfile = self.openfileOF("constantsFile","w")
      
        # Write the gravity-file
        self.gfileOF(data)

        # Write the fvSchemes (interpolation schemes)
        self.fvSchemesOF(data)

        # Write the fvSolution (variable solvers)
        self.fvSolutionOF(data)

        # Files written: required for log files
        filewritten = np.array(['g','fvScheme','fvSolution'])

        return filewritten

