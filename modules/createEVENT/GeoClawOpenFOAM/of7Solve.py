####################################################################  # noqa: INP001
# LICENSING INFORMATION
####################################################################
"""LICENSE INFORMATION:

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

"""  # noqa: E501, D400, D415
####################################################################
# AUTHOR INFORMATION
####################################################################
# 2020 - 2021: Ajay B Harish (ajaybh@berkeley.edu)

####################################################################
# Import all necessary modules
####################################################################
# Standard python modules

# Other custom modules
from hydroUtils import hydroUtils


####################################################################
# OpenFOAM7 solver class
####################################################################
class of7Solve:  # noqa: N801
    """This class includes the methods related to
    solver for openfoam7.

    Methods
    -------
            fvSchemetext: Get all the text for the fvSchemes

    """  # noqa: D205, D404

    #############################################################
    def solverheader(self, fileobjec):  # noqa: ANN001, ANN201
        """Creates the text for the header

        Variable
        -----------
                header: Header for the solver-files
        """  # noqa: D400, D401, D415
        header = (
            """/*--------------------------*- NHERI SimCenter -*----------------------------*\\ 
|	   | H |
|	   | Y | HydroUQ: Water-based Natural Hazards Modeling Application
|======| D | Website: simcenter.designsafe-ci.org/research-tools/hydro-uq
|	   | R | Version: 1.00
|	   | O |
\\*---------------------------------------------------------------------------*/ 
FoamFile
{\n\tversion\t2.0;\n\tformat\tascii;\n\tclass\tdictionary;\n\tlocation\t"system";\n\tobject\t"""  # noqa: E501, W291
            + fileobjec
            + """;\n}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"""  # noqa: E501
        )

        # Return the header for U file
        return header  # noqa: RET504

    #############################################################
    def fvSchemetext(self, data):  # noqa: ANN001, ANN201, ARG002, N802
        """Creates the necessary text for fvSchemes for openfoam7

        Arguments:
        ---------
                data: all the JSON data

        """  # noqa: D400, D401, D415
        # Get the header text for the U-file
        fvSchemetext = self.solverheader('fvSchemes')  # noqa: N806

        # Add all other items
        # ddt
        fvSchemetext = fvSchemetext + 'ddtSchemes\n{\n\tdefault\tEuler;\n}\n\n'  # noqa: N806

        # grad
        fvSchemetext = fvSchemetext + 'gradSchemes\n{\n\tdefault\tGauss linear;\n}\n'  # noqa: N806

        # div
        fvSchemetext = fvSchemetext + '\ndivSchemes\n{\n\t'  # noqa: N806
        fvSchemetext = fvSchemetext + 'div(rhoPhi,U)\tGauss limitedLinearV 1;\n\t'  # noqa: N806
        fvSchemetext = fvSchemetext + 'div(U)\tGauss linear;\n\t'  # noqa: N806
        fvSchemetext = (  # noqa: N806
            fvSchemetext
            + 'div((rhoPhi|interpolate(porosity)),U)\tGauss limitedLinearV 1;\n\t'
        )
        fvSchemetext = (  # noqa: N806
            fvSchemetext + 'div(rhoPhiPor,UPor)\tGauss limitedLinearV 1;\n\t'
        )
        fvSchemetext = fvSchemetext + 'div(rhoPhi,UPor)\tGauss limitedLinearV 1;\n\t'  # noqa: N806
        fvSchemetext = fvSchemetext + 'div(rhoPhiPor,U)\tGauss limitedLinearV 1;\n\t'  # noqa: N806
        fvSchemetext = fvSchemetext + 'div(phi,alpha)\tGauss vanLeer;\n\t'  # noqa: N806
        fvSchemetext = (  # noqa: N806
            fvSchemetext + 'div(phirb,alpha)\tGauss interfaceCompression;\n\t'
        )
        fvSchemetext = (  # noqa: N806
            fvSchemetext + 'div((muEff*dev(T(grad(U)))))\tGauss linear;\n\t'
        )
        fvSchemetext = fvSchemetext + 'div(phi,k)\tGauss upwind;\n\t'  # noqa: N806
        fvSchemetext = fvSchemetext + 'div(phi,epsilon)\tGauss upwind;\n\t'  # noqa: N806
        fvSchemetext = (  # noqa: N806
            fvSchemetext + 'div((phi|interpolate(porosity)),k)\tGauss upwind;\n\t'
        )
        fvSchemetext = (  # noqa: N806
            fvSchemetext + 'div((phi*interpolate(rho)),k)\tGauss upwind;\n\t'
        )
        fvSchemetext = (  # noqa: N806
            fvSchemetext
            + 'div((phi|interpolate(porosity)),epsilon)\tGauss upwind;\n\t'
        )
        fvSchemetext = fvSchemetext + 'div(phi,omega)\tGauss upwind;\n\t'  # noqa: N806
        fvSchemetext = (  # noqa: N806
            fvSchemetext
            + 'div((phi|interpolate(porosity)),omega)\tGauss upwind;\n\t'
        )
        fvSchemetext = (  # noqa: N806
            fvSchemetext + 'div((phi*interpolate(rho)),omega)\tGauss upwind;\n\t'
        )
        fvSchemetext = (  # noqa: N806
            fvSchemetext + 'div((phi*interpolate(rho)),epsilon)\tGauss upwind;\n'
        )
        fvSchemetext = fvSchemetext + '}\n\n'  # noqa: N806

        # Laplacian
        fvSchemetext = (  # noqa: N806
            fvSchemetext
            + 'laplacianSchemes\n{\n\tdefault\tGauss linear corrected;\n}\n\n'
        )

        # interpolation
        fvSchemetext = (  # noqa: N806
            fvSchemetext + 'interpolationSchemes\n{\n\tdefault\tlinear;\n}\n\n'
        )

        # snGrad
        fvSchemetext = (  # noqa: N806
            fvSchemetext + 'snGradSchemes\n{\n\tdefault\tcorrected;\n}\n\n'
        )

        # flux
        fvSchemetext = fvSchemetext + 'fluxRequired\n{\n\t'  # noqa: N806
        fvSchemetext = fvSchemetext + 'default\tno;\n\t'  # noqa: N806
        fvSchemetext = fvSchemetext + 'p_rgh;\n\t'  # noqa: N806
        fvSchemetext = fvSchemetext + 'pcorr;\n\t'  # noqa: N806
        fvSchemetext = fvSchemetext + 'alpha.water;\n'  # noqa: N806
        fvSchemetext = fvSchemetext + '}\n'  # noqa: N806

        return fvSchemetext  # noqa: RET504

    #############################################################
    def fvSolntext(self, data):  # noqa: ANN001, ANN201, N802, PLR0915
        """Creates the necessary text for fvSolution for openfoam7

        Arguments:
        ---------
                data: all the JSON data

        """  # noqa: D400, D401, D415
        # Create a utilities object
        hydroutil = hydroUtils()

        # Get the simulation type
        simtype = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'SimulationType'])
        )

        # Get the turbulence model
        turb = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'TurbulenceModel'])
        )

        # Get the header text for the U-file
        fvSolntext = self.solverheader('fvSolution')  # noqa: N806

        # Other data
        fvSolntext = fvSolntext + 'solvers\n{\n\t'  # noqa: N806

        # solvers: alpha  # noqa: ERA001
        fvSolntext = fvSolntext + '"alpha.water.*"\n\t{\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'nAlphaCorr\t1;\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'nAlphaSubCycles\t2;\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'alphaOuterCorrectors\tyes;\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'cAlpha\t1;\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'MULESCorr\tno;\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'nLimiterIter\t3;\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'solver\tsmoothSolver;\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'smoother\tsymGaussSeidel;\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'tolerance\t1e-08;\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'relTol\t0;\n\t}\n\n\t'  # noqa: N806

        # solvers: pcorr  # noqa: ERA001
        fvSolntext = fvSolntext + '"pcorr.*"\n\t{\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'solver\tPCG;\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'preconditioner\tDIC;\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'tolerance\t1e-05;\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'relTol\t0;\n\t}\n\n\t'  # noqa: N806

        # solvers: pcorrFinal  # noqa: ERA001
        fvSolntext = fvSolntext + 'pcorrFinal\n\t{\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + '$pcorr;\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'relTol\t0;\n\t}\n\n\t'  # noqa: N806

        # solvers: p_rgh  # noqa: ERA001
        fvSolntext = fvSolntext + 'p_rgh\n\t{\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'solver\tPCG;\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'preconditioner\tDIC;\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'tolerance\t1e-07;\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'relTol\t0.05;\n\t}\n\n\t'  # noqa: N806

        # solvers: p_rghFinal  # noqa: ERA001
        fvSolntext = fvSolntext + 'p_rghFinal\n\t{\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + '$p_rgh;\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'relTol\t0;\n\t}\n\n\t'  # noqa: N806

        # solvers: U  # noqa: ERA001
        fvSolntext = fvSolntext + 'U\n\t{\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'solver\tsmoothSolver;\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'smoother\tsymGaussSeidel;\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'tolerance\t1e-06;\n\t\t'  # noqa: N806
        fvSolntext = fvSolntext + 'relTol\t0;\n\t}\n'  # noqa: N806

        # Turbulece variables (if exist)
        if (int(turb) == 1) or (int(turb) == 2):  # noqa: PLR2004
            fvSolntext = fvSolntext + '\n\t'  # noqa: N806
            fvSolntext = fvSolntext + '"(k|epsilon|omega|B|nuTilda).*"\n\t{\n\t\t'  # noqa: N806
            fvSolntext = fvSolntext + 'solver\tsmoothSolver;\n\t\t'  # noqa: N806
            fvSolntext = fvSolntext + 'smoother\tsymGaussSeidel;\n\t\t'  # noqa: N806
            fvSolntext = fvSolntext + 'tolerance\t1e-08;\n\t\t'  # noqa: N806
            fvSolntext = fvSolntext + 'relTol\t0;\n\t}\n'  # noqa: N806

        # solvers: cellDisplacement (for flume)
        if int(simtype) == 4:  # noqa: PLR2004
            # solvers: cellDisplacement (for flume)
            fvSolntext = fvSolntext + '\n\t'  # noqa: N806
            fvSolntext = fvSolntext + 'cellDisplacement\n\t{\n\t\t'  # noqa: N806
            fvSolntext = fvSolntext + 'solver\tGAMG;\n\t\t'  # noqa: N806
            fvSolntext = fvSolntext + 'tolerance\t1e-05;\n\t\t'  # noqa: N806
            fvSolntext = fvSolntext + 'relTol\t0;\n\t\t'  # noqa: N806
            fvSolntext = fvSolntext + 'smoother\tGaussSeidel;\n\t\t'  # noqa: N806
            fvSolntext = fvSolntext + 'cacheAgglomeration\tfalse;\n\t\t'  # noqa: N806
            fvSolntext = fvSolntext + 'nCellsInCoarsestLevel\t10;\n\t\t'  # noqa: N806
            fvSolntext = fvSolntext + 'agglomerator\tfaceAreaPair;\n\t\t'  # noqa: N806
            fvSolntext = fvSolntext + 'mergeLevels\t1;\n\t}\n\n\t'  # noqa: N806

            # solvers: cellDisplacementFinal(for flume)
            fvSolntext = fvSolntext + 'cellDisplacementFinal\n\t{\n\t\t'  # noqa: N806
            fvSolntext = fvSolntext + '$cellDisplacement;\n\t\t'  # noqa: N806
            fvSolntext = fvSolntext + 'relTol\t0;\n\t}\n'  # noqa: N806

        # Close solvers
        fvSolntext = fvSolntext + '}\n\n'  # noqa: N806

        # PIMPLE
        fvSolntext = fvSolntext + 'PIMPLE\n{\n\t'  # noqa: N806
        fvSolntext = fvSolntext + 'momentumPredictor\tno;\n\t'  # noqa: N806
        fvSolntext = fvSolntext + 'nOuterCorrectors\t1;\n\t'  # noqa: N806
        fvSolntext = fvSolntext + 'nCorrectors\t3;\n\t'  # noqa: N806
        fvSolntext = fvSolntext + 'nNonOrthogonalCorrectors\t0;\n}\n\n'  # noqa: N806

        # Relaxation factors
        fvSolntext = fvSolntext + 'relaxationFactors\n{\n\t'  # noqa: N806
        fvSolntext = fvSolntext + 'fields\n\t{\n\t}\n\t'  # noqa: N806
        fvSolntext = fvSolntext + 'equations\n\t{\n\t\t".*"\t1;\n\t}\n}'  # noqa: N806

        return fvSolntext  # noqa: RET504

    #############################################################
    def cdicttext(self, data):  # noqa: ANN001, ANN201
        """Creates the necessary text for controlDict for openfoam7

        Arguments:
        ---------
                data: all the JSON data

        """  # noqa: D400, D401, D415
        # Create a utilities object
        hydroutil = hydroUtils()

        # Get the header text for the U-file
        cdicttext = self.solverheader('controlDict')

        # Get the simulation type: Solver
        simtype = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'SimulationType'])
        )
        if int(simtype) == 4:  # noqa: PLR2004
            cdicttext = cdicttext + '\napplication \t olaDyMFlow;\n\n'
        else:
            cdicttext = cdicttext + '\napplication \t olaFlow;\n\n'

        # Check restart situation and give start time
        restart = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'Restart'])
        )
        if restart == 'Yes':
            cdicttext = cdicttext + 'startFrom \t latestTime;\n\n'
        elif restart == 'No':
            # Start time
            startT = ', '.join(  # noqa: N806
                hydroutil.extract_element_from_json(data, ['Events', 'StartTime'])
            )
            cdicttext = cdicttext + 'startFrom \t startTime;\n\n'
            cdicttext = cdicttext + 'startTime \t' + startT + ';\n\n'

        # End time
        endT = ', '.join(  # noqa: N806
            hydroutil.extract_element_from_json(data, ['Events', 'EndTime'])
        )
        cdicttext = cdicttext + 'stopAt \t endTime;\n\n'
        cdicttext = cdicttext + 'endTime \t' + endT + ';\n\n'

        # Time interval (modified file needs to be made later)
        cdicttext = cdicttext + 'deltaT \t 1;\n\n'

        # Write control
        cdicttext = cdicttext + 'writeControl \t adjustableRunTime;\n\n'

        # Write interval (modified file needs to be made later)
        cdicttext = cdicttext + 'writeInterval \t 1;\n\n'

        # All others
        cdicttext = cdicttext + 'purgeWrite \t 0;\n\n'
        cdicttext = cdicttext + 'writeFormat \t ascii;\n\n'
        cdicttext = cdicttext + 'writePrecision \t 6;\n\n'
        cdicttext = cdicttext + 'writeCompression \t uncompressed;\n\n'
        cdicttext = cdicttext + 'timeFormat \t general;\n\n'
        cdicttext = cdicttext + 'timePrecision \t 6;\n\n'
        cdicttext = cdicttext + 'runTimeModifiable \t yes;\n\n'
        cdicttext = cdicttext + 'adjustTimeStep \t yes;\n\n'
        cdicttext = cdicttext + 'maxCo \t 1.0;\n\n'
        cdicttext = cdicttext + 'maxAlphaCo \t 1.0;\n\n'
        cdicttext = cdicttext + 'maxDeltaT \t 1;\n\n'

        return cdicttext  # noqa: RET504

    #############################################################
    def cdictcheck(self, data):  # noqa: ANN001, ANN201
        """Creates the check for controlDict for openfoam7

        Arguments:
        ---------
                data: all the JSON data

        """  # noqa: D400, D401, D415
        # Create a utilities object
        hydroutil = hydroUtils()

        # Start time
        startT = hydroutil.extract_element_from_json(data, ['Events', 'StartTime'])  # noqa: N806
        if startT == [None]:
            return -1

        # End time
        endT = hydroutil.extract_element_from_json(data, ['Events', 'EndTime'])  # noqa: N806
        if endT == [None]:
            return -1

        # deltaT
        deltaT = hydroutil.extract_element_from_json(  # noqa: N806
            data, ['Events', 'TimeInterval']
        )
        if deltaT == [None]:
            return -1

        # WriteT
        writeT = hydroutil.extract_element_from_json(  # noqa: N806
            data, ['Events', 'WriteInterval']
        )
        if writeT == [None]:
            return -1

        # Return 0 if all available
        return 0

    #############################################################
    def cdictFtext(self, data):  # noqa: ANN001, ANN201, N802
        """Creates the necessary text for controlDict for openfoam7
        This is used for force computation with Dakota

        Arguments:
        ---------
                data: all the JSON data

        """  # noqa: D205, D400, D401, D415
        # Create a utilities object
        hydroutil = hydroUtils()

        # Get the header text for the U-file
        cdicttext = self.solverheader('controlDict')

        # Get the simulation type: Solver
        simtype = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'SimulationType'])
        )
        if int(simtype) == 4:  # noqa: PLR2004
            cdicttext = cdicttext + '\napplication \t olaDyMFlow;\n\n'
        else:
            cdicttext = cdicttext + '\napplication \t olaFlow;\n\n'

        # Check restart situation and give start time
        restart = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'Restart'])
        )
        if restart == 'Yes':
            cdicttext = cdicttext + 'startFrom \t latestTime;\n\n'
        elif restart == 'No':
            # Start time
            startT = ', '.join(  # noqa: N806
                hydroutil.extract_element_from_json(data, ['Events', 'StartTime'])
            )
            cdicttext = cdicttext + 'startFrom \t startTime;\n\n'
            cdicttext = cdicttext + 'startTime \t' + startT + ';\n\n'

        # End time
        endT = ', '.join(  # noqa: N806
            hydroutil.extract_element_from_json(data, ['Events', 'EndTime'])
        )
        cdicttext = cdicttext + 'stopAt \t endTime;\n\n'
        cdicttext = cdicttext + 'endTime \t' + endT + ';\n\n'

        # Time interval
        deltaT = ', '.join(  # noqa: N806
            hydroutil.extract_element_from_json(data, ['Events', 'TimeInterval'])
        )
        cdicttext = cdicttext + 'deltaT \t' + deltaT + ';\n\n'

        # Write control
        cdicttext = cdicttext + 'writeControl \t adjustableRunTime;\n\n'

        # Write interval
        writeT = ', '.join(  # noqa: N806
            hydroutil.extract_element_from_json(data, ['Events', 'WriteInterval'])
        )
        cdicttext = cdicttext + 'writeInterval \t' + writeT + ';\n\n'

        # All others
        cdicttext = cdicttext + 'purgeWrite \t 0;\n\n'
        cdicttext = cdicttext + 'writeFormat \t ascii;\n\n'
        cdicttext = cdicttext + 'writePrecision \t 6;\n\n'
        cdicttext = cdicttext + 'writeCompression \t uncompressed;\n\n'
        cdicttext = cdicttext + 'timeFormat \t general;\n\n'
        cdicttext = cdicttext + 'timePrecision \t 6;\n\n'
        cdicttext = cdicttext + 'runTimeModifiable \t yes;\n\n'
        cdicttext = cdicttext + 'adjustTimeStep \t yes;\n\n'
        cdicttext = cdicttext + 'maxCo \t 1.0;\n\n'
        cdicttext = cdicttext + 'maxAlphaCo \t 1.0;\n\n'
        cdicttext = cdicttext + 'maxDeltaT \t 1;\n\n'

        # Function for building
        cdicttext = cdicttext + 'functions\n{\n\t'
        cdicttext = cdicttext + 'buildingsForces\n\t{\n\t\t'
        cdicttext = cdicttext + 'type\tforces;\n\t\t'
        cdicttext = cdicttext + 'functionObjectLibs\t("libforces.so");\n\t\t'
        cdicttext = cdicttext + 'writeControl\ttimeStep;\n\t\t'
        cdicttext = cdicttext + 'writeInterval\t1;\n\t\t'
        cdicttext = (
            cdicttext + 'patches\t("Building");\n\t\t'
        )  # This needs to be changed to Building
        cdicttext = cdicttext + 'rho\trhoInf;\n\t\t'
        cdicttext = cdicttext + 'log\ttrue;\n\t\t'
        cdicttext = cdicttext + 'rhoInf\t1;\n\t\t'
        cdicttext = cdicttext + 'CofR\t(0 0 0);\n\t\t'

        # Get the number of stories
        stories = hydroutil.extract_element_from_json(
            data, ['GeneralInformation', 'stories']
        )

        cdicttext = cdicttext + 'binData\n\t\t{\n\t\t\t'
        cdicttext = cdicttext + 'nBin\t' + str(stories[0]) + ';\n\t\t\t'
        cdicttext = cdicttext + 'direction\t(1 0 0);\n\t\t\t'
        cdicttext = cdicttext + 'cumulative\tno;\n\t\t}\n\t}\n}'

        return cdicttext  # noqa: RET504
