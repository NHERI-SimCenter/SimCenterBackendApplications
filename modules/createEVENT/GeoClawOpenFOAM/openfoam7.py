#  # noqa: INP001, EXE002
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

"""  # noqa: D400
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
from of7Alpboundary import of7Alpboundary
from of7Building import of7Building
from of7Dakota import of7Dakota
from of7Decomp import of7Decomp
from of7Geometry import of7Geometry
from of7Initial import of7Initial
from of7Materials import of7Materials
from of7Meshing import of7Meshing
from of7Others import of7Others
from of7Prboundary import of7Prboundary
from of7Process import of7Process
from of7PtDboundary import of7PtDboundary
from of7Solve import of7Solve
from of7Turbulence import of7Turbulence
from of7Uboundary import of7Uboundary


####################################################################
# OpenFOAM7 solver class
####################################################################
class openfoam7:
    """This class includes the methods related to openfoam7.

    Methods
    -------
            extract:

    """  # noqa: D404

    #############################################################
    def createfolder(self, data, path, args):
        """Creates the necessary folders for openfoam7

        Arguments:
        ---------
                data: all the JSON data
                path: Path where the new folder needs to be created

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()

        # Create directories for openfoam dictionaries
        # Access: Only owner can read and write
        access_rights = 0o700

        # Create 0-directory
        pathF = os.path.join(path, '0.org')  # noqa: PTH118, N806
        if os.path.exists(pathF):  # noqa: PTH110
            shutil.rmtree(pathF)
            os.mkdir(pathF, access_rights)  # noqa: PTH102
        else:
            os.mkdir(pathF, access_rights)  # noqa: PTH102

        # Create constant-directory
        pathF = os.path.join(path, 'constant')  # noqa: PTH118, N806
        if os.path.exists(pathF):  # noqa: PTH110
            shutil.rmtree(pathF)
            os.mkdir(pathF, access_rights)  # noqa: PTH102
        else:
            os.mkdir(pathF, access_rights)  # noqa: PTH102

        # Create the triSurface directory
        pathF = os.path.join(path, 'constant', 'triSurface')  # noqa: PTH118, N806
        if os.path.exists(pathF):  # noqa: PTH110
            shutil.rmtree(pathF)
            os.mkdir(pathF, access_rights)  # noqa: PTH102
        else:
            os.mkdir(pathF, access_rights)  # noqa: PTH102

        # Create system-directory
        pathF = os.path.join(path, 'system')  # noqa: PTH118, N806
        if os.path.exists(pathF):  # noqa: PTH110
            shutil.rmtree(pathF)
            os.mkdir(pathF, access_rights)  # noqa: PTH102
        else:
            os.mkdir(pathF, access_rights)  # noqa: PTH102

        # Get the information from json file
        hydrobrain = ', '.join(
            hydroutil.extract_element_from_json(data, ['remoteAppDir'])
        )
        mesher = ', '.join(  # noqa: F841
            hydroutil.extract_element_from_json(data, ['Events', 'MeshType'])
        )
        simtype = ', '.join(  # noqa: F841
            hydroutil.extract_element_from_json(data, ['Events', 'SimulationType'])
        )

        # Add all variables
        caseruntext = 'echo Setting up variables\n\n'
        caseruntext = caseruntext + 'export BIM=' + args.b + '\n\n'
        caseruntext = caseruntext + 'export HYDROPATH=' + path + '\n\n'
        caseruntext = caseruntext + 'export LD_LIBRARY_PATH=' + args.L + '\n\n'
        caseruntext = caseruntext + 'export PATH=' + args.P + '\n\n'
        caseruntext = caseruntext + 'export inputFile=' + args.i + '\n\n'
        caseruntext = caseruntext + 'export driverFile=' + args.d + '\n\n'
        caseruntext = caseruntext + 'export inputDirectory=' + path + '\n\n'
        caseruntext = (
            caseruntext
            + 'export HYDROBRAIN='
            + os.path.join(  # noqa: PTH118
                hydrobrain, 'applications', 'createEVENT', 'GeoClawOpenFOAM'
            )
            + '\n\n'
        )

        # Load all modules
        caseruntext = caseruntext + 'echo Loading modules on Stampede2\n'
        caseruntext = caseruntext + 'module load intel/18.0.2\n'
        caseruntext = caseruntext + 'module load impi/18.0.2\n'
        caseruntext = caseruntext + 'module load openfoam/7.0\n'
        caseruntext = caseruntext + 'module load dakota/6.8.0\n'
        caseruntext = caseruntext + 'module load python3\n\n'

        # Move the case files to the present folder
        zerofldr = os.path.join(path, '0.org')  # noqa: PTH118
        zero2fldr = '0'  # noqa: F841
        cstfldr = os.path.join(path, 'constant')  # noqa: PTH118
        systfldr = os.path.join(path, 'system')  # noqa: PTH118
        caseruntext = caseruntext + 'cp -r ' + zerofldr + ' .\n'
        caseruntext = caseruntext + 'cp -r 0.org 0\n'
        caseruntext = caseruntext + 'cp -r ' + cstfldr + ' .\n'
        caseruntext = caseruntext + 'cp -r ' + systfldr + ' .\n\n'

        # Create the caserun file
        if os.path.exists('caserun.sh'):  # noqa: PTH110
            os.remove('caserun.sh')  # noqa: PTH107
        scriptfile = open('caserun.sh', 'w')  # noqa: SIM115, PTH123
        scriptfile.write(caseruntext)
        scriptfile.close()

        # Return completion flag
        return 0  # noqa: DOC201, RUF100

    #############################################################
    def creategeometry(self, data, path):
        """Creates the necessary folders for openfoam7

        Arguments:
        ---------
                data: all the JSON data
                path: Path where the geometry files (STL) needs to be created

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()

        # Get mesher type
        mesher = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'MeshType'])
        )

        # Create the geometry related files
        Geometry = of7Geometry()  # noqa: N806
        if int(mesher[0]) == 1:
            return 0  # noqa: DOC201, RUF100
        elif int(mesher[0]) == 0 or int(mesher[0]) == 2:  # noqa: RET505, PLR2004
            geomcode = Geometry.geomcheck(data, path)
            if geomcode == -1:
                return -1
            else:  # noqa: RET505
                stlcode = Geometry.createOFSTL(data, path)
                if stlcode < 0:
                    return -1

        # Building related files
        Building = of7Building()  # noqa: N806
        if int(mesher[0]) == 1:
            return 0
        elif int(mesher[0]) == 0 or int(mesher[0]) == 2:  # noqa: RET505, PLR2004
            buildcode = Building.buildcheck(data, path)
            if buildcode == -1:
                return -1
            else:  # noqa: RET505
                buildcode2 = Building.createbuilds(data, path)
                if buildcode2 < 0:
                    return -1

        # Solution related files (SW solutions)
        # Always needed irrespective of geometry / mesh

        # Scripts
        Geometry.scripts(data)

        return 0

    #############################################################
    def createmesh(self, data, path):
        """Creates the mesh dictionaries for openfoam7

        Arguments:
        ---------
                data: all the JSON data
                path: Path where the geometry files (STL) needs to be created

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()

        # Get mesher type
        mesher = ', '.join(
            hydroutil.extract_element_from_json(data, ['Events', 'MeshType'])
        )

        # Create the meshing related file
        Meshing = of7Meshing()  # noqa: N806
        meshcode = Meshing.meshcheck(data, path)
        if meshcode == -1:
            return -1  # noqa: DOC201, RUF100
        elif int(mesher[0]) == 0:  # noqa: RET505
            # blockMesh
            bmeshtext = Meshing.bmeshtext(data)
            fname = 'blockMeshDict'
            filepath = os.path.join(path, 'system', fname)  # noqa: PTH118
            bmeshfile = open(filepath, 'w')  # noqa: SIM115, PTH123
            bmeshfile.write(bmeshtext)
            bmeshfile.close()
            # surfaceFeatureExtract
            sfetext = Meshing.sfetext()
            fname = 'surfaceFeatureExtractDict'
            filepath = os.path.join(path, 'system', fname)  # noqa: PTH118
            sfefile = open(filepath, 'w')  # noqa: SIM115, PTH123
            sfefile.write(sfetext)
            sfefile.close()
            # snappyHexMesh
            shmtext = Meshing.shmtext(data)
            fname = 'snappyHexMeshDict'
            filepath = os.path.join(path, 'system', fname)  # noqa: PTH118
            shmfile = open(filepath, 'w')  # noqa: SIM115, PTH123
            shmfile.write(shmtext)
            shmfile.close()

            # Mesh files from other software (1)
            # Do nothing here. Add to caserun.sh

            # User mesh dictionaries (2)
            # Do nothing here. Copy files to relevant place
            # in caserun.sh

        # Scripts
        Meshing.scripts(data, path)

        return 0

    #############################################################
    def materials(self, data, path):
        """Creates the material files for openfoam7

        Arguments:
        ---------
                data: all the JSON data
                path: Path where the geometry files (STL) needs to be created

        """  # noqa: D400, D401
        # Create the transportProperties file
        Materials = of7Materials()  # noqa: N806
        matcode = Materials.matcheck(data)
        if matcode == -1:
            return -1  # noqa: DOC201, RUF100
        else:  # noqa: RET505
            mattext = Materials.mattext(data)
            fname = 'transportProperties'
            filepath = os.path.join(path, 'constant', fname)  # noqa: PTH118
            matfile = open(filepath, 'w')  # noqa: SIM115, PTH123
            matfile.write(mattext)
            matfile.close()

        return 0

    #############################################################
    def initial(self, data, path):
        """Creates the initial condition files for openfoam7

        Arguments:
        ---------
                data: all the JSON data
                path: Path where the geometry files dakota.json lies

        """  # noqa: D400, D401
        # Create the setFields file
        Inicond = of7Initial()  # noqa: N806
        initcode = Inicond.alphacheck(data, path)
        if initcode == -1:
            return -1  # noqa: DOC201, RUF100
        else:  # noqa: RET505
            alphatext = Inicond.alphatext(data, path)
            fname = 'setFieldsDict'
            filepath = os.path.join(path, 'system', fname)  # noqa: PTH118
            alphafile = open(filepath, 'w')  # noqa: SIM115, PTH123
            alphafile.write(alphatext)
            alphafile.close()

        # Scripts
        Inicond.scripts(data, path)

        return 0

    #############################################################
    def boundary(self, data, path):
        """Creates the bc condition files for openfoam7

        Arguments:
        ---------
                data: all the JSON data
                path: Path where the geometry files (STL) needs to be created

        """  # noqa: D400, D401
        # Initialize the patches
        patches = ['Entry', 'Exit', 'Top', 'Bottom', 'Right', 'Left']

        # Create object for velocity boundary condition
        # Get the text for the velocity boundary
        # Write the U-file in 0.org
        Uboundary = of7Uboundary()  # noqa: N806
        utext = Uboundary.Utext(data, path, patches)
        # Check for boundary conditions here
        ecode = Uboundary.Uchecks(data, path, patches)
        if ecode == -1:
            return -1  # noqa: DOC201, RUF100
        else:  # noqa: RET505
            # Write the U-file if no errors
            # Path to the file
            fname = 'U'
            filepath = os.path.join(path, '0.org', fname)  # noqa: PTH118
            Ufile = open(filepath, 'w')  # noqa: SIM115, PTH123, N806
            Ufile.write(utext)
            Ufile.close()

        # Create object for pressure boundary condition
        # Get the text for the pressure boundary
        # Write the p_rgh-file in 0.org
        Prboundary = of7Prboundary()  # noqa: N806
        prtext = Prboundary.Prtext(data, patches)
        fname = 'p_rgh'
        filepath = os.path.join(path, '0.org', fname)  # noqa: PTH118
        pr_file = open(filepath, 'w')  # noqa: SIM115, PTH123
        pr_file.write(prtext)
        pr_file.close()

        # Create object for alpha boundary condition
        # Get the text for the alpha boundary
        # Write the alpha-file in 0.org
        Alpboundary = of7Alpboundary()  # noqa: N806
        Alptext = Alpboundary.Alptext(data, patches)  # noqa: N806
        fname = 'alpha.water'
        filepath = os.path.join(path, '0.org', fname)  # noqa: PTH118
        Alpfile = open(filepath, 'w')  # noqa: SIM115, PTH123, N806
        Alpfile.write(Alptext)
        Alpfile.close()

        # Loop over all the velocity type to see if any
        # has a moving wall. If so initialize the
        # pointDisplacement file
        PtDboundary = of7PtDboundary()  # noqa: N806
        ptDcode = PtDboundary.PtDcheck(data, patches)  # noqa: N806
        if ptDcode == 1:
            pdtext = PtDboundary.PtDtext(data, path, patches)
            fname = 'pointDisplacement'
            filepath = os.path.join(path, '0.org', fname)  # noqa: PTH118
            ptDfile = open(filepath, 'w')  # noqa: SIM115, PTH123, N806
            ptDfile.write(pdtext)
            ptDfile.close()

        return 0

    #############################################################
    def turbulence(self, data, path):
        """Creates the turbulenceDict and other files for openfoam7

        Arguments:
        ---------
                data: all the JSON data
                path: Path where the geometry files (STL) needs to be created

        """  # noqa: D400, D401
        # Create the domain decomposition file
        Turb = of7Turbulence()  # noqa: N806
        turbtext = Turb.turbtext(data)
        fname = 'turbulenceProperties'
        filepath = os.path.join(path, 'constant', fname)  # noqa: PTH118
        turbfile = open(filepath, 'w')  # noqa: SIM115, PTH123
        turbfile.write(turbtext)
        turbfile.close()

        return 0  # noqa: DOC201, RUF100

    #############################################################
    def parallelize(self, data, path):
        """Creates the domain decomposition files for openfoam7

        Arguments:
        ---------
                data: all the JSON data
                path: Path where the geometry files (STL) needs to be created

        """  # noqa: D400, D401
        # Create the domain decomposition file
        Decomp = of7Decomp()  # noqa: N806
        decomptext = Decomp.decomptext(data)
        fname = 'decomposeParDict'
        filepath = os.path.join(path, 'system', fname)  # noqa: PTH118
        decompfile = open(filepath, 'w')  # noqa: SIM115, PTH123
        decompfile.write(decomptext)
        decompfile.close()

        # Scripts
        Decomp.scripts(data, path)

        return 0  # noqa: DOC201, RUF100

    #############################################################
    def solve(self, data, path):
        """Creates the solver related files for openfoam7

        Arguments:
        ---------
                data: all the JSON data
                path: Path where the geometry files (STL) needs to be created

        """  # noqa: D400, D401
        # Create the solver files
        Solve = of7Solve()  # noqa: N806
        # fvSchemes
        fvschemetext = Solve.fvSchemetext(data)
        fname = 'fvSchemes'
        filepath = os.path.join(path, 'system', fname)  # noqa: PTH118
        fvschemefile = open(filepath, 'w')  # noqa: SIM115, PTH123
        fvschemefile.write(fvschemetext)
        fvschemefile.close()

        # fvSolutions
        fvsolntext = Solve.fvSolntext(data)
        fname = 'fvSolution'
        filepath = os.path.join(path, 'system', fname)  # noqa: PTH118
        fvsolnfile = open(filepath, 'w')  # noqa: SIM115, PTH123
        fvsolnfile.write(fvsolntext)
        fvsolnfile.close()

        # controlDict
        ecode = Solve.cdictcheck(data)
        if ecode == -1:
            return -1  # noqa: DOC201, RUF100
        else:  # noqa: RET505
            cdicttext = Solve.cdicttext(data)
            fname = 'controlDict'
            filepath = os.path.join(path, 'system', fname)  # noqa: PTH118
            cdictfile = open(filepath, 'w')  # noqa: SIM115, PTH123
            cdictfile.write(cdicttext)
            cdictfile.close()

            # Create CdictForce
            cdictFtext = Solve.cdictFtext(data)  # noqa: N806
            fname = 'cdictforce'
            cdictFfile = open(fname, 'w')  # noqa: SIM115, PTH123, N806
            cdictFfile.write(cdictFtext)
            cdictFfile.close()

        return 0

    #############################################################
    def others(self, data, path):
        """Creates the other auxiliary files for openfoam7

        Arguments:
        ---------
                data: all the JSON data
                path: Path where the geometry files (STL) needs to be created

        """  # noqa: D400, D401
        # Create the auxiliary files
        Others = of7Others()  # noqa: N806
        # g-file
        gfiletext = Others.gfiletext(data)
        fname = 'g'
        filepath = os.path.join(path, 'constant', fname)  # noqa: PTH118
        gfile = open(filepath, 'w')  # noqa: SIM115, PTH123
        gfile.write(gfiletext)
        gfile.close()

        return 0  # noqa: DOC201, RUF100

    #############################################################
    def dakota(self, args):
        """Creates the dakota scripts for openfoam7

        Arguments:
        ---------
                args: all arguments

        """  # noqa: D400, D401
        # Create the solver files
        dakota = of7Dakota()

        # Dakota Scripts
        dakota.dakotascripts(args)

        return 0  # noqa: DOC201, RUF100

    #############################################################
    def postprocessing(self, data, path):
        """Creates the postprocessing related files for openfoam7

        Arguments:
        ---------
                data: all the JSON data
                path: Path where the geometry files (STL) needs to be created

        """  # noqa: D400, D401
        # Create the solver files
        pprocess = of7Process()
        # controlDict
        ecode = pprocess.pprocesscheck(data, path)
        if ecode == -1:
            return -1  # noqa: DOC201, RUF100
        elif ecode == 0:  # noqa: RET505
            return 0
        else:
            # sample file
            pprocesstext = pprocess.pprocesstext(data, path)
            fname = 'sample'
            filepath = os.path.join(fname)  # noqa: PTH118
            samplefile = open(filepath, 'w')  # noqa: SIM115, PTH123
            samplefile.write(pprocesstext)
            samplefile.close()
            # Controldict
            pprocesstext = pprocess.pprocesscdict(data, path)
            fname = 'cdictpp'
            filepath = os.path.join(fname)  # noqa: PTH118
            samplefile = open(filepath, 'w')  # noqa: SIM115, PTH123
            samplefile.write(pprocesstext)
            samplefile.close()

        # Scripts
        pprocess.scripts(data, path)

        return 0

    #############################################################
    def cleaning(self, args, path):
        """Creates the cleaning scripts for openfoam7

        Arguments:
        ---------
                args: all arguments

        """  # noqa: D400, D401
        # Create the solver files
        cleaner = of7Dakota()

        # Dakota Scripts
        cleaner.cleaning(args, path)

        return 0  # noqa: DOC201, RUF100
