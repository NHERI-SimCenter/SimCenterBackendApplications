# # noqa: INP001
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

# Other custom modules
from hydroUtils import hydroUtils


####################################################################
# OpenFOAM7 solver class
####################################################################
class of7Uboundary:
    """This class includes the methods related to
    velocity boundary conditions for openfoam7.

    Methods
    -------
            Utext: Get s all the text for the U-file

    """  # noqa: D205, D404

    #############################################################
    def Utext(self, data, fipath, patches):  # noqa: N802
        """Creates the necessary folders for openfoam7

        Arguments:
        ---------
                data: all the JSON data
                patches: List of boundary patches
                fipath: Path where the dakota.json file exists

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()

        # Number of moving walls
        numMovWall = 0  # noqa: N806

        # Get the header text for the U-file
        utext = self.Uheader()

        # Start the outside
        utext = utext + 'boundaryField\n{\n'  # noqa: PLR6104

        # Loop over all patches
        for patchname in patches:
            utext = utext + '\t' + patchname + '\n'
            patch = hydroutil.extract_element_from_json(
                data, ['Events', 'VelocityType_' + patchname]
            )
            if patch == [None]:
                Utype = -1  # noqa: N806
            else:
                Utype = ', '.join(  # noqa: N806
                    hydroutil.extract_element_from_json(
                        data, ['Events', 'VelocityType_' + patchname]
                    )
                )
                if int(Utype) == 103 or int(Utype) == 104:  # noqa: PLR2004
                    numMovWall += 1  # noqa: N806
            utext = utext + self.Upatchtext(  # noqa: PLR6104
                data, Utype, patchname, fipath, numMovWall
            )

        # Check for building and other building
        utext = utext + '\tBuilding\n'  # noqa: PLR6104
        utext = utext + self.Upatchtext(data, '301', 'Building', fipath, numMovWall)  # noqa: PLR6104
        utext = utext + '\tOtherBuilding\n'  # noqa: PLR6104
        utext = utext + self.Upatchtext(  # noqa: PLR6104
            data, '301', 'OtherBuilding', fipath, numMovWall
        )

        # Close the outside
        utext = utext + '}\n\n'  # noqa: PLR6104

        # Return the text for velocity BC
        return utext  # noqa: RET504

    #############################################################
    def Uheader(self):  # noqa: N802, PLR6301
        """Creates the text for the header

        Variable
        -----------
                header: Header for the U-file
        """  # noqa: D400, D401
        header = """/*--------------------------*- NHERI SimCenter -*----------------------------*\\ 
|	   | H |
|	   | Y | HydroUQ: Water-based Natural Hazards Modeling Application
|======| D | Website: simcenter.designsafe-ci.org/research-tools/hydro-uq
|	   | R | Version: 1.00
|	   | O |
\\*---------------------------------------------------------------------------*/ 
FoamFile
{\n\tversion\t2.0;\n\tformat\tascii;\n\tclass\tvolVectorField;\n\tlocation\t"0";\n\tobject\tU;\n}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"""  # noqa: W291

        header = header + 'dimensions\t[0 1 -1 0 0 0 0];\n\n'  # noqa: PLR6104
        header = header + 'internalField\tuniform (0 0 0);\n\n'  # noqa: PLR6104

        # Return the header for U file
        return header  # noqa: RET504

    #############################################################
    def Upatchtext(self, data, Utype, patchname, fipath, numMovWall):  # noqa: C901, N802, N803
        """Creates the text the velocity boundary condition

        Arguments:
        ---------
                data: All the json data
                Utype: Type of velocity b.c
                patchname: Name of the patch
                fipath: Path to where dakota.json file exists

        Variable
        -----------
                Utext: Text for the particular patch

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()

        # Inlet types
        # For each type, get the text
        if int(Utype) == 101:  # noqa: PLR2004
            # SW solutions (1)
            Utext = '\t{\n\t\t'  # noqa: N806
            Utext = Utext + 'type\ttimeVaryingMappedFixedValue;\n\t\t'  # noqa: N806, PLR6104
            Utext = Utext + 'offset\t(0 0 0);\n\t\t'  # noqa: N806, PLR6104
            Utext = Utext + 'setAverage\toff;\n'  # noqa: N806, PLR6104
            Utext = Utext + '\t}\n'  # noqa: N806, PLR6104

        elif int(Utype) == 102:  # noqa: PLR2004
            # Inlet: constant velocity
            # Get the velocity values
            velo = hydroutil.extract_element_from_json(
                data, ['Events', 'Velocity_' + patchname]
            )
            if velo == [None]:
                vx = 0.0
                vy = 0.0
                vz = 0.0
            else:
                velvals = ', '.join(
                    hydroutil.extract_element_from_json(
                        data, ['Events', 'Velocity_' + patchname]
                    )
                )
                velvals = velvals.replace(',', ' ')
                vels = [float(vi) for vi in velvals.split()]
                vx = vels[0]
                vy = vels[1]
                vz = vels[2]

            # Get the text
            Utext = '\t{\n\t\t'  # noqa: N806
            Utext = Utext + 'type\tfixedValue;\n\t\t'  # noqa: N806, PLR6104
            Utext = (  # noqa: N806
                Utext
                + 'value\t('
                + str(vx)
                + '\t'
                + str(vy)
                + '\t'
                + str(vz)
                + ');\n\t}\n'
            )

        elif int(Utype) == 103:  # noqa: PLR2004
            # Inlet Moving wall (OSU flume)
            Utext = '\t{\n\t\t'  # noqa: N806
            Utext = Utext + 'type\tmovingWallVelocity;\n\t\t'  # noqa: N806, PLR6104
            Utext = Utext + 'value\tuniform (0 0 0);\n\t}\n'  # noqa: N806, PLR6104
            # Create the files required
            # Moving wall file
            # Get the displacement and waterheight file name
            dispfilename = hydroutil.extract_element_from_json(
                data, ['Events', 'OSUMovingWallDisp_' + patchname]
            )
            if dispfilename != [None]:
                dispfilename = ', '.join(
                    hydroutil.extract_element_from_json(
                        data, ['Events', 'OSUMovingWallDisp_' + patchname]
                    )
                )
                dispfilepath = os.path.join(fipath, dispfilename)  # noqa: PTH118
                if os.path.exists(dispfilepath):  # noqa: PTH110
                    heightfilename = hydroutil.extract_element_from_json(
                        data, ['Events', 'OSUMovingWallHeight_' + patchname]
                    )
                    if heightfilename != [None]:
                        heightfilename = ', '.join(
                            hydroutil.extract_element_from_json(
                                data, ['Events', 'OSUMovingWallHeight_' + patchname]
                            )
                        )
                        heightfilepath = os.path.join(fipath, heightfilename)  # noqa: PTH118
                        if not os.path.exists(heightfilepath):  # noqa: PTH110
                            heightfilepath = 'None'
                    else:
                        heightfilepath = 'None'
                    # Wave maker text file
                    self.OSUwavemakerText(
                        fipath, dispfilepath, heightfilepath, numMovWall
                    )
                    # Wavemakermovement dictionary
                    self.of7wavemakerdict(fipath)
                    # Dynamic mesh dictionary
                    self.of7dynamicMeshdict(fipath)

        elif int(Utype) == 104:  # noqa: PLR2004
            # Inlet Moving wall (Gen flume)
            Utext = '\t{\n\t\t'  # noqa: N806
            Utext = Utext + 'type\tmovingWallVelocity;\n\t\t'  # noqa: N806, PLR6104
            Utext = Utext + 'value\tuniform (0 0 0);\n\t}\n'  # noqa: N806, PLR6104
            # Create the files required
            # Moving wall file
            # Get the displacement and waterheight file name
            # # Get the displacement and waterheight file name
            dispfilename = hydroutil.extract_element_from_json(
                data, ['Events', 'MovingWallDisp_' + patchname]
            )
            if dispfilename != [None]:
                dispfilename = ', '.join(
                    hydroutil.extract_element_from_json(
                        data, ['Events', 'MovingWallDisp_' + patchname]
                    )
                )
                dispfilepath = os.path.join(fipath, dispfilename)  # noqa: PTH118
                if os.path.exists(dispfilepath):  # noqa: PTH110
                    heightfilename = hydroutil.extract_element_from_json(
                        data, ['Events', 'MovingWallHeight_' + patchname]
                    )
                    if heightfilename != [None]:
                        heightfilename = ', '.join(
                            hydroutil.extract_element_from_json(
                                data, ['Events', 'MovingWallHeight_' + patchname]
                            )
                        )
                        heightfilepath = os.path.join(fipath, heightfilename)  # noqa: PTH118
                        if not os.path.exists(heightfilepath):  # noqa: PTH110
                            heightfilepath = 'None'
                    else:
                        heightfilepath = 'None'
                    # Wave maker text file
                    self.GenwavemakerText(
                        fipath, dispfilepath, heightfilepath, numMovWall
                    )
                    # Wavemakermovement dictionary
                    self.of7wavemakerdict(fipath)
                    # Dynamic mesh dictionary
                    self.of7dynamicMeshdict(fipath)

        elif int(Utype) == 201:  # noqa: PLR2004
            # Outlet zero gradient
            Utext = '\t{\n\t\t'  # noqa: N806
            Utext = Utext + 'type\tzeroGradient;\n\t}\n'  # noqa: N806, PLR6104

        elif int(Utype) == 202:  # noqa: PLR2004
            # Outlet: inletOutlet
            # Get the velocity values
            velo = hydroutil.extract_element_from_json(
                data, ['Events', 'Velocity_' + patchname]
            )
            if velo == [None]:
                vx = 0.0
                vy = 0.0
                vz = 0.0
            else:
                velvals = ', '.join(
                    hydroutil.extract_element_from_json(
                        data, ['Events', 'Velocity_' + patchname]
                    )
                )
                velvals = velvals.replace(',', ' ')
                vels = [float(vi) for vi in velvals.split()]
                vx = vels[0]
                vy = vels[1]
                vz = vels[2]

            # Get the text
            Utext = '\t{\n\t\t'  # noqa: N806
            Utext = Utext + 'type\tinletOutlet;\n\t\t'  # noqa: N806, PLR6104
            Utext = (  # noqa: N806
                Utext
                + 'inletValue\tuniform ('
                + str(vx)
                + '\t'
                + str(vy)
                + '\t'
                + str(vz)
                + ');\n\t\t'
            )
            Utext = (  # noqa: N806
                Utext
                + 'value\tuniform ('
                + str(vx)
                + '\t'
                + str(vy)
                + '\t'
                + str(vz)
                + ');\n'
            )
            Utext = Utext + '\t}\n'  # noqa: N806, PLR6104

        elif int(Utype) == 301:  # noqa: PLR2004
            # Wall: noSlip
            Utext = '\t{\n\t\t'  # noqa: N806
            Utext = Utext + 'type\tnoSlip;\n\t}\n'  # noqa: N806, PLR6104

        else:
            # Default: Empty
            Utext = '\t{\n\t\t'  # noqa: N806
            Utext = Utext + 'type\tempty;\n\t}\n'  # noqa: N806, PLR6104

        # Return the header for U file
        return Utext

    #############################################################
    def Uchecks(self, data, fipath, patches):  # noqa: C901, N802, PLR6301
        """Creates the data files required for the OSU moving wall

        Arguments:
        ---------
                data: All the data from JSON file
                fipath: Path to the dakota.json file location
                patches: List of patches

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()

        # Number of moving walls
        numMovWall = 0  # noqa: N806

        # Loop over all patches
        for patchname in patches:
            # Get the type of velocity bc
            patch = hydroutil.extract_element_from_json(
                data, ['Events', 'VelocityType_' + patchname]
            )
            if patch == [None]:
                Utype = -1  # noqa: N806
            else:
                Utype = ', '.join(  # noqa: N806
                    hydroutil.extract_element_from_json(
                        data, ['Events', 'VelocityType_' + patchname]
                    )
                )

            # Checks for different U-types
            if int(Utype) == 103:  # noqa: PLR2004
                # Checking for multiple moving walls
                numMovWall += 1  # noqa: N806
                if numMovWall > 1:
                    return -1

                # Check for existing moving wall files
                dispfilename = hydroutil.extract_element_from_json(
                    data, ['Events', 'OSUMovingWallDisp_' + patchname]
                )
                if dispfilename == [None]:
                    return -1
                else:  # noqa: RET505
                    dispfilename = ', '.join(
                        hydroutil.extract_element_from_json(
                            data, ['Events', 'OSUMovingWallDisp_' + patchname]
                        )
                    )
                pathF = os.path.join(fipath, dispfilename)  # noqa: PTH118, N806
                if not os.path.exists(pathF):  # noqa: PTH110
                    return -1

            elif int(Utype) == 104:  # noqa: PLR2004
                # Checking for multiple moving walls
                numMovWall += 1  # noqa: N806
                if numMovWall > 1:
                    return -1

                # Check for existing moving wall files
                dispfilename = hydroutil.extract_element_from_json(
                    data, ['Events', 'MovingWallDisp_' + patchname]
                )
                if dispfilename == [None]:
                    return -1
                else:  # noqa: RET505
                    dispfilename = ', '.join(
                        hydroutil.extract_element_from_json(
                            data, ['Events', 'MovingWallDisp_' + patchname]
                        )
                    )
                pathF = os.path.join(fipath, dispfilename)  # noqa: PTH118, N806
                if not os.path.exists(pathF):  # noqa: PTH110
                    return -1

        # If all checks passes
        return 0

    #############################################################
    def of7wavemakerdict(self, fipath):  # noqa: PLR6301
        """Creates the wavemaker dictionary for the moving wall

        Arguments:
        ---------
                fipath: Path to the dakota.json file location

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()

        # Get the file ID
        filepath = os.path.join(fipath, 'constant', 'wavemakerMovementDict')  # noqa: PTH118
        fileID = open(filepath, 'w')  # noqa: N806, PLW1514, PTH123, SIM115
        # Header
        header = hydroutil.of7header(
            'dictionary', 'constant', 'wavemakerMovementDict'
        )
        fileID.write(header)
        # Other data
        fileID.write('\nreread\tfalse;\n\n')
        fileID.write('#include\t"wavemakerMovement.txt"\n')
        # Close the file
        fileID.close()

    #############################################################
    def of7dynamicMeshdict(self, fipath):  # noqa: N802, PLR6301
        """Creates the dynamic mesh dictionary for the moving wall

        Arguments:
        ---------
                fipath: Path to the dakota.json file location

        """  # noqa: D400, D401
        # Create a utilities object
        hydroutil = hydroUtils()

        # Get the file ID
        filepath = os.path.join(fipath, 'constant', 'dynamicMeshDict')  # noqa: PTH118
        fileID = open(filepath, 'w')  # noqa: N806, PLW1514, PTH123, SIM115
        # Header
        header = hydroutil.of7header('dictionary', 'constant', 'dynamicMeshDict')
        fileID.write(header)
        # Other data
        fileID.write('\ndynamicFvMesh\tdynamicMotionSolverFvMesh;\n\n')
        fileID.write('motionSolverLibs\t("libfvMotionSolvers.so");\n\n')
        fileID.write('solver\tdisplacementLaplacian;\n\n')
        fileID.write('displacementLaplacianCoeffs\n{\n\tdiffusivity uniform;\n}\n')
        # Close the file
        fileID.close()

    #############################################################
    def OSUwavemakerText(self, fipath, dispfilepath, heightfilepath, numMovWall):  # noqa: ARG002, C901, N802, N803, PLR6301
        """Creates the wavemaker text file for the OSU moving wall

        Arguments:
        ---------
                fipath: Path to the dakota.json file location

        """  # noqa: D400, D401
        # Get the file ID
        filepath = os.path.join(fipath, 'constant', 'wavemakerMovement.txt')  # noqa: PTH118
        fileID = open(filepath, 'w')  # noqa: N806, PLW1514, PTH123, SIM115

        # Start writing the file
        fileID.write('wavemakerType\tPiston;\n')
        fileID.write('tSmooth\t1.5;\n')
        fileID.write('genAbs\t0;\n\n')

        # Create the wavemaker movement file
        # Get the frequency of the wavemaker
        frequency = 0
        waterdepth = 0
        filewm = open(dispfilepath)  # noqa: PLW1514, PTH123, SIM115
        Lines = filewm.readlines()  # noqa: N806
        count = 0
        for line in Lines:
            count += 1  # noqa: SIM113
            if count == 37:  # noqa: PLR2004
                stra = line.replace('% SampleRate: ', '')
                stra2 = stra.replace(' Hz', '')
                frequency = 1 / float(stra2)
                break
        count = 0
        for line in Lines:
            count += 1
            if count == 61:  # noqa: PLR2004
                stra = line.replace('% StillWaterDepth: ', '')
                waterdepth = float(stra)
                break

        # Count the number of lines
        countlines = 0
        with open(dispfilepath) as fdisp:  # noqa: PLW1514, PTH123
            for line2 in fdisp:
                if line2.strip():
                    countlines += 1
        countlines = countlines - 72  # noqa: PLR6104

        # Create the timeseries
        time = 0
        fileID.write('timeSeries\t' + str(countlines) + '(\n')
        for ii in range(countlines):  # noqa: B007
            fileID.write(str(time) + '\n')
            time = time + frequency  # noqa: PLR6104
        fileID.write(');\n\n')

        # Create the paddle position
        fileID.write('paddlePosition 1(\n' + str(countlines) + '(\n')
        count = 0
        for line in Lines:
            count += 1
            if count > 72:  # noqa: PLR2004
                if line != '\n':
                    data = float(line)
                    fileID.write(str(data) + '\n')
        fileID.write(')\n);\n\n')

        # Write the paddle Eta
        if heightfilepath != 'None':
            # Write the paddle Eta
            fileID.write('paddleEta 1(\n' + str(countlines) + '(\n')
            filewmg = open(heightfilepath)  # noqa: PLW1514, PTH123, SIM115
            Lines2 = filewmg.readlines()  # noqa: N806
            count = 0
            for line in Lines2:
                count += 1
                if count > 72:  # noqa: PLR2004
                    if line != '\n':
                        data = float(line) + waterdepth
                        fileID.write(str(data) + '\n')
            fileID.write(')\n);')

    #############################################################
    def GenwavemakerText(self, fipath, dispfilepath, heightfilepath, numMovWall):  # noqa: ARG002, C901, N802, N803, PLR6301
        """Creates the wavemaker text file for a general moving wall

        Arguments:
        ---------
                fipath: Path to the dakota.json file location

        """  # noqa: D400, D401
        # Get the file ID
        filepath = os.path.join(fipath, 'constant', 'wavemakerMovement.txt')  # noqa: PTH118
        fileID = open(filepath, 'w')  # noqa: N806, PLW1514, PTH123, SIM115

        # Start writing the file
        fileID.write('wavemakerType\tPiston;\n')
        fileID.write('tSmooth\t1.5;\n')
        fileID.write('genAbs\t0;\n\n')

        # Create the wavemaker movement file
        # Get the frequency of the wavemaker
        filewm = open(dispfilepath)  # noqa: PLW1514, PTH123, SIM115
        Lines = filewm.readlines()  # noqa: N806
        count = 0
        for line in Lines:
            count += 1  # noqa: SIM113
            if count == 1:
                frequency = float(line)
                break

        # Count the number of lines
        countlines = 0
        with open(dispfilepath) as fdisp:  # noqa: PLW1514, PTH123
            for line2 in fdisp:
                if line2.strip():
                    countlines += 1
        countlines = countlines - 1  # noqa: PLR6104

        # Create the timeseries
        time = 0
        fileID.write('timeSeries\t' + str(countlines) + '(\n')
        for ii in range(countlines):  # noqa: B007
            fileID.write(str(time) + '\n')
            time = time + frequency  # noqa: PLR6104
        fileID.write(');\n\n')

        # Create the paddle position
        fileID.write('paddlePosition 1(\n' + str(countlines) + '(\n')
        count = 0
        for line in Lines:
            count += 1
            if count > 1:
                if line != '\n':
                    data = float(line)
                    fileID.write(str(data) + '\n')
        fileID.write(')\n);\n\n')

        # Get the water depth and paddle eta
        if heightfilepath != 'None':
            # Get the height
            filewmg = open(heightfilepath)  # noqa: PLW1514, PTH123, SIM115
            Lines2 = filewmg.readlines()  # noqa: N806
            count = 0
            for line in Lines2:
                count += 1
                if count == 1:
                    waterdepth = float(line)
                    break

            # Get the paddle eta
            fileID.write('paddleEta 1(\n' + str(countlines) + '(\n')
            count = 0
            for line in Lines2:
                count += 1
                if count > 1:
                    if line != '\n':
                        data = float(line) + waterdepth
                        fileID.write(str(data) + '\n')
            fileID.write(')\n);')
