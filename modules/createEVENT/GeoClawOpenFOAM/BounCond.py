####################################################################
# Import necessary packages for OpenFOAM boundary condition class
####################################################################
import numpy as np
import meshio
import os
from shapely.geometry import Polygon, Point
import triangle as tr

from GenUtilities import genUtilities # General utilities

class OFBounCond(object):

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
    def bountypenum(self,data,patchname,bcfield):
        '''
        Method to determine the type of boundary condition

        Variables
        ------------
            patch: String array 
            patchtype: Int variable with boundary type

        '''

        # Get an object for utilities
        hydroutil = genUtilities()

        # Get the patch
        patch = hydroutil.extract_element_from_json(data, ["Events",bcfield+"Type_"+patchname])
        if patch == [None]:
            # Default if none is selected - this is wall
            patchtype = -1 
        else:
            # Get the boundary condition selected
            patchtype = ', '.join(hydroutil.extract_element_from_json(data, ["Events",bcfield+"Type_"+patchname]))

        return patchtype

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
    def OSUwavemakerfile(self,dispfilename,heightfilename):
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
        filewm = open(dispfilename,'r')
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
        countlines = 0
        with open(dispfilename) as fdisp:
            for line2 in fdisp:
                if line2.strip():
                    countlines += 1
        countlines = countlines - 72

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
                if line != "\n":
                    data = float(line)
                    fileID.write(str(data)+'\n')
        fileID.write(')\n);\n\n')

        # Write the paddle Eta
        fileID.write('paddleEta 1(\n'+str(countlines)+'(\n')
        filewmg = open(heightfilename,'r')
        Lines2 = filewmg.readlines()
        count = 0
        for line in Lines2:
            count += 1
            if count > 72:
                if line != "\n":
                    data = float(line)+waterdepth
                    fileID.write(str(data)+'\n')
        fileID.write(')\n);')

    ####################################################################
    def GENwavemakerfile(self,dispfilename,heightfilename):
        '''
        This method is used to write the wavemaker movement file
        '''
        # Open the wavemaker movement file
        fileID = open("constant/wavemakerMovement.txt","w")
        fileID.write('wavemakerType\tPiston;\n')
        fileID.write('tSmooth\t1.5;\n')
        fileID.write('genAbs\t0;\n\n')

        # Get the frequency
        filewmdisp = open(dispfilename,'r') 
        Lineswmdisp = filewmdisp.readlines()
        count = 0
        for line in Lineswmdisp:
            count += 1
            if count == 1:
                frequency = float(line)
                break

        # Count the number of lines
        countlines = 0
        with open(dispfilename) as fdisp:
            for line2 in fdisp:
                if line2.strip():
                    countlines += 1
        countlines = countlines - 1

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
        for line in Lineswmdisp:
            count += 1
            if count > 1:
                if line != "\n":
                    data = float(line)
                    fileID.write(str(data)+'\n')
        fileID.write(')\n);\n\n')

        # Get the water depth
        filewmwg = open(heightfilename,'r')
        Lineswmwg = filewmwg.readlines()
        count = 0
        for line in Lineswmwg:
            count += 1
            if count == 1:
                waterdepth = float(line)
                break

        # Write the paddle Eta
        fileID.write('paddleEta 1(\n'+str(countlines)+'(\n')
        count = 0
        for line in Lineswmwg:
            count += 1
            if count > 1:
                if line != "\n":
                    data = float(line)+waterdepth
                    fileID.write(str(data)+'\n')
        fileID.write(')\n);')

    ####################################################################
    def Uboun(self,data,velbountype,patchname,fpath):

        # Get an object for utilities
        hydroutil = genUtilities()

        # Get text for different velocity bc
        if int(velbountype) == 1: # SW solutions
            patchtext = "\t"+patchname+"\n\t{\n\t\t"
            patchtext = patchtext+"type\ttimeVaryingMappedFixedValue;\n\t\t"
            patchtext = patchtext+"offset\t(0 0 0);\n\t\t"
            patchtext = patchtext+"setAverage\toff;\n\t}"

            # Set up the file in constant/boundary
            # TO BE COMPLETED

        elif int(velbountype) == 2: # Inlet: constant velocity
            # Get value of velocity
            velvalues = ', '.join(hydroutil.extract_element_from_json(data, ["Events","Velocity_"+patchname]))
            velvalues = velvalues.replace(',', ' ')
            vels = [float(n) for n in velvalues.split()]

            # Get the patch text
            patchtext = "\t"+patchname+"\n\t{\n\t\t"
            patchtext = patchtext+"type\tfixedValue;\n\t\t"
            patchtext = patchtext+"value\tuniform("+str(vels[0])+"\t"+str(vels[1])+"\t"+str(vels[2])+");\n\t}"

        elif (int(velbountype) == 3) or (int(velbountype) == 4): # OSU / General flume
            patchtext = "\t"+patchname+"\n\t{\n\t\t"
            patchtext = patchtext+"type\tmovingWallVelocity;\n\t\t"
            patchtext = patchtext+"value\tuniform(0 0 0);\n\t}"

            # Create the dynamic mesh dictionary
            self.dynmeshFlumeOF()
            # Create the wavemaker dictionary
            self.wavemakerOF()

            # Create the movement file
            if int(velbountype) == 3:
                # Get the displacement and waterheight file name
                dispfilename = ', '.join(hydroutil.extract_element_from_json(data, ["Events","OSUMovingWallDisp_"+patchname]))
                dispfilename = os.path.join(fpath,dispfilename)
                heightfilename = ', '.join(hydroutil.extract_element_from_json(data, ["Events","OSUMovingWallHeight_"+patchname]))
                heightfilename = os.path.join(fpath,heightfilename)
                self.OSUwavemakerfile(dispfilename,heightfilename)
            elif int(velbountype) == 4:
                # Get the displacement and waterheight file name
                dispfilename = ', '.join(hydroutil.extract_element_from_json(data, ["Events","MovingWallDisp_"+patchname]))
                dispfilename = os.path.join(fpath,dispfilename)
                heightfilename = ', '.join(hydroutil.extract_element_from_json(data, ["Events","MovingWallHeight_"+patchname]))
                heightfilename = os.path.join(fpath,heightfilename)
                self.GENwavemakerfile(dispfilename,heightfilename)

        elif int(velbountype) == 5: # Outlet (zeroGradient)
            patchtext = "\t"+patchname+"\n\t{\n\t\t"
            patchtext = patchtext+"type\tzeroGradient;\n\t}"

        elif int(velbountype) == 6: # Outlet (inletOutlet)
            # Get value of velocity
            velvalues = ', '.join(hydroutil.extract_element_from_json(data, ["Events","Velocity_"+patchname]))
            velvalues = velvalues.replace(',', ' ')
            vels = [float(n) for n in velvalues.split()]

            # Get the patch text
            patchtext = "\t"+patchname+"\n\t{\n\t\t"
            patchtext = patchtext+"type\tinletOutlet;\n\t\t"
            patchtext = patchtext+"inletValue\tuniform("+str(vels[0])+"\t"+str(vels[1])+"\t"+str(vels[2])+");\n\t}"

        elif int(velbountype) == 7: # Wall (noSlip)
            patchtext = "\t"+patchname+"\n\t{\n\t\t"
            patchtext = patchtext+"type\tnoSlip;\n\t}"

        else: # Empty (0 and -1)
            patchtext = "\t"+patchname+"\n\t{\n\t\t"
            patchtext = patchtext+"type\tempty;\n\t}"

        return patchtext