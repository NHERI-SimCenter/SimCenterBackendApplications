####################################################################
# OpenFOAM class for Hydro-UQ
####################################################################
import numpy as np
import os
import shutil

class solver(object):
    '''
    This class includes all the general utilities that are
    required for the Hydro-UQ.
    
    Methods
    --------
        dircreate: Method to create necessary directories for the solver
    '''

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