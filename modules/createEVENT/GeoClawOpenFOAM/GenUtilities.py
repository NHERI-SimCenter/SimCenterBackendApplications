####################################################################
# General utilities class for Hydro-UQ
####################################################################
class genUtilities(object):
    '''
    This class includes all the general utilities that are
    required for the Hydro-UQ.
    
    Methods
    --------
        extract: Extracts an element from a nested json 
        extract_element_from_json: Extracts an element from a nested json
        hydrolog: Initializes the log file
        general_header: Creates the header for the Hydro-UQ files
    '''

    ####################################################################
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

    ####################################################################
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

    ####################################################################
    def hydrolog(self,projname):
        '''
        Used to initialize the log file for the Hydro-UQ program

        Arguments
        -----------
            projname: Name of the project as given by the user

        Variables
        -----------
            flog: File pointer to the log file
        '''

        # Open a log file to write the outputs
        # Use project name for the log file
        # If no project name is specified, call it Untitled
        if projname != "":
            self.flog = open(''.join(projname.split())+".h20log","w")
        else:
            self.flog = open("Untitled.h20log","w")

    ####################################################################
    def general_header(self):
        '''
        Used to create a general header for Hydro-UQ related files

        Variables
        -----------
            header: Stores the general header for the Hydro-UQ files
        '''

        header = """/*--------------------------*- NHERI SimCenter -*----------------------------*\ 
|       | H  |
|       | Y  | HydroUQ: Water-based Natural Hazards Modeling Application
|=======| D  | Website: simcenter.designsafe-ci.org/research-tools/hydro-uq
|       | R  | Version: 1.00
|       | O  |
\*---------------------------------------------------------------------------*/ \n\n"""
        
        return header

