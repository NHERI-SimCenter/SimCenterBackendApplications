####################################################################
# Import necessary packages for bathymetry
####################################################################
import numpy as np
import meshio
from shapely.geometry import Polygon, Point
import triangle as tr

class Bathymetry(object):

    ####################################################################
    # Generate the flume geometry
    ####################################################################
    def generateflume(self,IpPTFile):
        
        # Get the triangulated flume
        self.extreme = self.flumedata(IpPTFile)

        # Right face
        self.right()  # Right vertices
        self.npt_right = self.npt # Right triangles
        self.writeSTL("Right",self.npa_right,self.npt_right) # Write right STL file

        # Left face
        self.left()  # Left vertices
        self.lefttri() # Left triangles
        self.writeSTL("Left",self.npa_left,self.npt_left) # Write left STL file

        # Front face
        self.front() # Front faces
        self.fronttri() # Front triangles
        self.writeSTL("Front",self.npa_front,self.npt_front) # Write front STL file

        # Back face
        self.back() # Back vertices
        self.backtri() # Back triangles
        self.writeSTL("Back",self.npa_back,self.npt_back) # Write back STL file

        # Top face
        self.top() # Top vertices
        self.toptri() # Top triangles
        self.writeSTL("Top",self.npa_top,self.npt_top) # Write top STL file

        # Bottom face
        self.bottom() # Bottom vertices
        self.bottomtri() # Bottom triangles
        self.writeSTL("Bottom",self.npa_bottom,self.npt_bottom) # Write bottom STL file

        # Return the extreme values
        return self.extreme

    ####################################################################
    # Define the triangulated flume
    ####################################################################
    def flumedata(self,IpPTFile):
        # Get the data for the boundary
        data_boun = np.genfromtxt(IpPTFile, delimiter=',',dtype=(float, float))

        # Add extremum to the constants file
        maxvalues = np.max(data_boun,axis=0)
        minvalues = np.min(data_boun,axis=0)
        extremeval = np.array([minvalues[0],maxvalues[0],minvalues[1],maxvalues[1]])

        # Initialize segments for left and right
        segmentLR = []

        # Loop over all coordinates and create coordinates
        for ii in range(0,data_boun.shape[0]):

            # Get each of the user points
            if ii < data_boun.shape[0]-1:
                segmentLR.extend([(ii, ii+1)])
            else:
                segmentLR.extend([(ii, 0)]) 

        # Triangulate the polygon
        ALR = dict(vertices=data_boun,segments=segmentLR)
        BLR = tr.triangulate(ALR)

        # Get the tringles and vertices
        nm_triangle = BLR['triangles'].tolist()
        self.npt = np.asarray(nm_triangle, dtype=np.int32)
        nm_vertices = BLR['vertices'].tolist()
        self.npa = np.asarray(nm_vertices, dtype=np.float32)

        # Define the polygon
        mypoly = Polygon(data_boun)

        # Loop over all triangles to find if inside polygon
        indexes = []
        noindexes = []
        for ii in range(0,self.npt.shape[0]):
            n0 = self.npt[ii,0]
            n1 = self.npt[ii,1]
            n2 = self.npt[ii,2]
            centroidX = (1/3)*(self.npa[n0,0]+self.npa[n1,0]+self.npa[n2,0])
            centroidZ = (1/3)*(self.npa[n0,1]+self.npa[n1,1]+self.npa[n2,1])
            po = Point(centroidX,centroidZ)    
            if mypoly.contains(po):
                indexes.extend([(ii)])
            else:
                noindexes.extend([(ii)])

        # Delete extra triangles
        self.npt = np.delete(self.npt, noindexes, axis=0)

        # Return extreme values
        return extremeval

    ####################################################################
    # Define the nodes of the triangulated right face
    ####################################################################
    def right(self):
        self.npa_right = np.zeros(shape=(self.npa.shape[0],3))
        self.npa_right[:,0] = self.npa[:,0]
        self.npa_right[:,2] = self.npa[:,1]
        self.npa_right[:,1] = -self.breadth/2

    ####################################################################
    # Define the nodes of the triangulated left face
    ####################################################################    
    def left(self):
        self.npa_left = np.zeros(shape=(self.npa.shape[0],3))
        self.npa_left[:,0] = self.npa[:,0]
        self.npa_left[:,2] = self.npa[:,1]
        self.npa_left[:,1] = self.breadth/2

    ####################################################################
    # Define the triangles of the triangulated left face
    #################################################################### 
    def lefttri(self):
        self.npt_left = np.array(self.npt)
        self.npt_left[:, [1, 0]] = self.npt_left[:, [0, 1]]

    ####################################################################
    # Define the nodes of the triangulated front face
    #################################################################### 
    def front(self):
        self.npa_front = np.zeros(shape=(4,3))
        self.npa_front[0,:] = self.npa_right[0,:]
        self.npa_front[1,:] = self.npa_right[self.npa_right.shape[0]-1,:]
        self.npa_front[2,:] = self.npa_left[0,:]
        self.npa_front[3,:] = self.npa_left[self.npa_left.shape[0]-1,:]

    ####################################################################
    # Define the triangles of the triangulated front face
    #################################################################### 
    def fronttri(self):
        self.npt_front = np.array([[0,1,2], [1,3,2]])

    ####################################################################
    # Define the nodes of the triangulated back face
    #################################################################### 
    def back(self):
        self.npa_back = np.zeros(shape=(4,3))
        self.npa_back[0,:] = self.npa_right[self.npa_right.shape[0]-3,:]
        self.npa_back[1,:] = self.npa_right[self.npa_right.shape[0]-2,:]
        self.npa_back[2,:] = self.npa_left[self.npa_left.shape[0]-3,:]
        self.npa_back[3,:] = self.npa_left[self.npa_left.shape[0]-2,:]

    ####################################################################
    # Define the triangles of the triangulated back face
    #################################################################### 
    def backtri(self):
        self.npt_back = np.array([[3,1,0], [0,2,3]])

    ####################################################################
    # Define the nodes of the triangulated top face
    #################################################################### 
    def top(self):
        self.npa_top = np.zeros(shape=(4,3))
        self.npa_top[0,:] = self.npa_right[self.npa_right.shape[0]-1,:]
        self.npa_top[1,:] = self.npa_right[self.npa_right.shape[0]-2,:]
        self.npa_top[2,:] = self.npa_left[self.npa_left.shape[0]-1,:]
        self.npa_top[3,:] = self.npa_left[self.npa_left.shape[0]-2,:]

    ####################################################################
    # Define the triangles of the triangulated top face
    #################################################################### 
    def toptri(self):
        self.npt_top = np.array([[2,0,1], [2,1,3]])

    ####################################################################
    # Define the nodes of the triangulated bottom face
    #################################################################### 
    def bottom(self):
        # Create the coordinate vector
        self.npa_bottom = []

        # Loop over all the points
        for ii in range(0,self.npa_right.shape[0]-3):
            npa_temp1 = np.zeros(shape=(4,3))
            npa_temp2 = np.zeros(shape=(2,3))
            
            # Get the points
            if ii ==0:
                npa_temp1[0,:] = self.npa_right[ii,:]
                npa_temp1[1,:] = self.npa_left[ii,:]
                npa_temp1[2,:] = self.npa_right[ii+1,:]
                npa_temp1[3,:] = self.npa_left[ii+1,:]
            else:
                npa_temp2[0,:] = self.npa_right[ii+1,:]
                npa_temp2[1,:] = self.npa_left[ii+1,:]
            
            # Concatenate as necessary
            if ii==0:
                self.npa_bottom = npa_temp1
            else:
                self.npa_bottom = np.concatenate((self.npa_bottom,npa_temp2),axis=0)

    ####################################################################
    # Define the triangles of the triangulated bottom face
    #################################################################### 
    def bottomtri(self):
        # Create the coordinate vector
        self.npt_bottom = []
        ntri = 2

        # Loop over all the points
        for ii in range(0,self.npa_right.shape[0]-3):
            npt_temp = np.zeros(shape=(2,3))
            
            # Get the triangles
            npt_temp = np.array([[0,1,2], [1,3,2]])
            npt_temp = npt_temp + ii*ntri
            
            # Concatenate as necessary
            if ii==0:
                self.npt_bottom = npt_temp
            else:
                self.npt_bottom = np.concatenate((self.npt_bottom,npt_temp),axis=0) 

    ####################################################################
    # Write out an STL file and add solid name to first and last line
    #################################################################### 
    # Write the STL file
    def writeSTL(self,base_filename,npa,npt):
        # Create a filename
        filename = base_filename + ".stl"
        # Create the STL file
        cells = [("triangle", npt)]
        meshio.write_points_cells(filename, npa, cells)
        # Modify first and last line
        with open(filename) as f:
            lines = f.readlines()
            lines[0] = 'solid '+ base_filename + '\n'
            lines[len(lines)-1] = 'endsolid ' + base_filename + '\n'
        # Write the updated file    
        with open(filename, "w") as f:
            f.writelines(lines)
