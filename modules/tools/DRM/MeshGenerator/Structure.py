# %%
import pyvista as pv 
import numpy as np
import ctypes 
import os 
import sys
# =============================================================================
# information
# =============================================================================
# getting the meshing information from the command line

# xwidth, ywidth, zwidth           = (float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]))
# eps                              = 1e-6
# Xmeshsize, Ymeshsize, Zmeshsize  = (float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]))
# PMLThickness                     = np.array([float(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9])])            ; # thickness of the each PML layer
# numPMLLayers                     = int(sys.argv[10])                                 ; # number of PML layers
# PMLTotalThickness                = PMLThickness * numPMLLayers       ; # total thickness of the PML layers
# DRMThickness                     = np.array([float(sys.argv[11]), float(sys.argv[12]), float(sys.argv[13])])            ; # thickness of the DRM layers
# numDrmLayers                     = int(sys.argv[14])                                 ; # number of DRM layers
# DRMTotalThickness                = DRMThickness * numDrmLayers       ; # total thickness of the DRM layers
# padLayers                        = numPMLLayers + numDrmLayers       ; # number of layers to pad the meshgrid
# padThickness                     = PMLTotalThickness + DRMThickness  ; # thickness of the padding layers
# reg_num_cores                    = int(sys.argv[15])
# DRM_num_cores                    = int(sys.argv[16])
# PML_num_cores                    = int(sys.argv[17])
# Dir                              = sys.argv[18]
# OutputDir                        = sys.argv[19]
# pileboxcenterx                   = float(sys.argv[20])
# pileboxcentery                   = float(sys.argv[21])
# pileboxcenterz                   = float(sys.argv[22])
# pileboxlx                        = float(sys.argv[23])
# pileboxly                        = float(sys.argv[24])
# pileboxdepth                     = float(sys.argv[25])
# AbsorbingElements                = sys.argv[26]
# DRMfile                          = sys.argv[27]



xwidth                          = 100
ywidth                          = 100
zwidth                          = 40
eps                             = 1e-6
Xmeshsize, Ymeshsize, Zmeshsize = (2.5, 2.5, 2.5)
PMLThickness                    = np.array([Xmeshsize, Ymeshsize, Zmeshsize])            ; # thickness of the each PML layer
numPMLLayers                    = 2                                 ; # number of PML layers
PMLTotalThickness               = PMLThickness * numPMLLayers       ; # total thickness of the PML layers
DRMThickness                    = np.array([Xmeshsize, Ymeshsize, Zmeshsize])            ; # thickness of the DRM layers                                      
numDrmLayers                    = 1                                 ; # number of DRM layers
DRMTotalThickness               = DRMThickness * numDrmLayers       ; # total thickness of the DRM layers
padLayers                       = numPMLLayers + numDrmLayers       ; # number of layers to pad the meshgrid
padThickness                    = PMLTotalThickness + DRMThickness  ; # thickness of the padding layers
reg_num_cores                   = 3
DRM_num_cores                   = 3
PML_num_cores                   = 3
Dir                             = "OpenSeesMesh"
OutputDir                       = "results"
pileboxcenterx                  = 0
pileboxcentery                  = 0
pileboxcenterz                  = 0
pileboxlx                       = 20
pileboxly                       = 20
pileboxdepth                    = 10.0
AbsorbingElements                = "PML"
DRMfile                          = "SurfaceWave.h5drm"


# # print the input information
# print(f"xwidth: {xwidth}")
# print(f"ywidth: {ywidth}")
# print(f"zwidth: {zwidth}")
# print(f"Xmeshsize: {Xmeshsize}")
# print(f"Ymeshsize: {Ymeshsize}")
# print(f"Zmeshsize: {Zmeshsize}")
# print(f"PMLThickness: {PMLThickness}")
# print(f"numPMLLayers: {numPMLLayers}")
# print(f"PMLTotalThickness: {PMLTotalThickness}")
# print(f"DRMThickness: {DRMThickness}")
# print(f"numDrmLayers: {numDrmLayers}")
# print(f"DRMTotalThickness: {DRMTotalThickness}")
# print(f"padLayers: {padLayers}")
# print(f"padThickness: {padThickness}")
# print(f"reg_num_cores: {reg_num_cores}")
# print(f"DRM_num_cores: {DRM_num_cores}")
# print(f"PML_num_cores: {PML_num_cores}")
# print(f"Dir: {Dir}")
# print(f"OutputDir: {OutputDir}")
# print(f"pileboxcenterx: {pileboxcenterx}")
# print(f"pileboxcentery: {pileboxcentery}")
# print(f"pileboxcenterz: {pileboxcenterz}")
# print(f"pileboxlx: {pileboxlx}")
# print(f"pileboxly: {pileboxly}")
# print(f"pileboxdepth: {pileboxdepth}")
# print(f"AbsorbingElements: {AbsorbingElements}")
# print(f"DRMfile: {DRMfile}")

# create a dictionary for meshing information
info = {
    "RegularDomain": 1,
    "DRMDomain": 2,
    "PMLDomain": 3,
}

pileboxinfo = {}
# adding different plots
meshplotdir = OutputDir + "/meshplots"
if not os.path.exists(meshplotdir):
    os.makedirs(meshplotdir)
# =============================================================================
# buiding an structure mesh
# =============================================================================
embededDepth = {
    "xmax": 20,
    "xmin": -20,
    "ymax": 15,
    "ymin": -15,
    "zmax": 0,
    "zmin": -5,
}



foundationBlocks = []

cnetersx = np.arange(-15,15+eps,7.5)
cnetersy = np.arange(-7.5,7.5+eps,7.5)
cnetersx,cnetersy = np.meshgrid(cnetersx,cnetersy)
# make them tuples 
centers = np.vstack((cnetersx.flatten(),cnetersy.flatten())).T

for i in range(len(centers)):
    blockinfo = {
        "xmax" : centers[i][0] + 2,
        "xmin" : centers[i][0] - 2,
        "ymax" : centers[i][1] + 2,
        "ymin" : centers[i][1] - 2,
        "zmax" : -3,
        "zmin" : -5,
        "Xmeshsize": 1.0,
        "Ymeshsize": 1.0,
        "Zmeshsize": 1.0,
    }
    foundationBlocks.append(blockinfo)

# =============================================================================
# define the piles
# =============================================================================
pilelist = []
for ii in range(len(centers)):

    for tu in [(1,1),(1,-1),(-1,1),(-1,-1)]:
        i,j = tu
        pilelist.append({
            "xtop": centers[ii][0] + i,
            "ytop": centers[ii][1] + j,
            "ztop": -4,
            "xbottom": centers[ii][0] + i,
            "ybottom": centers[ii][1] + j,
            "zbottom": -10.0,
        })



# =============================================================================
# define partioner
# =============================================================================
# change the directory to the current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
libpath = os.getcwd().split("OpenSeesProjects")[0] + "OpenSeesProjects/" + "MeshGenerator/lib"
print(libpath)
if os.name == 'nt':
    metis_partition_lib = ctypes.CDLL(f'{libpath}/Partitioner.dll')
if os.name == 'posix':
    metis_partition_lib = ctypes.CDLL(f'{libpath}/libPartitioner.so')

# Define function argument and return types
metis_partition_lib.Partition.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.POINTER(ctypes.c_int32)]
metis_partition_lib.Partition.restype  = ctypes.c_int

def partition(mesh, numcores):
    numcells = mesh.n_cells
    numpoints = mesh.n_points
    numweights = 1
    cells = np.array(mesh.cells.reshape(-1, 9), dtype=np.int32)
    cells = cells[:,1:]
    cellspointer = cells.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    partitioned = np.empty(numcells, dtype=np.int32)
    partitionedpointer = partitioned.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    metis_partition_lib.Partition(numcells, numpoints, cellspointer, numcores, partitionedpointer)
    mesh.cell_data['partitioned'] = partitioned




def partition_with_pilebox(messh,numcores, pileboxinfo,tol=1e-6):
    # seperate the core of the pilebox
    mesh = messh.copy()
    eps = 1e-6
    # check if the xmin,xmax, ymin ymax ,zmin zmax is exist in the infokeys
    if "xmin" in pileboxinfo.keys():
        pilebox_xmin = pileboxinfo["xmin"]
    if "xmax" in pileboxinfo.keys():
        pilebox_xmax = pileboxinfo["xmax"]
    if "ymin" in pileboxinfo.keys():
        pilebox_ymin = pileboxinfo["ymin"]
    if "ymax" in pileboxinfo.keys():
        pilebox_ymax = pileboxinfo["ymax"]
    if "zmin" in pileboxinfo.keys():
        pilebox_zmin = pileboxinfo["zmin"]
    if "zmax" in pileboxinfo.keys():
        pilebox_zmax = pileboxinfo["zmax"]
    if "center" in pileboxinfo.keys():
        pilebox_xmin = pileboxinfo["center"][0] - pileboxinfo["lx"]/2 + eps
        pilebox_xmax = pileboxinfo["center"][0] + pileboxinfo["lx"]/2 - eps
        pilebox_ymin = pileboxinfo["center"][1] - pileboxinfo["ly"]/2 + eps
        pilebox_ymax = pileboxinfo["center"][1] + pileboxinfo["ly"]/2 - eps
        pilebox_zmin = pileboxinfo["center"][2] - pileboxinfo["depth"] + eps
        pilebox_zmax = pileboxinfo["center"][2] + tol + eps
    

    # find the cells that are inside the pilebox
    cube = pv.Cube(bounds=[pilebox_xmin,pilebox_xmax,pilebox_ymin,pilebox_ymax,pilebox_zmin,pilebox_zmax])
    # extract the cells that are outside the pilebox
    indices = mesh.find_cells_within_bounds(cube.bounds)
    
    # crete newindices for outside the pilebox
    newindices = np.ones(mesh.n_cells,dtype=bool)
    newindices[indices] = False
    newindices = np.where(newindices)[0]


    # extract the cells that are inside the pilebox
    mesh2 = mesh.extract_cells(newindices)
    # partition the mesh
    
    if numcores > 2:
        partition(mesh2,numcores-1)
    if numcores == 2:
        mesh2.cell_data['partitioned'] = np.zeros(mesh2.n_cells,dtype=np.int32)
    
    mesh.cell_data['partitioned'] = np.zeros(mesh.n_cells,dtype=np.int32)
    mesh.cell_data['partitioned'][newindices] = mesh2.cell_data['partitioned'] + 1
    messh.cell_data['partitioned'] = mesh.cell_data['partitioned']


# =============================================================================
# meshing
# =============================================================================
x = np.arange(-xwidth/2., xwidth/2.+eps, Xmeshsize)
y = np.arange(-ywidth/2., ywidth/2.+eps, Ymeshsize)
z = np.arange(-zwidth, 0+eps, Zmeshsize)

# padding x and y for PML and DRM layers
x  = np.pad(x, (numDrmLayers,numDrmLayers), "linear_ramp", end_values=(x.min()-DRMTotalThickness[0], x.max()+DRMTotalThickness[0]))
y  = np.pad(y, (numDrmLayers,numDrmLayers), "linear_ramp", end_values=(y.min()-DRMTotalThickness[1], y.max()+DRMTotalThickness[1]))
z  = np.pad(z, (numDrmLayers,0), "linear_ramp", end_values=(z.min()-DRMTotalThickness[2]))

# padding the x and y for PML and PML layers
x  = np.pad(x, (numPMLLayers,numPMLLayers), "linear_ramp", end_values=(x.min()-PMLTotalThickness[0], x.max()+PMLTotalThickness[0]))
y  = np.pad(y, (numPMLLayers,numPMLLayers), "linear_ramp", end_values=(y.min()-PMLTotalThickness[1], y.max()+PMLTotalThickness[1]))
z  = np.pad(z, (numPMLLayers,0), "linear_ramp", end_values=(z.min()-PMLTotalThickness[2]))

# %%
x, y, z = np.meshgrid(x, y, z, indexing='ij')

mesh = pv.StructuredGrid(x, y, z)
# =============================================================================
# sperate embedding layer 
# =============================================================================
cube = pv.Cube(bounds=[embededDepth["xmin"],embededDepth["xmax"],embededDepth["ymin"],embededDepth["ymax"],embededDepth["zmin"],embededDepth["zmax"]])
mesh = mesh.clip_box(cube,invert=True,crinkle=True,progress_bar = True)
mesh.clear_data()
# =============================================================================
# Add foundation blocks
# =============================================================================
for i,block in enumerate(foundationBlocks):
    xBLOCK = np.arange(block["xmin"], block["xmax"]+eps, block["Xmeshsize"])
    yBLOCK = np.arange(block["ymin"], block["ymax"]+eps, block["Ymeshsize"])
    zBLOCK = np.arange(block["zmin"], block["zmax"]+eps, block["Zmeshsize"])
    xBLOCK, yBLOCK, zBLOCK = np.meshgrid(xBLOCK, yBLOCK, zBLOCK, indexing='ij')
    if i == 0:
        foundation = pv.StructuredGrid(xBLOCK, yBLOCK, zBLOCK)
    else:
        foundation = foundation.merge(pv.StructuredGrid(xBLOCK, yBLOCK, zBLOCK),merge_points=False,tolerance=1e-6,progress_bar = True)



# =============================================================================
# adding piles
# =============================================================================
pilenodes = np.zeros((len(pilelist)*2,3))
pileelement = np.zeros((len(pilelist),3),dtype=int)
for i in range(len(pilelist)):
    pilenodes[i*2] = [pilelist[i]["xtop"],pilelist[i]["ytop"],pilelist[i]["ztop"]]
    pilenodes[i*2+1] = [pilelist[i]["xbottom"],pilelist[i]["ybottom"],pilelist[i]["zbottom"]]
    pileelement[i] = [2,2*i,2*i+1]
celltypes  = np.ones(pileelement.shape[0],dtype= int) * pv.CellType.LINE
piles = pv.UnstructuredGrid(pileelement.tolist(),celltypes.tolist(),pilenodes.tolist())
    
# %%
pl = pv.Plotter()
pl.add_mesh(piles, show_edges=True, color = "r" , line_width=4.0,)
pl.add_mesh(foundation, color="gray", opacity=0.5 )
pl.add_mesh(mesh, opacity=0.5 )
pl.camera_position = 'xz'
pl.export_html(meshplotdir+"/pile.html")
# pl.show()

# %%
# merge the piles and foundation
foundation = foundation.merge(piles,merge_points=False,tolerance=1e-6,progress_bar = True)
foundationbounds = foundation.bounds
pileboxinfo["xmin"] = foundationbounds[0]
pileboxinfo["xmax"] = foundationbounds[1]
pileboxinfo["ymin"] = foundationbounds[2]
pileboxinfo["ymax"] = foundationbounds[3]
pileboxinfo["zmin"] = foundationbounds[4]
pileboxinfo["zmax"] = foundationbounds[5]
mesh = mesh.merge(foundation,merge_points=False,tolerance=1e-6,progress_bar = True)
# %%
# =============================================================================
# sperate PML layer 
# =============================================================================
xmin = x.min() + PMLTotalThickness[0]
xmax = x.max() - PMLTotalThickness[0]
ymin = y.min() + PMLTotalThickness[1]
ymax = y.max() - PMLTotalThickness[1]
zmin = z.min() + PMLTotalThickness[2]
zmax = z.max() + PMLTotalThickness[2]
cube = pv.Cube(bounds=[xmin,xmax,ymin,ymax,zmin,zmax])
PML = mesh.clip_box(cube,invert=True, crinkle=True, progress_bar = True)
reg = mesh.clip_box(cube,invert=False,crinkle=True, progress_bar = True)



    




# reg.plot(show_edges=True,show_grid=True,show_axes=True,show_bounds=True)
pl = pv.Plotter()
pl.add_mesh(reg, show_edges=True, style="wireframe" )
pl.show()



# %%
# now find DRM layer
indices = reg.find_cells_within_bounds([xmin + DRMTotalThickness[0] + eps,
                              xmax - DRMTotalThickness[0] - eps,
                              ymin + DRMTotalThickness[1] + eps,
                              ymax - DRMTotalThickness[1] - eps,
                              zmin + DRMTotalThickness[2] + eps,
                              zmax + DRMTotalThickness[2] + eps])
# print(DRMTotalThickness)
# print(xmax, xmin, ymax, ymin, zmax, zmin)
# pl = pv.Plotter()
# pl.add_mesh(reg,show_edges=True)
# cube = pv.Cube(bounds=[xmin + DRMTotalThickness[0] + eps,
#                        xmax - DRMTotalThickness[0] - eps,
#                        ymin + DRMTotalThickness[1] + eps,
#                        ymax - DRMTotalThickness[1] - eps,
#                        zmin + DRMTotalThickness[2] + eps,
#                        zmax + DRMTotalThickness[2] + eps])
# pl.add_mesh(cube,show_edges=True)
# pl.show()
# exit
# now create complemntary indices for DRM
DRMindices = np.ones(reg.n_cells,dtype=bool)
DRMindices[indices] = False
DRMindices = np.where(DRMindices)[0]



reg.cell_data['Domain'] = np.ones(reg.n_cells,dtype=np.int8)*info["DRMDomain"]
reg.cell_data['Domain'][indices] = info["RegularDomain"]
PML.cell_data['Domain'] = np.ones(PML.n_cells,dtype=np.int8)*info["PMLDomain"]
reg.cell_data['partitioned'] = np.zeros(reg.n_cells,dtype=np.int32)
PML.cell_data['partitioned'] = np.zeros(PML.n_cells,dtype=np.int32)


# partitioning regular mesh
regular = reg.extract_cells(indices,progress_bar=True)
DRM     = reg.extract_cells(DRMindices,progress_bar=True)

if reg_num_cores > 1:
    partition_with_pilebox(regular,reg_num_cores,pileboxinfo,tol=10)
if DRM_num_cores > 1:
    partition(DRM,DRM_num_cores)
if PML_num_cores > 1:
    partition(PML,PML_num_cores)

reg.cell_data['partitioned'][regular["vtkOriginalCellIds"]] = regular.cell_data['partitioned']
reg.cell_data['partitioned'][DRM["vtkOriginalCellIds"]] = DRM.cell_data['partitioned'] + reg_num_cores
PML.cell_data['partitioned'] = PML.cell_data['partitioned'] + reg_num_cores + DRM_num_cores

# %%
# merging PML and regular mesh to create a single mesh
mesh = reg.merge(PML,merge_points=False,tolerance=1e-6,progress_bar = True)


# mapping between PML and regular mesh on the boundary
mapping = mesh.clean(produce_merge_map=True)["PointMergeMap"]
regindicies = np.where(mapping[PML.n_points:]<PML.n_points)[0]
PMLindicies = mapping[PML.n_points+regindicies]


mesh.point_data["boundary"] = np.zeros(mesh.n_points,dtype=int)-1
mesh.point_data["boundary"][PMLindicies] = regindicies + PML.n_points
mesh.point_data["boundary"][PML.n_points + regindicies] = PMLindicies 

indices = np.where(mesh.point_data["boundary"]>0)[0]

# %%
mesh["matTag"] = np.ones(mesh.n_cells,dtype=np.uint8)





# %%
# define the ASDA absorbing elements
if AbsorbingElements == "ASDA":

    mesh = mesh.clean(tolerance=1e-6,remove_unused_points=False)
    mesh["ASDA_type"] = np.zeros(mesh.n_cells,dtype=np.uint8)

    ASDAelem_type = {
        "B"   :1 ,
        "L"   :2 , 
        "R"   :3 ,
        "F"   :4 ,
        "K"   :5 ,
        "BL"  :6 ,
        "BR"  :7 ,
        "BF"  :8 ,
        "BK"  :9 ,
        "LF"  :10,
        "LK"  :11,
        "RF"  :12,
        "RK"  :13,
        "BLF" :14,
        "BLK" :15,
        "BRF" :16,
        "BRK" :17,
    }

    ASDAelem_typereverse = {
        1 :"B"   ,
        2 :"L"   ,
        3 :"R"   ,
        4 :"F"   ,
        5 :"K"   ,
        6 :"BL"  ,
        7 :"BR"  ,
        8 :"BF"  ,
        9 :"BK"  ,
        10:"LF"  ,
        11:"LK"  ,
        12:"RF"  ,
        13:"RK"  ,
        14:"BLF" ,
        15:"BLK" ,
        16:"BRF" ,
        17:"BRK" ,   
    }
    xmin, xmax, ymin, ymax, zmin, zmax = reg.bounds
    ASDA_xwidth = xmax - xmin
    ASDA_ywidth = ymax - ymin
    ASDA_zwidth = zmax - zmin
    print("ASDA_xwidth", ASDA_xwidth)
    print("ASDA_ywidth", ASDA_ywidth)
    print("ASDA_zwidth", ASDA_zwidth)

    i = 0
    for ele_center in mesh.cell_centers().points:
        # check if the element is in the left or rightside
        if ele_center[0] < (-ASDA_xwidth/2.):
            # it is in the left side
            # check if it is in the front or back
            if ele_center[1] < (-ASDA_ywidth/2.):
                # it is in the back 
                # check if it is in the bottom or top
                if ele_center[2] < -ASDA_zwidth:
                    # it is in the bottom
                    mesh["ASDA_type"][i] = ASDAelem_type["BLK"]
                else:
                    # it is in the top
                    mesh["ASDA_type"][i] = ASDAelem_type["LK"]
            elif ele_center[1] > (ASDA_ywidth/2.):
                # it is in the front
                # check if it is in the bottom or top
                if ele_center[2] < -ASDA_zwidth:
                    # it is in the bottom
                    mesh["ASDA_type"][i] = ASDAelem_type["BLF"]
                else:
                    # it is in the top
                    mesh["ASDA_type"][i] = ASDAelem_type["LF"]
            else:
                # it is in the middle
                # check if it is in the bottom or top
                if ele_center[2] < -ASDA_zwidth:
                    # it is in the bottom
                    mesh["ASDA_type"][i] = ASDAelem_type["BL"]
                else:
                    # it is in the top
                    mesh["ASDA_type"][i] = ASDAelem_type["L"]

        elif ele_center[0] > (ASDA_xwidth/2.):
            # it is in the right side
            # check if it is in the front or back
            if ele_center[1] < (-ASDA_ywidth/2.):
                # it is in the back 
                # check if it is in the bottom or top
                if ele_center[2] < -ASDA_zwidth:
                    # it is in the bottom
                    mesh["ASDA_type"][i] = ASDAelem_type["BRK"]
                else:
                    # it is in the top
                    mesh["ASDA_type"][i] = ASDAelem_type["RK"]
            elif ele_center[1] > (ASDA_ywidth/2.):
                # it is in the front
                # check if it is in the bottom or top
                if ele_center[2] < -ASDA_zwidth:
                    # it is in the bottom
                    mesh["ASDA_type"][i] = ASDAelem_type["BRF"]
                else:
                    # it is in the top
                    mesh["ASDA_type"][i] = ASDAelem_type["RF"]
            else:
                # it is in the middle
                # check if it is in the bottom or top
                if ele_center[2] < -ASDA_zwidth:
                    # it is in the bottom
                    mesh["ASDA_type"][i] = ASDAelem_type["BR"]
                else:
                    # it is in the top
                    mesh["ASDA_type"][i] = ASDAelem_type["R"]
        else:
            # it is in the middle
            # check if it is in the front or back
            if ele_center[1] < (-ASDA_ywidth/2.):
                # it is in the back 
                # check if it is in the bottom or top
                if ele_center[2] < -ASDA_zwidth:
                    # it is in the bottom
                    mesh["ASDA_type"][i] = ASDAelem_type["BK"]
                else:
                    # it is in the top
                    mesh["ASDA_type"][i] = ASDAelem_type["K"]
            elif ele_center[1] > (ASDA_ywidth/2.):
                # it is in the front
                # check if it is in the bottom or top
                if ele_center[2] < -ASDA_zwidth:
                    # it is in the bottom
                    mesh["ASDA_type"][i] = ASDAelem_type["BF"]
                else:
                    # it is in the top
                    mesh["ASDA_type"][i] = ASDAelem_type["F"]
            else:
                # it is in the middle
                # check if it is in the bottom or top
                if ele_center[2] < -ASDA_zwidth:
                    # it is in the bottom
                    mesh["ASDA_type"][i] = ASDAelem_type["B"]
        
        i += 1
# %%
#  =============================================================================
# write the mesh
# =============================================================================
if not os.path.exists(Dir):
    os.makedirs(Dir)
else :
    # remove the files in the directory
    for file in os.listdir(Dir):
        os.remove(os.path.join(Dir,file))


min_core = mesh.cell_data['partitioned'].min()
max_core = mesh.cell_data['partitioned'].max()



# write the  mesh nodes
for core  in range(min_core,max_core+1):
    tmp  = mesh.extract_cells(np.where(mesh.cell_data['partitioned']==core)[0])
    f  = open(Dir + "/Nodes" + str(core) + ".tcl", "w")

    for i in range(tmp.n_points):
        f.write(f"node  {tmp['vtkOriginalPointIds'][i]} {tmp.points[i][0]} {tmp.points[i][1]} {tmp.points[i][2]}\n")
    f.close()
# %%

# writing the mesh elements
if AbsorbingElements == "ASDA":
    # writing the mesh elements
    for core in range(min_core,max_core+1):
        tmp  = mesh.extract_cells(np.where(mesh.cell_data['partitioned']==core)[0])
        f    = open(Dir + "/Elements" + str(core) + ".tcl", "w")
        if core >= reg_num_cores + DRM_num_cores:
            for eletag in range(tmp.n_cells):
                f.write(f"eval \"element $elementType {tmp['vtkOriginalCellIds'][eletag]} {' '.join(str(x) for x in tmp['vtkOriginalPointIds'][tmp.get_cell(eletag).point_ids])} $matTag{tmp['matTag'][eletag]} {ASDAelem_typereverse[tmp['ASDA_type'][eletag]]}\" \n")
        else:
            for eletag in range(tmp.n_cells):
                f.write(f"eval \"element $elementType {tmp['vtkOriginalCellIds'][eletag]} {' '.join(str(x) for x in tmp['vtkOriginalPointIds'][tmp.get_cell(eletag).point_ids])} $matTag{tmp['matTag'][eletag]}\" \n")
        f.close()
else :
    for core in range(min_core,max_core+1):
        tmp  = mesh.extract_cells(np.where(mesh.cell_data['partitioned']==core)[0])
        f    = open(Dir + "/Elements" + str(core) + ".tcl", "w")
        for eletag in range(tmp.n_cells):
            f.write(f"eval \"element $elementType {tmp['vtkOriginalCellIds'][eletag]} {' '.join(str(x) for x in tmp['vtkOriginalPointIds'][tmp.get_cell(eletag).point_ids])} $matTag{tmp['matTag'][eletag]}\" \n")
        f.close()

if AbsorbingElements == "PML":
    # writing the boundary files
    for core in range(reg_num_cores + DRM_num_cores , max_core+1):
        tmp = mesh.extract_cells(np.where(mesh.cell_data['partitioned']==core)[0])
        f = open(Dir + "/Boundary" + str(core) + ".tcl", "w")
        for i in range(tmp.n_points):
            if tmp["boundary"][i] != -1:
                x,y,z = tmp.points[i]
                nodeTag1 = tmp['vtkOriginalPointIds'][i]
                nodeTag2 = tmp['boundary'][i]
                f.write(f"node {nodeTag2} {str(x)} {str(y)} {str(z)}\n")
                f.write(f"equalDOF {nodeTag2} {nodeTag1} 1 2 3\n")











# =============================================================================
# printing information
# =============================================================================
print(f"Number of regular cores: {reg_num_cores}")
print(f"Number of DRM cores: {DRM_num_cores}")
print(f"Number of PML cores: {PML_num_cores}")
print(f"Number of regular elements: {regular.n_cells} roughly {int(regular.n_cells/reg_num_cores)} each core")
print(f"Number of DRM elements: {DRM.n_cells} roughly {int(DRM.n_cells/DRM_num_cores)} each core")
print(f"Number of PML elements: {PML.n_cells} roughly {int(PML.n_cells/PML_num_cores)} each core")
print(f"Number of total elements: {mesh.n_cells}")
print(f"Number of total points: {mesh.n_points}")
print(f"Number of cores: {max_core-min_core+1}")
print(f"Number of PML nodes: {PML.n_points}")
print(f"Number of regular nodes: {regular.n_points}")
print(f"Number of DRM nodes: {DRM.n_points}")
if AbsorbingElements == "PML":
    print(f"Number of MP constraints: {regindicies.size}")

# calculating number of surface points on the boundaries
eps = 1e-2
bounds = np.array(mesh.bounds)-np.array([-eps,eps,-eps,eps,-eps,-eps])
cube = pv.Cube(bounds=bounds)
# points outside the cube
selected  = mesh.select_enclosed_points(cube,inside_out = True)
pts =  mesh.extract_points(
    selected['SelectedPoints'].view(bool),
    include_cells=False,
)
print(f"number of sp constriants: {pts.n_points*9}")





import h5py 
# f = h5py.File('./DRMloadSmall.h5drm', 'r')
f = h5py.File(DRMfile, 'r')
pts = f["DRM_Data"]["xyz"][:]
internal = f["DRM_Data"]["internal"][:]
# %%



pl = pv.Plotter()
# plot the DRM layer with the DRM nodes loaded from the DRM file
pl.add_mesh(DRM,scalars="partitioned",show_edges=True)
pl.add_points(pts, color='r')
pl.export_html(meshplotdir+"/DRM.html")
pl.clear()


# plot the regular mesh with the internal nodes loaded from the DRM file
pl.add_mesh(regular,scalars="partitioned",show_edges=True)
pl.export_html(meshplotdir+"/Regular.html")
pl.clear()

# plot the PML mesh
pl.add_mesh(PML,scalars="partitioned",show_edges=True)
pl.export_html(meshplotdir+"/PML.html")
pl.clear()


# plot the total mesh
pl.add_mesh(mesh,scalars="partitioned",show_edges=True)
pl.export_html(meshplotdir+"/Total_partitioned.html")
pl.clear()

# plot the total mesh with domain scalars
pl.add_mesh(mesh,scalars="Domain",show_edges=True)
pl.export_html(meshplotdir+"/Total_domain.html")
pl.clear()


if AbsorbingElements == "ASDA":
    pl.add_mesh(mesh,scalars="ASDA_type",show_edges=True,cmap="tab20")
    pl.export_html(meshplotdir+"/ASDA_total.html")
    pl.clear()

    # filter the mesh with domain pml
    indices = mesh['Domain'] == info["PMLDomain"]
    grid = mesh.extract_cells(indices)
    pl.add_mesh(grid,scalars="ASDA_type",show_edges=True,cmap="tab20b")
    pl.export_html(meshplotdir+"/ASDA_PML.html")
    pl.clear()

pl.close()

# save the mesh
mesh.save(os.path.join(OutputDir,"mesh.vtk"),binary=True)
# mesh.plot(scalars="partitioned",show_edges=True,show_grid=True,show_axes=True,show_bounds=True)
# %%
# print number of PML nodes


# %%
# chek for each element that the nodes go counter clockwise and first the below surface nodes and then the above surface nodes
# def iscounterclockwise(points):
#     # points = np.array(points)
#     points = points - points.mean(axis=0)
#     signed_area = 0
#     for j in range(points.shape[0]):
#         p1 = points[j,:]
#         p2 = points[(j+1)%points.shape[0],:]
#         x1,y1,_ = p1
#         x2,y2,_ = p2
#         signed_area += (x1*y2 - x2*y1)
    
#     return signed_area/2.0

# for i in range(mesh.n_cells):
#     cell = mesh.get_cell(i)
#     points = mesh.points[cell.point_ids]
    
#      # first surface nodes
#     S1 = points[:4,:]
#     S2 = points[4:,:]
    
#     # check that the S2 is above S1
#     if S1[:,2].mean() > S2[:,2].mean():
#         print(f"Element {i} S1 is above S2")
#     # else :
#     #     print(f"Element {i} S2 is above S1")
        
#     # calcuting signed area of the S1
#     S1 = iscounterclockwise(S1)
#     S2 = iscounterclockwise(S2)
#     if S1 < 0:
#         print(f"Element {i} S1 is not counter clockwise")
#     if S2 < 0:
#         print(f"Element {i} S2 is not counter clockwise")



        


