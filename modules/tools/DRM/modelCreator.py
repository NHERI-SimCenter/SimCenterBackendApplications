# %%
import numpy as np
from MeshGenerator.MeshGenrator import *
from MeshGenerator.Infowriter import *
import os
import shutil
import argparse

# change the current directory to the directory of the script
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


parser = argparse.ArgumentParser(description='Create a model for the DRM')
parser.add_argument(
    '--soilwidth_x', type=float, help='width of the soil in x direction'
)
parser.add_argument(
    '--soilwidth_y', type=float, help='width of the soil in y direction'
)
parser.add_argument(
    '--soilwidth_z', type=float, help='width of the soil in z direction'
)
parser.add_argument(
    '--soilNumEle_x', type=int, help='number of elements in x direction'
)
parser.add_argument(
    '--soilNumEle_y', type=int, help='number of elements in y direction'
)
parser.add_argument(
    '--soilNumEle_z', type=int, help='number of elements in z direction'
)
parser.add_argument('--Vs', type=float, help='shear wave velocity')
parser.add_argument('--nu', type=float, help='poisson ratio')
parser.add_argument('--rho', type=float, help='density')
parser.add_argument('--DRM_filePath', type=str, help='DRM file path')
parser.add_argument('--DRM_numLayers', type=int, help='number of DRM layers')
parser.add_argument('--DRM_loadFactor', type=float, help='DRM load factor')
parser.add_argument('--DRM_crdScale', type=float, help='DRM coordinate scale')
parser.add_argument('--DRM_tolerance', type=float, help='DRM tolerance')
parser.add_argument('--DRM_T00', type=float, help='DRM T00')
parser.add_argument('--DRM_T01', type=float, help='DRM T01')
parser.add_argument('--DRM_T02', type=float, help='DRM T02')
parser.add_argument('--DRM_T10', type=float, help='DRM T10')
parser.add_argument('--DRM_T11', type=float, help='DRM T11')
parser.add_argument('--DRM_T12', type=float, help='DRM T12')
parser.add_argument('--DRM_T20', type=float, help='DRM T20')
parser.add_argument('--DRM_T21', type=float, help='DRM T21')
parser.add_argument('--DRM_T22', type=float, help='DRM T22')
parser.add_argument('--DRM_originX', type=float, help='DRM origin X')
parser.add_argument('--DRM_originY', type=float, help='DRM origin Y')
parser.add_argument('--DRM_originZ', type=float, help='DRM origin Z')
parser.add_argument('--DRM_Software', type=str, help='DRM software')
parser.add_argument(
    '--DRM_CoordinateTransformation', type=bool, help='DRM coordinate transformation'
)
parser.add_argument(
    '--Absorb_HaveAbsorbingElements', type=str, help='Absorbing elements'
)
parser.add_argument(
    '--Absorb_NumAbsorbingElements', type=int, help='Number of absorbing elements'
)
parser.add_argument('--Absorb_rayleighAlpha', type=float, help='Rayleigh alpha')
parser.add_argument('--Absorb_rayleighBeta', type=float, help='Rayleigh beta')
parser.add_argument('--Absorb_type', type=str, help='Absorbing type')
parser.add_argument('--dt', type=float, help='time step')
parser.add_argument('--t_final', type=float, help='final time')
parser.add_argument('--recording_dt', type=float, help='recording time step')
parser.add_argument('--outputDir', type=str, help='output directory')
parser.add_argument('--PartitionAlgorithm', type=str, help='partition algorithm')
parser.add_argument('--soilcores', type=int, help='number of cores for the soil')
parser.add_argument('--absorbingcores', type=int, help='number of cores for the PML')
parser.add_argument('--drmcores', type=int, help='number of cores for the DRM')
parser.add_argument('--DRM_Location', type=str, help='DRM location')

args = parser.parse_args()

# ============================================================================
# Cores information
# ============================================================================
regcores = args.soilcores
pmlcores = args.absorbingcores
drmcores = args.drmcores
structurecores = 0
AnalysisType = 'PML'  # options: "DRMFIXED", PMLDRM", "ASDADRM"
Target = 'Soil-with-structure'  # options: "Soil", "Structure", "Soil-with-structure"
PartitionAlgorithm = args.PartitionAlgorithm  # options: "kd-tree", "metis"
# ============================================================================
# Structure information
# ============================================================================
StructureType = 'STEEL'  # Options: STEEL, CONCRETE, Custom
NStory = 10  # number of stories above ground level
NBay = 4  # number of bays in X direction
NBayZ = 4  # number of bays in Y direction
StartNodeX = -9.0  # X coordinate of the first node
StartNodeY = -9.0  # Y coordinate of the first node
StartNodeZ = -1.5  # Z coordinate of the first node
meter = 1.0  # meter to specified unit conversion (SI unit)
ft = 0.3048  # feet to meter conversion (SI unit)
LCol = 3 * meter  # column height (parallel to Z axis)
LBeam = 4.5 * meter  # beam length (parallel to X axis)
LGird = 4.5 * meter  # girder length (parallel to Y axis)
SectionType = 'Elastic'  # options: Elastic, FiberSection
HaveStructure = 'NO'  # options: "YES", "NO"

# ============================================================================
# Soil information
# ============================================================================
dx = args.soilwidth_x / args.soilNumEle_x
dy = args.soilwidth_y / args.soilNumEle_y
dz = args.soilwidth_z / args.soilNumEle_z
lx = args.soilwidth_x
ly = args.soilwidth_y
lz = args.soilwidth_z
drmthicknessx = dx
drmthicknessy = dy
drmthicknessz = dz
numdrmlayers = args.DRM_numLayers
llx = lx + 2 * numdrmlayers * drmthicknessx
lly = ly + 2 * numdrmlayers * drmthicknessy
llz = lz + 1 * numdrmlayers * drmthicknessz

nx = args.soilNumEle_x
ny = args.soilNumEle_y
nz = args.soilNumEle_z

# modelCreator.py: error: unrecognized arguments:  --soilwidth_x  100  --soilwidth_y  100  --soilwidth_z  30  --soilNumEle_x  40  --soilNumEle_y  40  --soilNumEle_z  12  --Vs  200  --nu  0.3  --rho  2000  --DRM_filePath  /mnt/d/Projects/OpenSeesProjects/EEUQ_DRMBackend/DRMload/DRMLoad.h5drm  --DRM_numLayers  3  --DRM_laodFactor  1  --DRM_crdScale  1  --DRM_tolernace  1  --DRM_T00  0.0  --DRM_T01  1.0  --DRM_T02  0.0  --DRM_T10  1.0  --DRM_T11  0.0  --DRM_T12  0.0  --DRM_T20  0.0  --DRM_T21  0.0  --DRM_T22  -1.0  --DRM_originX  0  --DRM_originY  0  --DRM_originZ  0  --DRM_Sofrware  ShakerMaker  --Absorb_HaveAbsorbingElements NO  --Absorb_NumAbsorbingElements  2  --Absorb_type PML  --Absorb_rayleighAlpha 0.0  --Absorb_rayleighBeta 0.0  --dt  0.005  --t_final  60  --recording_dt  0.005  --outputDir  /home/amnp95/Documents/EE-UQ/LocalWorkDir/DRM_Model
# ============================================================================
# PML information
# ============================================================================
AbsorbingElements = args.Absorb_type  # could be "ASDA" or "NORMAL"
numpmllayers = args.Absorb_NumAbsorbingElements
pmlthicknessx = dx
pmlthicknessy = dy
pmlthicknessz = dz
pmltotalthickness = numpmllayers * pmlthicknessx
HaveAbsorbingElements = args.Absorb_HaveAbsorbingElements  # options: "YES", "NO"
Absorbing_rayleigh_alpha = args.Absorb_rayleighAlpha
Absorbing_rayleigh_beta = args.Absorb_rayleighBeta


# ============================================================================
# General information
# ============================================================================
tmpdir = args.outputDir
meshdir = f'{tmpdir}/Mesh'
outputdir = f'{tmpdir}/Results'
DRMFile = args.DRM_filePath

# ============================================================================
# Embedding foundation
# ============================================================================
EmbeddingFoundation = 'NO'  # options: "YES", "NO"
EmbeddedFoundation = {
    'xmax': 10.0,
    'xmin': -10.0,
    'ymax': 10.0,
    'ymin': -10.0,
    'zmax': 0.0,
    'zmin': -5.0,
}

# ============================================================================
# Fondation information
# ============================================================================
HaveFoundation = 'NO'
AttachFoundation = 'NO'
foundationBlocks = []

# adding a block
foundationBlocks.append(
    {
        'matTag': 1,
        'xmax': 10.0,
        'xmin': -10.0,
        'ymax': 10.0,
        'ymin': -10.0,
        'zmax': -1.5,
        'zmin': -4.5,
        'Xmeshsize': 1.0,
        'Ymeshsize': 1.0,
        'Zmeshsize': 1.0,
    }
)


# ============================================================================
# piles information
# ============================================================================
pilelist = []

x = np.arange(-7.0, 7.0 + 1e-6, 3.5)
y = np.arange(-7.0, 7.0 + 1e-6, 3.5)

x, y = np.meshgrid(x, y)
x = x.flatten()
y = y.flatten()

HavePiles = 'NO'
for i in range(len(x)):
    pilelist.append(
        {
            'xtop': x[i],
            'ytop': y[i],
            'ztop': -3.0,
            'xbottom': x[i],
            'ybottom': y[i],
            'zbottom': -10.0,
            'numberofElements': 6,
        }
    )

# ============================================================================
# DRM information
# ============================================================================
# DRM information
DRMinformation = {
    'DRM_Provider_Software': args.DRM_Software,
    'factor': args.DRM_loadFactor,
    'crd_scale': args.DRM_crdScale,
    'distance_tolerance': args.DRM_tolerance,
    'do_coordinate_transformation': 1,
    'T00': args.DRM_T00,
    'T01': args.DRM_T01,
    'T02': args.DRM_T02,
    'T10': args.DRM_T10,
    'T11': args.DRM_T11,
    'T12': args.DRM_T12,
    'T20': args.DRM_T20,
    'T21': args.DRM_T21,
    'T22': args.DRM_T22,
    'originX': args.DRM_originX,
    'originY': args.DRM_originY,
    'originZ': args.DRM_originZ,
    'DRM_Location': args.DRM_Location,
}


# ============================================================================
# analysis information
# ============================================================================
dt = args.dt
t_final = args.t_final
recording_dt = args.recording_dt

analysisinfo = {
    'dt': dt,
    't_final': t_final,
    'recording_dt': recording_dt,
}


# ============================================================================
# material information
# ============================================================================
# creating the material information
materialInfo = {
    'Vs': args.Vs,
    'nu': args.nu,
    'rho': args.rho,
}

# ============================================================================
# Create directories
# ============================================================================
# delete the directories if exists
if os.path.exists(meshdir):
    # removing directory
    shutil.rmtree(meshdir, ignore_errors=False)

if os.path.exists(outputdir):
    shutil.rmtree(outputdir, ignore_errors=False)

# creating directories using shutil
os.makedirs(meshdir)
os.makedirs(outputdir)


# ============================================================================
# Creating the mesh
# ============================================================================
info = {
    'xwidth': lx,
    'ywidth': ly,
    'zwidth': lz,
    'Xmeshsize': dx,
    'Ymeshsize': dy,
    'Zmeshsize': dz,
    'PMLThickness': [pmlthicknessx, pmlthicknessy, pmlthicknessz],
    'numPMLLayers': numpmllayers,
    'DRMThickness': [drmthicknessx, drmthicknessy, drmthicknessz],
    'numDrmLayers': numdrmlayers,
    'reg_num_cores': regcores,
    'DRM_num_cores': drmcores,
    'PML_num_cores': pmlcores,
    'Structure_num_cores': structurecores,
    'Dir': meshdir,
    'OutputDir': outputdir,
    'AbsorbingElements': AbsorbingElements,
    'DRMfile': DRMFile,
    'EmbeddingFoundation': EmbeddingFoundation,
    'EmbeddedFoundationDict': EmbeddedFoundation,
    'HaveFoundation': HaveFoundation,
    'foundationBlocks': foundationBlocks,
    'pilelist': pilelist,
    'HavePiles': HavePiles,
    'HaveStructure': HaveStructure,
    'AttachFoundation': AttachFoundation,
    'DRMinformation': DRMinformation,
    'AnalysisInfo': analysisinfo,
    'PartitionAlgorithm': PartitionAlgorithm,
    'Absorbing_rayleigh_alpha': Absorbing_rayleigh_alpha,
    'Absorbing_rayleigh_beta': Absorbing_rayleigh_beta,
}

numcells, numpoints = DRM_PML_Foundation_Meshgenrator(info)
print(f'Number of cells: {numcells}')
print(f'Number of points: {numpoints}')
# ============================================================================
# Writing the information file
# ============================================================================
info = {
    'soilfoundation_num_cells': numcells,
    'soilfoundation_num_points': numpoints,
    'AnalysisType': AnalysisType,
    'regcores': regcores,
    'pmlcores': pmlcores,
    'drmcores': drmcores,
    'structurecores': structurecores,
    'StructureType': StructureType,
    'NStory': NStory,
    'NBay': NBay,
    'NBayZ': NBayZ,
    'StartNodeX': StartNodeX,
    'StartNodeY': StartNodeY,
    'StartNodeZ': StartNodeZ,
    'meter': meter,
    'ft': ft,
    'LCol': LCol,
    'LBeam': LBeam,
    'LGird': LGird,
    'SectionType': SectionType,
    'dx': dx,
    'dy': dy,
    'dz': dz,
    'llx': llx,
    'lly': lly,
    'llz': llz,
    'drmthicknessx': drmthicknessx,
    'drmthicknessy': drmthicknessy,
    'drmthicknessz': drmthicknessz,
    'numdrmlayers': numdrmlayers,
    'lx': lx,
    'ly': ly,
    'lz': lz,
    'nx': nx,
    'ny': ny,
    'nz': nz,
    'AbsorbingElements': AbsorbingElements,
    'numpmllayers': numpmllayers,
    'pmlthicknessx': pmlthicknessx,
    'pmlthicknessy': pmlthicknessy,
    'pmlthicknessz': pmlthicknessz,
    'pmltotalthickness': pmltotalthickness,
    'HaveAbsorbingElements': HaveAbsorbingElements,
    'meshdir': meshdir,
    'outputdir': outputdir,
    'DRMFile': args.DRM_filePath.split('/')[-1],
    'EmbeddingFoundation': EmbeddingFoundation,
    'EmbeddedFoundation': EmbeddedFoundation,
    'HaveFoundation': HaveFoundation,
    'foundationBlocks': foundationBlocks,
    'pilelist': pilelist,
    'HavePiles': HavePiles,
    'HaveStructure': HaveStructure,
    'DRMinformation': DRMinformation,
    'Absorbing_rayleigh_alpha': Absorbing_rayleigh_alpha,
    'Absorbing_rayleigh_beta': Absorbing_rayleigh_beta,
    'AnalysisInfo': analysisinfo,
    'MaterialInfo': materialInfo,
}
infowriter(info, meshdir)


# ============================================================================
# copy the related file as model.tcl to the current directory
# ============================================================================
def copy_file(source_path, destination_path):
    with open(destination_path, 'wb') as dst_file:
        with open(f'{meshdir}/Modelinfo.tcl', 'rb') as src_file:
            dst_file.write(src_file.read())
        with open(source_path, 'rb') as src_file:
            dst_file.write(src_file.read())


# delete the model file if exists
if os.path.exists('./model.tcl'):
    os.remove('./model.tcl')

if Target == 'Soil-with-structure':
    copy_file(
        f'MeshGenerator/models/Soil_with_structure.tcl', f'{meshdir}/model.tcl'
    )

# %%
