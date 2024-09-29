# %%
#############################################################
# This is shaker Maker model created for runing simulations #
# for the EE-UQ app.                                        #
# This code created by Amin Pakzad and Pedro Arduino based  #
# on the initial code by Jose Abell and Jorge Crempien      #
# date = September 2024                                     #
# ########################################################### 

import os
import json
from geopy.distance import geodesic
from shakermaker import shakermaker
from shakermaker.crustmodel import CrustModel
from shakermaker.pointsource import PointSource 
from shakermaker.faultsource import FaultSource
from shakermaker.slw_extensions import DRMHDF5StationListWriter
from shakermaker.sl_extensions import DRMBox
from shakermaker.station import Station
from shakermaker.stationlist import StationList
from mpi4py import MPI

def calculate_distances_with_direction(lat1, lon1, lat2, lon2):
    # Original points
    point1 = (lat1, lon1)
    
    # Calculate north-south distance (latitude changes, longitude constant)
    north_point = (lat2, lon1)
    north_south_distance = geodesic(point1, north_point).kilometers
    north_south_direction = "north" if lat2 > lat1 else "south"
    
    # Calculate east-west distance (longitude changes, latitude constant)
    west_point = (lat1, lon2)
    west_east_distance = geodesic(point1, west_point).kilometers
    west_east_direction = "east" if lon2 > lon1 else "west"
    
    # south is negative
    north_south_distance = -north_south_distance if north_south_direction == "south" else north_south_distance
    # west is negative
    west_east_distance = -west_east_distance if west_east_direction == "west" else west_east_distance

    return north_south_distance, west_east_distance

# ======================================================================================
# Code initialization
# ======================================================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

# reading the metdadatafile
metadata_file = "metadata.json"
with open(metadata_file) as f:
    metadata = json.load(f)

# make results directory if it doesn't exist
results_dir = "results"
if rank == 0:
    if not os.path.exists("results"):
        os.makedirs("results")
comm.barrier()
# Define the source parameters
# For estimating wave arrival windows, 
# assume the following maximum and 
# minimum propagation velocities
Vs_min = 3.14  # Minimum shear wave velocity
Vp_max = 8.00  # Maximum primary wave velocity
MINSLIP = 0.0  # Minimum slip for the fault

# ======================================================================================
# Shakermaker configuration
# ======================================================================================

_m = 0.001        # meters (ShakerMaker uses kilometers)
dt   = metadata["analysisdata"]["dt"]        # Time step
nfft = metadata["analysisdata"]["nfft"]      # Number of samples in the record
dk   = metadata["analysisdata"]["dk"]         # (Wavelength space discretization) adjust using theory
tb   = 0         # How much to "advance" the simulation window... no advance
tmin = metadata["analysisdata"]["tmin"]       # Time when the final results start
tmax = metadata["analysisdata"]["tmax"]       # Time when the final results end
delta_h     = metadata["analysisdata"]["dh"]*_m   # Horizontal distance increment
delta_v_rec = metadata["analysisdata"]["delta_v_rec"]*_m   # Vertical distance increment for receivers
delta_v_src = metadata["analysisdata"]["delta_v_src"]*_m   # Vertical distance increment for sources

# nfft = 2048       # Number of samples in the record
# dk = 0.2          # (Wavelength space discretization) adjust using theory
# tb = 0            # How much to "advance" the simulation window... no advance
# tmin = 0.         # Time when the final results start
# tmax = 45.        # Time when the final results end
# delta_h = 40*_m   # Horizontal distance increment
# delta_v_rec = 5.0*_m   # Vertical distance increment for receivers
# delta_v_src = 200*_m   # Vertical distance increment for sources


# options for the simulation
npairs_max = 200000
allow_out_of_bounds = False


# ======================================================================================
# Loading the crust model
# ======================================================================================
layers = metadata["crustdata"]
num_layers = len(layers)
CRUST = CrustModel(num_layers)
for layer in layers:
    name = layer['name']
    vp = layer['vp']
    vs = layer['vs']
    rho = layer['rho']
    thickness = layer['thick']
    Qa = layer['Qa']
    Qb = layer['Qb']
    CRUST.add_layer(thickness, vp, vs, rho, Qa, Qb)

# ======================================================================================
# Loading the fault
# ======================================================================================
# faultpath  = metadata["faultdata_path"]

# make Fault Directory
fault_dir = "fault"

# if rank == 0:
#     # remove the directory if it exists
#     if os.path.exists(fault_dir):
#         if os.name == 'nt':
#             os.system(f'rmdir /S /Q "{fault_dir}"')
#         else:
#             os.system(f'rm -rf "{fault_dir}"')

#     if not os.path.exists(fault_dir):
#         os.makedirs(fault_dir)

comm.barrier()

# copy the fault files to the fault directory
# src = faultpath 
# dst = fault_dir
# # windows, linux or mac
# if os.name == 'nt':  # Windows
#     cmd = 'copy'
# if os.name == 'posix':  # Unix/Linux
#     cmd = 'cp'
# if os.name == 'mac':  # Mac
#     cmd = 'cp'

# if rank == 0:
#     # copy the SourceTimeFunction.py file to the fault directory
#     os.system(f'{cmd} "{src}/SourceTimeFunction.py" "{dst}"')

#     # copy faultInfo.json to the fault directory
#     os.system(f'{cmd} "{src}/faultInfo.json" "{dst}"')

# wait for all processes to finish
comm.barrier()

# load the faultInfo.json file into faultdata
with open(f"faultInfo.json") as f:
    faultdata = json.load(f)


faultLat   = faultdata["latitude"]
faultLon   = faultdata["longitude"]
filenames  = faultdata["Faultfilenames"]
M0         = faultdata["M0"]
faultName  = faultdata["name"]
xmean      = faultdata["xmean"]
ymean      = faultdata["ymean"]

# if rank == 0:
#     for filename in filenames:
#         os.system(f'{cmd} "{src}/{filename}" "{dst}"')

#     # check that SourceTimeFunction is in the fault file
#     files = os.listdir(fault_dir)
#     if "SourceTimeFunction.py" not in files:
#         raise ValueError("SourceTimeFunction.py not found in the fault file")


# wait for all processes to finish
comm.barrier()

# import SourceTimeFunction from fault file
from SourceTimeFunction import source_time_function


for filename in filenames:
    sources = []

    # read the json fault file
    f = open(f"{filename}")
    faultsources = json.load(f)

    for source in faultsources:
        xsource   = source["x"] - xmean
        ysource   = source["y"] - ymean
        zsource   = source["z"]
        strike    = source["strike"]
        dip       = source["dip"]
        rake      = source["rake"]
        t0        = source["t0"]
        stf       = source["stf"]
        stf_type  = stf["type"]
        params    = stf["parameters"]
        numparams = stf["numParameters"]
        stf_func  = source_time_function(*params)
        sources.append(PointSource([xsource, ysource, zsource], [strike,dip,rake], tt=t0, stf=stf_func))
    f.close()
FAULT = FaultSource(sources, 
                    metadata = {
                        "name": f"{faultName} M0={M0}"
                    })
# ======================================================================================
# Loading the stations
# ======================================================================================
stationsType = metadata["stationdata"]["stationType"]
# single station
if stationsType.lower() in ["singlestation" ,"single"]:
    stationslist = []
    for station in  metadata["stationdata"]["Singlestations"]:
        stationLat    = station["latitude"]
        stationLon    = station["longitude"]
        stationDepth  = station["depth"]
        meta          = station["metadata"]
        xstation, ystation = calculate_distances_with_direction(faultLat, faultLon, stationLat, stationLon)
        stationslist.append(Station([xstation, ystation, stationDepth], metadata=meta))
    
    meta = {"name": metadata["stationdata"]["name"]}
    STATIONS = StationList(stationslist,metadata=meta)

elif stationsType.lower() in ["drmbox", "drm", "drm box", "drm_box","drm station"]:
    DRMdata = metadata["stationdata"]["DRMbox"]
    name = DRMdata["name"]
    latitude = DRMdata["latitude"]
    longitude = DRMdata["longitude"]
    depth = DRMdata["Depth"]
    Lx = DRMdata["Width X"] 
    Ly = DRMdata["Width Y"] 
    Lz = DRMdata["Depth"] 
    dx = DRMdata["Mesh Size X"] 
    dy = DRMdata["Mesh Size Y"] 
    dz = DRMdata["Mesh Size Z"] 
    nx, ny, nz = int(Lx/dx), int(Ly/dy), int(Lz/dz)
    dx = dx * _m; dy = dy * _m; dz = dz * _m
    Lx = Lx * _m; Ly = Ly * _m; Lz = Lz * _m
    xstation, ystation = calculate_distances_with_direction(faultLat, faultLon, latitude, longitude)
    STATIONS = DRMBox([xstation, ystation, 0], [nx, ny, nz], [dx, dy, dz], metadata={"name": name})
else:
    raise ValueError(f"Unknown station type: {stationsType}")


# ======================================================================================
# Create the shakermaker model
# ======================================================================================
model = shakermaker.ShakerMaker(CRUST, FAULT, STATIONS)

if stationsType.lower() in ["drmbox", "drm", "drm box", "drm_box","drm station"]:
    # creating the pairs
    model.gen_greens_function_database_pairs(
        dt=dt,             # Output time-step
        nfft=nfft,         # N timesteps
        dk=dk,             # wavenumber discretization
        tb=tb,             # Initial zero-padding
        tmin=tmin,
        tmax=tmax,
        smth=1,
        sigma=2,
        verbose=True,
        debugMPI=False,
        showProgress=True,
        store_here="results/greensfunctions_database",
        delta_h=delta_h,
        delta_v_rec=delta_v_rec,
        delta_v_src=delta_v_src,
        npairs_max=npairs_max,
        using_vectorize_manner=True,
        cfactor=0.5
    )

    # wait for all processes to finish
    comm.barrier()
    model.run_create_greens_function_database(
        h5_database_name="results/greensfunctions_database",
        dt=dt,             # Output time-step
        nfft=nfft,         # N timesteps
        dk=dk,             # wavenumber discretization
        tb=tb,             # Initial zero-padding
        tmin=tmin,
        tmax=tmax,
        smth=1,
        sigma=2,
        verbose=False,
        debugMPI=False,
        showProgress=True,
    )

    # wait for all processes to finish
    comm.barrier()
    writer = DRMHDF5StationListWriter("results/DRMLoad.h5drm")
    model.run_faster(
        h5_database_name="results/greensfunctions_database",
        dt=dt,           # Output time-step
        nfft=nfft,       # N timesteps
        dk=dk,           # wavenumber discretization
        tb=tb,           # Initial zero-padding
        tmin=tmin,
        tmax=tmax,
        smth=1,
        sigma=2,
        verbose=False,
        debugMPI=False,
        showProgress=True,
        writer=writer,
        delta_h=delta_h,
        delta_v_rec=delta_v_rec,
        delta_v_src=delta_v_src,
        allow_out_of_bounds=allow_out_of_bounds,
    )


# single station
if stationsType.lower() in ["singlestation" ,"single"]:
    model.run(
        dt=dt,           # Output time-step
        nfft=nfft,       # N timesteps
        dk=dk,           # wavenumber discretization
        tb=tb,           # Initial zero-padding
        tmin=tmin,
        tmax=tmax,
        smth=1,
        sigma=2,
        verbose=False,
        debugMPI=False,
        showProgress=True,
    )
    if rank == 0:
        for i, s in enumerate(stationslist):
            output_filename = f"results/station{i+1}.npz"
            s.save(output_filename)










