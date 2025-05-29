import json  # noqa: INP001, D100
import os

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import taichi.math as tm
from scipy.interpolate import griddata


# ti.init(arch=ti.cuda)
def checjson(variable, data):  # noqa: D103
    """
    Checks if a key exists in a JSON config file (i.e., CelerisWebGPU configuration file).

    Args:
        key (str): The key to check in the JSON config file.
        config (dict): The JSON-loaded dictionary.

    Returns:
        int: 1 if the key is found in the JSON dictionary, 0 otherwise.
    """
    R = 0  # noqa: N806
    if variable in data:
        R = 1  # noqa: N806
    return R


def ti2np(ti_type):  # noqa: D103
    """
    Converts a Taichi precision type to a NumPy dtype.

    Args:
        precision (ti.types.primitive_types): Taichi precision type (e.g., ti.f32, ti.f64).

    Returns:
        numpy.dtype: Corresponding NumPy dtype (e.g., np.float32, np.float64).
    """
    np_type = np.float32
    if ti_type == ti.f32:
        np_type = np.float32
    if ti_type == ti.f16:
        np_type = np.float16
    return np_type


class Topodata:  # noqa: D101
    """
    Manages all topography/bathymetry data formats for CelerisAi.

    This class handles different input formats for bathymetric or topographic data,
    such as 2D (XYZ), 1D (XZ), and Celeris-native formats. It reads the corresponding
    files (or folders) and loads them into a NumPy array for further processing or
    analysis.

    Attributes:
        filename (str, optional): Name of the file containing topographic/bathymetric 
            data. If `path` is provided, the file is assumed to be located there.
        datatype (str, optional): Format of the data. Accepted values are:
            - "xyz": 2D data in three columns (x, y, z)
            - "xz": 1D data in two columns (x, z)
            - "celeris": 2D data located in a folder with a file named "bathy.txt"
        path (str, optional): Directory path where the file is located. If not provided, 
            `filename` should be a complete path or in the current working directory.
    Example:
        >>> Bfrom celeris.domain import Topodata
        >>> baty = Topodata(datatype='celeris',path='./examples/DuckFRF_NC')
    """
    def __init__(self, filename=None, datatype=None, path=None):
        """
        Initializes the Topodata object with the necessary parameters.

        Args:
            filename (str, optional): Name of the file containing the data. If `path`
                is also specified, the file is read from that path. Defaults to None.
            datatype (str, optional): Specifies the data format. Accepted values are:
                "xyz", "xz", or "celeris". Defaults to None.
            path (str, optional): Directory path where the data file is located. 
                Defaults to None.
        """
        self.filename = filename
        self.datatype = datatype
        self.path = path

    def z(self, seaLevel=0.0): # noqa: D102
        """
        Loads the bathymetry/topography data based on the specified datatype.

        Depending on the `datatype`, this method reads the corresponding file(s) and 
        returns the data as a NumPy array.

        - For "xyz": Expects a file with three columns (x, y, z).
        - For "xz": Expects a file with two columns (x, z).
        - For "xyz" and "xz", bathymetry must be positive.
        - For "celeris": Expects a folder containing a file named "bathy.txt". 
          The returned array values are multiplied by -1.

        Returns:
            numpy.ndarray or str: 
                - A NumPy array if `datatype` is recognized and the file is successfully 
                  read. 
                - The string "No supported format" if `datatype` is not recognized.

        Raises:
            OSError: If the file (or directory for "celeris") cannot be found or read.
        """
        if self.datatype == 'xyz':
            if self.path != None:  # noqa: E711
                if self.filename == None:  # noqa: E711
                    self.filename = 'test_curve.xyz'
                return np.loadtxt(os.path.join(self.path, self.filename))  # noqa: PTH118
            return np.loadtxt(self.filename)
        if self.datatype == 'xz':
            if self.path!=None:
                return np.loadtxt(os.path.join(self.path, self.filename))
            else:
                return np.loadtxt(self.filename)
        if self.datatype == 'celeris' or self.datatype == 'txt':  # noqa: PLR1714
            if self.path != None:  # noqa: E711
                if self.filename == None:  # noqa: E711
                    self.filename = 'bathy.txt'
                bathy = np.loadtxt(os.path.join(self.path, self.filename))  # noqa: PTH118
                bathy = bathy - seaLevel
                return bathy * -1
        return 'No supported format'


class BoundaryConditions:
    """
    Manages boundary conditions for a rectangular domain in the CelerisAi model.

    The domain has four faces: north, south, east, and west. Each face can be configured
    with different boundary types (e.g., sponge layer, solid wall, incoming wave).
    Boundary conditions can be set in one of two ways:

    1) **Celeris format** (`celeris=True`): Reads from a `config.json` file.
    2) **Manual** (`celeris=False`): Uses manually supplied values.

    If incoming waves (boundary type = 2) are defined, the wave type can be specified via
    `WaveType`. If `WaveType` is -1, the wave parameters are read from a file (e.g., 
    `"waves.txt"`). Otherwise, a sine wave is assumed and set by `sine_wave`.

    This class also includes utility methods for handling wave data input and conversion
    from NumPy arrays to Taichi tensors.
    
    Attributes:

        precision (taichi.types.primitive_types): The precision used by Taichi (e.g. `ti.f32`, `ti.f64`).
        North (int): Boundary type for the north face. Valid types are:

            - 0: Solid wall
            - 1: Sponge layer
            - 2: Incoming wave

        South (int): Boundary type for the south face.
        East (int): Boundary type for the east face.
        West (int): Boundary type for the west face.
        WaveType (int): Wave type indicator:

            - -1: Wave parameters read from a file (`waves.txt`)
            -  Any other integer: Use `sine_wave` array for wave parameters
        
        Amplitude (float): Wave amplitude, used if not reading from a file.
        path (str): Path to the directory containing `waves.txt` and/or `config.json`.
        filename (str): Name of the file that stores wave parameters (e.g. `waves.txt`).
        BoundaryWidth (int): Width of the sponge or boundary zone.
        sine_wave (list of float): Parameters defining a sine wave if `WaveType` is not -1.
        celeris (bool): If True, boundary conditions are read from `config.json`.
        configfile (dict, optional): Loaded JSON configuration when `celeris=True`.
        data (numpy.ndarray): Placeholder array for wave data (size depends on `N_data`).
        N_data (int): Number of wave entries to read from file.
        W_data (None): Unused placeholder (could store wave data in some contexts).

    Example:
        >>> from celeris.domain import BoundaryConditions
        >>> bc = BoundaryConditions(celeris=True,path='./examples/DuckFRF_NC',precision=precision)
    """
    def __init__(  # noqa: C901
        self,
        celeris=True,  # noqa: FBT002
        precision=ti.f32,
        North=10,  # noqa: N803
        South=10,  # noqa: N803
        East=10,  # noqa: N803
        West=10,  # noqa: N803
        WaveType=-1,  # noqa: N803
        Amplitude=0.5,  # noqa: N803
        Period=10.0,  # noqa: N803
        path='./scratch',
        filename='waves.txt',
        configfile='config.json',
        BoundaryWidth=20,  # noqa: N803
        init_eta=5,
        sine_wave=None,  # noqa: B006
    ):
        """
        Initializes the BoundaryConditions object with specified or default parameters.

        Args:
            celeris (bool, optional): If True, reads boundary settings from `config.json` 
                located at `path`. Defaults to True.
            precision (taichi.types.primitive_types, optional): Taichi precision type 
                (e.g., `ti.f32`, `ti.f64`). Defaults to `ti.f32`.
            North (int, optional): Boundary type for the north face. Defaults to 10 (unrecognized).
            South (int, optional): Boundary type for the south face. Defaults to 10 (unrecognized).
            East (int, optional): Boundary type for the east face. Defaults to 10 (unrecognized).
            West (int, optional): Boundary type for the west face. Defaults to 10 (unrecognized).
            WaveType (int, optional): Indicator for wave input. Defaults to -1 (read from file).
            Amplitude (float, optional): Amplitude of the wave if not read from a file. Defaults to 0.5.
            path (str, optional): Directory path containing `config.json` and/or `waves.txt`. 
                Defaults to './scratch'.
            filename (str, optional): Wave file name (e.g. `waves.txt`). Defaults to 'waves.txt'.
            BoundaryWidth (int, optional): Width of the sponge or boundary zone. Defaults to 20.
            sine_wave (list of float, optional): Parameters defining a sine wave if `WaveType` is not -1.
                If None, defaults to `[0, 0, 0, 0]`.
        """
        if sine_wave is None:
            sine_wave = [0, 0, 0, 0]
            
        self.precision = precision
        self.North = North
        self.South = South
        self.East = East
        self.West = West
        self.WaveType = WaveType  # -1 to read from a file
        self.sine_wave = sine_wave
        self.data = np.zeros((2, 4))
        self.N_data = 1
        self.W_data = None
        self.init_eta = init_eta
        self.amplitude = Amplitude
        self.period = Period
        self.path = path
        self.configfile = None
        self.celeris = celeris
        if self.celeris == True:  # noqa: E712
            with open(os.path.join(self.path, configfile)) as uf:  # noqa: PTH118, PTH123
                self.configfile = json.load(uf)
            if checjson('WaveType', self.configfile) == 1:
                self.WaveType = int(self.configfile['WaveType'])
            if checjson('amplitude', self.configfile) == 1:
                self.amplitude = float(self.configfile['amplitude'])
            if checjson('period', self.configfile) == 1:
                self.period = float(self.configfile['period'])
            if checjson('sine_wave', self.configfile) == 1:
                self.sine_wave = [float(self.configfile['sine_wave'][0]),
                                 float(self.configfile['sine_wave'][1]),
                                 float(self.configfile['sine_wave'][2]),
                                 float(self.configfile['sine_wave'][3])]
                self.N_data = 1 # Just one wave
            if checjson('BoundaryWidth', self.configfile) == 1:
                self.BoundaryWidth = int(self.configfile['BoundaryWidth'])
            else:
                self.BoundaryWidth = BoundaryWidth
            if checjson('north_boundary_type', self.configfile) == 1:
                self.North = int(self.configfile['north_boundary_type'])
            else:
                self.North = North
            if checjson('south_boundary_type', self.configfile) == 1:
                self.South = int(self.configfile['south_boundary_type'])
            else:
                self.South = South
            if checjson('east_boundary_type', self.configfile) == 1:
                self.East = int(self.configfile['east_boundary_type'])
            else:
                self.East = East
            if checjson('west_boundary_type', self.configfile) == 1:
                self.West = int(self.configfile['west_boundary_type'])
            else:
                self.West = West
        else:
            self.BoundaryWidth = BoundaryWidth
            self.North = North
            self.South = South
            self.West = West
            self.East = East

        if self.WaveType == -1:
            self.filename = os.path.join(path, filename)  # noqa: PTH118

    def Sponge(self, width=None):  # noqa: N802, D102
        """
        Returns the sponge (boundary) width to be used in the model.

        Args:
            width (int, optional): Overrides the class-level sponge width. Defaults to None.

        Returns:
            int: The sponge width (either the class attribute or the passed-in argument).
        """
        return width

    def SineWave(self):  # noqa: N802, D102
        """
        Provides the sine wave parameters as a NumPy array of the specified precision.

        Returns:
            numpy.ndarray: An array of length 4 containing the sine wave parameters.
        """
        return np.array(self.sine_wave, dtype=self.precision)

    def tseries(self):  # noqa: D102
        """
        Checks if a wave parameter file is specified.

        Returns:
            bool: True if `self.filename` is not None, indicating that time series 
            wave data can be read from a file. Otherwise, False.
        """
        R = False  # noqa: N806
        if self.filename != None:  # noqa: E711
            R = True  # noqa: N806
        return R

    def load_data(self):  # noqa: D102
        """
        Loads wave parameters from `waves.txt` if `WaveType` is -1.

        If a valid file is found, it reads the wave parameters:

        - The first three lines are header-like info (only the 'NumberOfWaves' line is used).
        - After skipping three lines, wave parameters are loaded. The shape of `self.data`
          is `(N_data, 4)`, where `N_data` is determined by reading the 'NumberOfWaves' 
          line in the file.
        """
        if self.WaveType == -1:
            with open(self.filename) as wf:  # noqa: PTH123
                for line in wf:
                    if 'NumberOfWaves' in line:
                        self.N_data = int(line.split()[1])
            temp = np.loadtxt(self.filename, skiprows=3, dtype=ti2np(self.precision))
            self.data = np.zeros((self.N_data, 4), ti2np(self.precision))
            for i in range(self.N_data):
                if self.N_data == 1:
                    self.data[i, :] = temp
                else:
                    self.data[i, :] = temp[i]
            self.data = self.data.astype(ti2np(self.precision))
        else:
            pass

    def get_data(self):  # noqa: D102
        """
        Ensures wave data is loaded if reading from a file, then returns a Taichi field.

        If `WaveType` is -1, calls `load_data()` to populate `self.data` from the file.
        Converts the resulting NumPy array into a Taichi field and returns it.

        Returns:
            ti.field: A Taichi field of shape `(N_data, 4)` containing wave parameters.
        """
        data = None  # noqa: N806
        if self.WaveType == -1:
            self.load_data()
            data = ti.field(self.precision, shape=(self.N_data, 4))
            data.from_numpy(self.data)
        elif self.WaveType == 1 or self.WaveType == 2:
            data = ti.field(self.precision, shape=(self.N_data, 4))
            data.from_numpy(np.array([self.sine_wave]))
        else:
            data = ti.field(self.precision, shape=(self.N_data, 4))
            data.from_numpy(np.array([[0, 0, 0, 0]]))
        return data


@ti.data_oriented
class Domain:  # noqa: D101
    """
    Defines the numerical domain for wave propagation solver in CelerisAi.

    This class sets up the domain geometry (x1, x2, y1, y2) and resolution (Nx, Ny), 
    handles bathymetric/topographic data (via an instance of a `Topodata` class), 
    configures boundary sea levels on each face (north, south, east, west), 
    and stores critical parameters such as Courant number (`Courant`), friction, 
    and base depth. Two main branches are supported for configuration:

        1. **Celeris format**: Reads from a `config.json` file (if `topodata.datatype == "celeris"`).
        2. **Manual**: Uses provided arguments directly.

    The class also defines utility methods to:

        - Create a meshgrid for the domain (`grid`).
        - Load and interpolate topographic/bathymetric data (`topofield`, `bottom`).
        - Compute maximum depth (`maxdepth`) and highest topography (`maxtopo`).
        - Compute the time step (`dt`) based on Courant criteria.
        - Provide reflection indices for solid boundary conditions (`reflect_x`, `reflect_y`).
        - Create Taichi field templates for solver states (`states`, `states_one`).

    Attributes:
        precision (ti.types.primitive_types): Taichi precision (e.g., `ti.f32`, `ti.f64`).
        x1 (float): Minimum x-coordinate of the domain.
        x2 (float): Maximum x-coordinate of the domain.
        y1 (float): Minimum y-coordinate of the domain.
        y2 (float): Maximum y-coordinate of the domain.
        Nx (int): Number of grid cells in the x-direction.
        Ny (int): Number of grid cells in the y-direction.
        topodata (Topodata): Instance that handles bathymetry/topography data.
        north_sl (float): Sea level at the north boundary.
        south_sl (float): Sea level at the south boundary.
        east_sl (float): Sea level at the east boundary.
        west_sl (float): Sea level at the west boundary.
        Courant (float): Courant number for numerical stability.
        isManning (int): Switch indicating whether Manning friction is used.
        friction (float): Friction (e.g., Manning n) value.
        base_depth_ (float or None): Reference base depth of the domain. 
            If None, it is inferred from the topography.
        Boundary_shift (int): Shift parameter used for boundary indexing or conditions.
        pixels (ti.field): Taichi field for potential 2D visualization or debugging (shape = [Nx, Ny]).
        g (float): Gravitational constant (9.80665).
        configfile (dict or None): Loaded JSON dictionary if using Celeris format.
        seaLevel (float): Reference sea level (set to 0.0 here).

    Example:
        >>> from celeris.domain import Domain,Topodata
        >>> topo = Topodata(filename='bathy.txt', datatype='xyz')
        >>> domain = Domain(x1=0, x2=100, y1=0, y2=100, Nx=100, Ny=100, topodata=topo)
        >>> xgrid, ygrid, bathy = domain.grid()
        >>> dt = domain.dt()
        >>> print(f"Time step: {dt}")
    """
    def __init__(  # noqa: C901, PLR0913
        self,
        precision=ti.f32,
        x1=0.0,
        x2=0.0,
        y1=0.0,
        y2=0.0,
        Nx=1,  # noqa: N803
        Ny=1,  # noqa: N803
        path='./scratch',
        configfile='config.json',
        topodata=None,
        north_sl=0.0,
        south_sl=0.0,
        east_sl=0.0,
        west_sl=0.0,
        Courant=0.2,  # noqa: N803
        isManning=0,  # noqa: N803
        friction=0.001,
        base_depth=None,
        BoundaryShift=4,  # noqa: N803
    ):
        """
        Initializes the Domain object with specified domain parameters and reads from 
        a `config.json` file if the topodata datatype is 'celeris'.

        Args:
            precision (ti.types.primitive_types, optional): Taichi precision (e.g. `ti.f32`). 
                Defaults to `ti.f32`.
            x1 (float, optional): Minimum x-coordinate. Defaults to 0.0.
            x2 (float, optional): Maximum x-coordinate. Defaults to 0.0.
            y1 (float, optional): Minimum y-coordinate. Defaults to 0.0.
            y2 (float, optional): Maximum y-coordinate. Defaults to 0.0.
            Nx (int, optional): Number of grid cells in x-direction. Defaults to 1.
            Ny (int, optional): Number of grid cells in y-direction. Defaults to 1.
            topodata (Topodata, optional): Instance of Topodata class that manages 
                bathymetric/topographic data. Defaults to None.
            north_sl (float, optional): Sea level at north boundary. Defaults to 0.0.
            south_sl (float, optional): Sea level at south boundary. Defaults to 0.0.
            east_sl (float, optional): Sea level at east boundary. Defaults to 0.0.
            west_sl (float, optional): Sea level at west boundary. Defaults to 0.0.
            Courant (float, optional): Courant number. Defaults to 0.2.
            isManning (int, optional): Switch for using Manning friction. Defaults to 0.
            friction (float, optional): Friction parameter (e.g., Manning n). 
                Defaults to 0.001.
            base_depth (float, optional): Base depth for the domain; if None, 
                will be inferred from topography. Defaults to None.
            BoundaryShift (int, optional): Shift value used for boundary conditions. 
                Defaults to 4.
        """
        self.precision = precision
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.Nx = Nx
        self.Ny = Ny
        self.topodata = topodata
        self.north_sl = north_sl
        self.south_sl = south_sl
        self.east_sl = east_sl
        self.west_sl = west_sl
        self.seaLevel = 0.0
        self.g = 9.80665
        self.path = path
        self.configfile = None
        self.Boundary_shift = BoundaryShift
        if self.topodata.datatype == 'celeris' or self.topodata.datatype == 'txt':  # noqa: PLR1714
            with open(os.path.join(self.path, configfile)) as uf:  # noqa: PTH118, PTH123
                self.configfile = json.load(uf)
            if checjson('WIDTH', self.configfile) == 1:
                self.Nx = int(self.configfile['WIDTH'])
            else:
                if (self.topodata != None) and (self.topodata.datatype == 'celeris' or self.topodata.datatype == 'txt'):
                    self.Nx = self.topodata.z().shape[0]
                else:
                    self.Nx = Nx
                self.configfile["WIDTH"] = self.Nx    
            if checjson('HEIGHT', self.configfile) == 1:
                self.Ny = int(self.configfile['HEIGHT'])
            else:
                if (self.topodata != None) and (self.topodata.datatype == 'celeris' or self.topodata.datatype == 'txt'):
                    self.Ny = self.topodata.z().shape[1]
                else:
                    self.Ny = Ny
                self.configfile["HEIGHT"] = self.Ny
            if checjson('dx', self.configfile) == 1:
                self.dx = float(self.configfile['dx'])
            else:
                self.dx = (self.x2 - self.x1) / self.Nx
            if checjson('dy', self.configfile) == 1:
                self.dy = float(self.configfile['dy'])
            else:
                self.dy = (self.y2 - self.y1) / self.Ny
            if checjson('Courant_num', self.configfile) == 1:
                self.Courant = float(self.configfile['Courant_num'])
                print('Courant number from config file:', self.Courant)  # noqa: T201
            else:
                self.Courant = Courant
            if checjson('base_depth', self.configfile) == 1:
                self.base_depth_ = float(self.configfile['base_depth'])
            else:
                self.base_depth_ = base_depth
            if checjson('visual_depth', self.configfile) == 1:
                self.visual_depth_ = float(self.configfile['visual_depth'])
            else:
                self.visual_depth_ = self.base_depth_
            if checjson('friction', self.configfile) == 1:
                self.friction = float(self.configfile['friction'])
            else:
                self.friction = friction
            if checjson('isManning', self.configfile) == 1:
                self.isManning = int(self.configfile['isManning'])
            else:
                self.isManning = isManning
            if checjson('boundary_shift', self.configfile) == 1:
                self.Boundary_shift = int(self.configfile['boundary_shift'])

            # Sea level is not a constant, it is a time series, but we can set the initial value here
            # Allow for north, south, east, and west sea levels to be different from each other
            # TODO: interpolate between the four corners to get the sea level at each exterior grid-line points  # noqa: TD002
            # TODO: interpolate between NWSE sea levels to get the sea level at each grid point  # noqa: TD002
            # TODO: interpolate between exterior grid-line sea levels and interior topography to get the sea level at each grid point  # noqa: TD002
            if checjson('seaLevel', self.configfile) == 1:
                self.seaLevel = float(self.configfile['seaLevel'])
            elif checjson('sea_level', self.configfile) == 1:
                self.seaLevel = float(self.configfile['sea_level'])
            elif checjson('swl', self.configfile) == 1:
                self.seaLevel = float(self.configfile['swl'])
            else:
                self.seaLevel = 0.0

            if checjson('north_seaLevel', self.configfile) == 1:
                self.north_sl = float(self.configfile['north_seaLevel'])
            elif checjson('north_sl', self.configfile) == 1:
                self.north_sl = float(self.configfile['north_sl'])
            elif checjson('north_sea_level', self.configfile) == 1:
                self.north_sl = float(self.configfile['north_sea_level'])
            else:
                self.north_sl = self.seaLevel

            if checjson('south_seaLevel', self.configfile) == 1:
                self.south_sl = float(self.configfile['south_seaLevel'])
            elif checjson('south_sl', self.configfile) == 1:
                self.south_sl = float(self.configfile['south_sl'])
            elif checjson('south_sea_level', self.configfile) == 1:
                self.south_sl = float(self.configfile['south_sea_level'])
            else:
                self.south_sl = self.seaLevel

            if checjson('east_seaLevel', self.configfile) == 1:
                self.east_sl = float(self.configfile['east_seaLevel'])
            elif checjson('east_sl', self.configfile) == 1:
                self.east_sl = float(self.configfile['east_sl'])
            elif checjson('east_sea_level', self.configfile) == 1:
                self.east_sl = float(self.configfile['east_sea_level'])
            else:
                self.east_sl = self.seaLevel

            if checjson('west_seaLevel', self.configfile) == 1:
                self.west_sl = float(self.configfile['west_seaLevel'])
            elif checjson('west_sl', self.configfile) == 1:
                self.west_sl = float(self.configfile['west_sl'])
            elif checjson('west_sea_level', self.configfile) == 1:
                self.west_sl = float(self.configfile['west_sea_level'])
            else:
                self.west_sl = self.seaLevel

        if self.topodata.datatype == 'xyz':
            self.dy = (self.y2 - self.y1) / self.Ny
            self.dx = (self.x2 - self.x1) / self.Nx
            self.isManning = isManning
            self.friction = friction
            self.Courant = Courant
            self.base_depth_ = base_depth

        if self.topodata.datatype=='xz':
            self.Ny = 1
            self.dy = 1.0
            self.dx = (self.x2 - self.x1)/self.Nx
            self.isManning = isManning
            self.friction = friction
            self.Courant = Courant
            self.base_depth_ = base_depth      

        self.pixels = ti.field(float, shape=(self.Nx, self.Ny))

    def topofield(self):  # noqa: D102
        """
        Loads and/or interpolates topographic/bathymetric data into a NumPy meshgrid 
        and returns it in a format suitable for use in the solver.

        Returns:
            tuple:
                - **x_out** (numpy.ndarray): Mesh of x-coordinates for the domain.
                - **y_out** (numpy.ndarray): Mesh of y-coordinates for the domain 
                  (or 1D array if `datatype='xz'`).
                - **z_out** (numpy.ndarray): Corresponding bathymetry/topography values.
        Raises:
            ValueError: If `topodata.datatype` is not one of 'celeris', 'xyz', or 'xz'.
        """
        if self.topodata.datatype == 'celeris' or self.topodata.datatype == 'txt':  # noqa: PLR1714
            x_out, y_out = np.meshgrid(
                np.arange(0.0, self.Nx * self.dx, self.dx),
                np.arange(0, self.Ny * self.dy, self.dy),
            )
            foo = self.topodata.z(seaLevel=self.seaLevel)
            return x_out, y_out, foo.T
        if self.topodata.datatype == 'xyz':  # noqa: RET503
            dum = self.topodata.z()
            x_out, y_out = np.meshgrid(
                np.arange(self.x1, self.x2, self.dx),
                np.arange(self.y1, self.y2, self.dy),
            )
            dem = griddata(dum[:, :2], dum[:, 2], (x_out, y_out), method='nearest')
            return x_out.T, y_out.T, dem.T
        if self.topodata.datatype=='xz':
            dum = self.topodata.z()
            x_out = np.arange( self.x1, self.x2, self.dx)
            dem = np.interp(x_out,dum[:,0],dum[:,1])
            return x_out, dem

    def bottom(self):  # noqa: D102
        """
        Creates a 3D NumPy array of shape (4, Nx, Ny) to store bottom elevation 
        (inverted sign), plus any other auxiliary fields (e.g., near-dry flags).

        Index mapping:
            - [2, :, :] => Stores the bathymetry/topography (with a -1 factor).
            - [3, :, :] => A placeholder used for near-dry or similar state flags.

        Returns:
            ti.field: A Taichi field of shape [4, Nx, Ny] with the bottom information.
        """
        nbottom = np.zeros((4, self.Nx, self.Ny), dtype=ti2np(self.precision))
        if self.topodata.datatype=='xz':
            # VERSION 1D
            nbottom[2,:,0] =-1.0* self.topofield()[1]
        else:
            nbottom[2] =-1.0* self.topofield()[2]
        nbottom[3] = 99.0  # To be used in neardry

        bottom = ti.field(self.precision, shape=(4, self.Nx, self.Ny))
        bottom.from_numpy(nbottom)
        return bottom

    def grid(self):  # noqa: D102
        """
        Returns the meshgrid of domain coordinates and corresponding bathymetry/topography.

        Returns:
            tuple:
                - **xx** (numpy.ndarray): x-coordinates (shape: Nx x Ny if 2D; Nx if 1D).
                - **yy** (numpy.ndarray): y-coordinates (shape: Nx x Ny if 2D; 
                  or topography for 1D).
                - **zz** (numpy.ndarray): Bathymetry/topography data in 2D case; 
                  None or irrelevant in 1D case (depending on interpretation).
        """
        xx = self.topofield()[0]
        yy = self.topofield()[1]
        zz = self.topofield()[2]
        return xx, yy, zz

    def maxdepth(self):  # noqa: D102
        """
        Computes the maximum depth in the domain. If `base_depth_` is specified, 
        returns that. Otherwise:

            - For 1D ('xz'): returns the maximum of the interpolated topofield array.
            - For 2D ('xyz', 'celeris'): returns the maximum of topofield array values.

        Returns:

            float: Maximum depth (`base_depth_` if set, else maximum from topofield).
        """
        if self.base_depth_ == None:  # noqa: E711
            # VERSION 1D
            if self.topodata.datatype=='xz':
                return np.max(self.topofield()[1])
            else: 
                return np.max(self.topofield()[2])
        else:
            return self.base_depth_
        
    def visualdepth(self):
        """
        Returns the visual depth for the domain. If `visual_depth_` is set, returns that.
        Otherwise, returns `base_depth_`.

        Returns:
            float: The visual depth of the domain.
        """
        if self.visual_depth_ == None:
            return self.maxdepth()
        else:
            return self.visual_depth_

    def maxtopo(self):  # noqa: D102
        """
        Computes the highest topographic elevation in the domain. 
        For 1D ('xz'), returns the minimum of the array (assuming negative values 
        represent depth). For 2D, returns the minimum of the topofield array 
        for a similar reason.

        Returns:
            float: The highest elevation (or least negative) in the domain.
        """
        # VERSION 1D
        if self.topodata.datatype=='xz':
            return np.min(self.topofield()[1])
        else:
            return np.min(self.topofield()[2])

    def dt(self, Courant: ti.f32):  # noqa: D102
        """
        Computes the time step based on the Courant criterion:
            dt = Courant * dx / sqrt(g * maxdepth)

        Returns:
            float: The computed time step.
        """
        maxdepth = self.maxdepth()  # noqa: F841
        return Courant * self.dx / np.sqrt(self.g * self.maxdepth())

    def reflect_x(self):  # noqa: D102
        """
        Computes an x-reflection index for enforcing solid boundary conditions.

        Returns:
            int: The x-reflection index (2*(Nx-3)).
        """
        return 2 * (self.Nx - 3)

    def reflect_y(self):  # noqa: D102
        """
        Computes a y-reflection index for enforcing solid boundary conditions.

        Returns:
            int: The y-reflection index (2*(Ny-3)).
        """
        return 2 * (self.Ny - 3)

    def states(self):  # noqa: D102
        """
        Creates a Taichi Vector field of shape [Nx, Ny], each containing 4 components 
        (e.g., water depth, momentum in x, momentum in y, and an scalaritional parameter).

        Returns:
            ti.types.vector.field: A 4-component vector field in Taichi.
        """
        foo = ti.Vector.field(4, self.precision, shape=(self.Nx, self.Ny))
        return foo  # noqa: RET504

    def states_one(self):  # noqa: D102
        """
        Creates a Taichi Vector field of shape [Nx, Ny], each containing 1 component.

        Returns:
            ti.types.vector.field: A 1-component vector field in Taichi.
        """
        foo = ti.Vector.field(1, self.precision, shape=(self.Nx, self.Ny))
        return foo  # noqa: RET504


if __name__ == '__main__':
    print('Domain module used in Celeris')  # noqa: T201
