import json  # noqa: INP001, D100
import os

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import taichi.math as tm
from scipy.interpolate import griddata


# ti.init(arch=ti.cuda)
def checjson(variable, data):  # noqa: D103
    # To check if variables exist in Celeris configuration file
    R = 0  # noqa: N806
    if variable in data:
        R = 1  # noqa: N806
    return R


def ti2np(ti_type):  # noqa: D103
    # Change the dtype between Taichi and Numpy
    np_type = np.float32
    if ti_type == ti.f32:
        np_type = np.float32
    if ti_type == ti.f16:
        np_type = np.float16
    return np_type


class Topodata:  # noqa: D101
    def __init__(self, filename=None, datatype=None, path=None):
        self.filename = filename
        self.datatype = datatype
        self.path = path

    def z(self):  # noqa: D102
        if self.datatype == 'xyz':
            if self.path != None:  # noqa: E711
                if self.filename == None:  # noqa: E711
                    self.filename = 'test_curve.xyz'
                return np.loadtxt(os.path.join(self.path, self.filename))  # noqa: PTH118
            return np.loadtxt(self.filename)
        if self.datatype == 'celeris' or self.datatype == 'txt':  # noqa: PLR1714
            if self.path != None:  # noqa: E711
                if self.filename == None:  # noqa: E711
                    self.filename = 'bathy.txt'
                bathy = np.loadtxt(os.path.join(self.path, self.filename))  # noqa: PTH118
                return bathy * -1
        return 'No supported format'


class BoundaryConditions:
    """
    0: solid wall  ; 1: Sponge   ; 2:SineWaves   ; 3:DamBreak
    """  # noqa: D200, D400

    def __init__(  # noqa: C901
        self,
        celeris=True,  # noqa: FBT002
        precision=ti.f32,
        North=None,  # noqa: N803
        South=None,  # noqa: N803
        East=None,  # noqa: N803
        West=None,  # noqa: N803
        WaveType=-1,  # noqa: N803
        Amplitude=0.5,  # noqa: N803
        Period=10.0,  # noqa: N803
        path='./scratch',
        configfile='config.json',
        wavefile='waves.txt',
        BoundaryWidth=20,  # noqa: N803
        init_eta=5,
        sine_wave=[0, 0, 0, 0],  # noqa: B006
    ):
        self.precision = precision
        self.North = North
        self.South = South
        self.East = East
        self.West = West
        self.WaveType = WaveType  # -1 to read from a file
        self.sine_wave = sine_wave
        self.data = np.zeros((2, 4))
        self.N_data = 2
        self.W_data = None
        self.init_eta = init_eta
        self.amplitude = Amplitude
        self.period = Period
        self.path = path
        self.configfile = None
        self.celeris = celeris
        self.wavefile = None
        if self.celeris == True:  # noqa: E712
            filename = wavefile
            with open(os.path.join(self.path, configfile)) as uf:  # noqa: PTH118, PTH123
                self.configfile = json.load(uf)
            if checjson('WaveType', self.configfile) == 1:
                self.WaveType = int(self.configfile['WaveType'])
            if checjson('amplitude', self.configfile) == 1:
                self.amplitude = float(self.configfile['amplitude'])
            if checjson('period', self.configfile) == 1:
                self.period = float(self.configfile['period'])
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
            self.wavefile = wavefile  # Just a string filename without a path, not a json object as for self.configfile
            self.filename = os.path.join(path, filename)  # noqa: PTH118

    def Sponge(self, width=None):  # noqa: N802, D102
        return width

    def SineWave(self):  # noqa: N802, D102
        return np.array(self.sine_wave, dtype=self.precision)

    def tseries(self):  # noqa: D102
        R = False  # noqa: N806
        if self.filename != None:  # noqa: E711
            R = True  # noqa: N806
        return R

    def load_data(self):  # noqa: D102
        if self.WaveType == -1:
            self.filename = os.path.join(self.path, self.wavefile)  # noqa: PTH118
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
        if self.WaveType == -1:
            self.load_data()
        data = ti.field(self.precision, shape=(self.N_data, 4))
        data.from_numpy(self.data)
        return data


@ti.data_oriented
class Domain:  # noqa: D101
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
                self.Nx = Nx
            if checjson('HEIGHT', self.configfile) == 1:
                self.Ny = int(self.configfile['HEIGHT'])
            else:
                self.Ny = Ny
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
            else:
                self.Courant = Courant
            if checjson('base_depth', self.configfile) == 1:
                self.base_depth_ = float(self.configfile['base_depth'])
            else:
                self.base_depth_ = base_depth
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

        self.pixels = ti.field(float, shape=(self.Nx, self.Ny))

    def topofield(self):  # noqa: D102
        if self.topodata.datatype == 'celeris' or self.topodata.datatype == 'txt':  # noqa: PLR1714
            x_out, y_out = np.meshgrid(
                np.arange(0.0, self.Nx * self.dx, self.dx),
                np.arange(0, self.Ny * self.dy, self.dy),
            )
            foo = self.topodata.z()
            return x_out, y_out, foo.T
        if self.topodata.datatype == 'xyz':  # noqa: RET503
            dum = self.topodata.z()
            x_out, y_out = np.meshgrid(
                np.arange(self.x1, self.x2, self.dx),
                np.arange(self.y1, self.y2, self.dy),
            )
            dem = griddata(dum[:, :2], dum[:, 2], (x_out, y_out), method='nearest')
            return x_out.T, y_out.T, dem.T

    def bottom(self):  # noqa: D102
        nbottom = np.zeros((4, self.Nx, self.Ny), dtype=ti2np(self.precision))
        nbottom[2] = -1.0 * self.topofield()[2]
        nbottom[3] = 99.0  # To be used in neardry

        bottom = ti.field(self.precision, shape=(4, self.Nx, self.Ny))
        bottom.from_numpy(nbottom)
        return bottom

    def grid(self):  # noqa: D102
        xx = self.topofield()[0]
        yy = self.topofield()[1]
        zz = self.topofield()[2]
        return xx, yy, zz

    def maxdepth(self):  # noqa: D102
        if self.base_depth_ == None:  # noqa: E711
            return np.max(self.topofield()[2])
        return self.base_depth_

    def maxtopo(self):  # noqa: D102
        return np.min(self.topofield()[2])

    def dt(self):  # noqa: D102
        maxdepth = self.maxdepth()  # noqa: F841
        return self.Courant * self.dx / np.sqrt(self.g * self.maxdepth())

    def reflect_x(self):  # noqa: D102
        return 2 * (self.Nx - 3)

    def reflect_y(self):  # noqa: D102
        return 2 * (self.Ny - 3)

    def states(self):  # noqa: D102
        foo = ti.Vector.field(4, self.precision, shape=(self.Nx, self.Ny))
        return foo  # noqa: RET504

    def states_one(self):  # noqa: D102
        foo = ti.Vector.field(1, self.precision, shape=(self.Nx, self.Ny))
        return foo  # noqa: RET504


if __name__ == '__main__':
    print('Domain module used in Celeris')  # noqa: T201
