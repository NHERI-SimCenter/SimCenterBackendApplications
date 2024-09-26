"""
No license was provided with the initial Celeris_TL_v0 codebase (dropbox),
so tentatively we assume all rights are reserved by the original authors.

Permission for respectful distribution related to the SimCenter's mission
was granted by Patrick Lynett and Willington Renteria on 2024-9-25.
"""  # noqa: INP001, D205

import json

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import taichi.math as tm
from scipy.interpolate import griddata

ti.init(arch=ti.cuda)


class Topodata:  # noqa: D101
    def __init__(self, filename=None, datatype=None):
        self.filename = filename
        self.datatype = datatype

    def z(self):  # noqa: D102
        if self.datatype == 'xyz':
            return 0, np.loadtxt(self.filename)
        if self.datatype == 'celeris':
            with open('config.json') as uf:  # noqa: PTH123
                fc = json.load(uf)
            bathy = np.loadtxt(self.filename)
            return fc, bathy
        return 'No supported format'


class BoundaryConditions:
    """
    0: solid wall  ; 1: Sponge   ; 2:SineWaves  ; 3:IrregularWaves ; 4:DamBreak
    """  # noqa: D200, D400

    def __init__(
        self,
        North=0,  # noqa: N803
        South=0,  # noqa: N803
        East=0,  # noqa: N803
        West=None,  # noqa: N803
        filename='irrWaves.txt',
        init_eta=5,
        sine_wave=[0, 0, 0, 0],  # noqa: B006
    ):
        self.North = North
        self.South = South
        self.East = East
        self.West = West
        self.filename = filename
        self.sine_wave = sine_wave
        self.data = None
        self.N_data = None
        self.W_data = None
        self.init_eta = init_eta

    def Sponge(self, width=None):  # noqa: N802, D102
        return width

    def SineWave(self):  # noqa: N802, D102
        return np.array(self.sine_wave)

    def tseries(self):  # noqa: D102
        R = False  # noqa: N806
        if self.filename != None:  # noqa: E711
            R = True  # noqa: N806
        return R

    def load_data(self):  # noqa: D102
        if self.filename != None:  # noqa: E711
            with open(self.filename) as wavefile:  # noqa: PTH123
                for line in wavefile:
                    if 'NumberOfWaves' in line:
                        self.N_data = int(line.split()[1])
            self.data = np.loadtxt(self.filename, skiprows=3)
            self.data = self.data.astype(np.float32)
        else:
            self.data = np.zeros((2, 4))

    def get_data(self):  # noqa: D102
        self.load_data()
        data = ti.field(ti.f32, shape=(self.N_data, 4))
        data.from_numpy(self.data[: self.N_data])
        return data


@ti.data_oriented
class Domain:  # noqa: D101
    def __init__(
        self,
        x1=ti.f32,
        x2=ti.f32,
        y1=ti.f32,
        y2=ti.f32,
        Nx=int,  # noqa: N803
        Ny=int,  # noqa: N803
        topodata=None,
        structure=None,
        north_sl=0.0,
        south_sl=0.0,
        east_sl=0.0,
        west_sl=0.0,
        Courant=None,  # noqa: ARG002, N803
        isManning=0,  # noqa: N803
        friction=0.001,
    ):
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
        self.isManning = isManning
        self.friction = friction
        self.structure = structure
        self.pixels = ti.field(float, shape=(self.Nx, self.Ny))
        self.rgb = ti.Vector.field(
            3, float, shape=(self.Nx, self.Ny)
        )  # RGB version of pixels

    def dx(self):  # noqa: D102
        return (self.x2 - self.x1) / self.Nx

    def dy(self):  # noqa: D102
        return (self.y2 - self.y1) / self.Ny

    def topofield(self):  # noqa: D102
        dum = self.topodata[1]
        x_out, y_out = np.meshgrid(
            np.arange(self.x1, self.x2, self.dx()),
            np.arange(self.y1, self.y2, self.dy()),
        )
        dem = griddata(dum[:, :2], dum[:, 2], (x_out, y_out), method='nearest')
        return x_out, y_out, dem

    def bottom(self):  # noqa: D102
        nbottom = np.zeros((4, self.Ny, self.Nx), dtype=np.float32)
        nbottom[2] = -1 * self.topofield()[2]
        nbottom[3] = 99  # To be used in neardry

        bottom = ti.field(ti.f32, shape=(4, self.Ny, self.Nx))
        bottom.from_numpy(nbottom)
        return bottom

    def grid(self):  # noqa: D102
        xx = self.topofield()[0]
        yy = self.topofield()[1]
        zz = self.topofield()[2]
        return xx, yy, zz

    def maxdepth(self):  # noqa: D102
        return np.max(self.topofield()[2])

    def maxtopo(self):  # noqa: D102
        return np.min(self.topofield()[2])

    def dt(self):  # noqa: D102
        maxdepth = self.maxdepth()
        return self.Courant * self.dx() / np.sqrt(self.g * maxdepth)

    def reflect_x(self):  # noqa: D102
        return 2 * self.Nx - 6

    def reflect_y(self):  # noqa: D102
        return 2 * self.Ny - 6

    def states(self):  # noqa: D102
        foo = ti.Vector.field(4, ti.f32, shape=(self.Ny, self.Nx))
        return foo  # noqa: RET504


if __name__ == '__main__':
    baty = Topodata('dummy.xyz', 'xyz')
    d = Domain(1.0, 100.0, 1.0, 50.0, 200, 100)
