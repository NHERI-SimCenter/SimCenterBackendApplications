import numpy as np
import os
import json
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import taichi as ti
import taichi.math as tm

#ti.init(arch=ti.cuda)
def checjson(variable,data):
    # To check if variables exist in Celeris configuration file
    R=0
    if variable in data:
        R =1
    return R

def ti2np(ti_type):
    # Change the dtype between Taichi and Numpy
    np_type = np.float32
    if ti_type==ti.f32:
        np_type = np.float32
    if ti_type==ti.f16:
        np_type=np.float16
    return np_type

class Topodata:
    def __init__(self,
                 filename=None,
                 datatype=None,
                 path=None):
        self.filename = filename
        self.datatype = datatype
        self.path = path

    def z(self):
        if self.datatype == 'xyz':
            if self.path!=None:
                return np.loadtxt(os.path.join(self.path, self.filename))
            else:
                return np.loadtxt(self.filename)
        if self.datatype == 'celeris':
                bathy =  np.loadtxt(os.path.join(self.path, 'bathy.txt'))
                return bathy*-1
        else:
            return 'No supported format'


class BoundaryConditions:
    """
    0: solid wall  ; 1: Sponge   ; 2:SineWaves   ; 3:DamBreak
    """
    def __init__(self,
                 celeris=True,
                 precision = ti.f32,
                 North=None,
                 South=None,
                 East =None,
                 West =None,
                 WaveType=-1,
                 Amplitude = 0.5,
                 path = './scratch',
                 filename='waves.txt',
                 BoundaryWidth= 20,
                 init_eta=5,
                 sine_wave=[0,0,0,0]
                 ):
        self.precision = precision
        self.North = North
        self.South = South
        self.East = East
        self.West = West
        self.WaveType = WaveType # -1 to read from a file
        self.sine_wave = sine_wave
        self.data = np.zeros((2,4))
        self.N_data  = 2
        self.W_data  = None
        self.init_eta=init_eta
        self.amplitude = Amplitude
        self.path=path
        self.configfile=None
        self.celeris=celeris
        if self.celeris==True:
            filename='waves.txt'
            with open(os.path.join(self.path, 'config.json'),'r') as uf:
                    self.configfile = json.load(uf)
            if checjson('BoundaryWidth',self.configfile)==1:
                self.BoundaryWidth = int(self.configfile['BoundaryWidth'])
            else:
                self.BoundaryWidth=BoundaryWidth
            if checjson('north_boundary_type',self.configfile)==1:
                self.North = int(self.configfile['north_boundary_type'])
            else:
                self.North = North
            if checjson('south_boundary_type',self.configfile)==1:
                self.South = int(self.configfile['south_boundary_type'])
            else:
                self.South = South
            if checjson('east_boundary_type',self.configfile)==1:
                self.East = int(self.configfile['east_boundary_type'])
            else:
                self.East = East
            if checjson('west_boundary_type',self.configfile)==1:
                self.West = int(self.configfile['west_boundary_type'])
            else:
                self.West = West
        else:
            self.BoundaryWidth=BoundaryWidth
            self.North = North
            self.South = South
            self.West = West
            self.East = East

        if self.WaveType==-1:
            self.filename = os.path.join(path,filename)



    def Sponge(self,width=None):
        return width

    def SineWave(self):
        return np.array(self.sine_wave,dtype=self.precision)

    def tseries(self):
        R = False
        if self.filename!=None:
            R = True
        return R
    def load_data(self):
        if self.WaveType==-1:
            with open(self.filename,'r') as wavefile:
                for line in wavefile:
                    if 'NumberOfWaves' in line:
                        self.N_data = int(line.split()[1])
            temp = np.loadtxt(self.filename, skiprows=3,dtype=ti2np(self.precision))
            self.data = np.zeros((self.N_data,4),ti2np(self.precision))
            for i in range(self.N_data):
                if self.N_data==1:
                    self.data[i,:] = temp
                else:
                    self.data[i,:] = temp[i]
            self.data = self.data.astype(ti2np(self.precision))
        else:
            pass
    def get_data(self):
        if self.WaveType==-1:
            self.load_data()
        data = ti.field(self.precision,shape=(self.N_data,4))
        data.from_numpy(self.data)
        return data


@ti.data_oriented
class Domain:
    def __init__(self,
                 precision=ti.f32,
                 x1=0.0,
                 x2=0.0,
                 y1=0.0,
                 y2=0.0,
                 Nx=1,
                 Ny=1,
                 topodata=None,
                 north_sl = 0.,
                 south_sl = 0.,
                 east_sl = 0.,
                 west_sl = 0.,
                 Courant=0.2,
                 isManning = 0,
                 friction =0.001,
                 base_depth=None,
                 BoundaryShift=4
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
        self.east_sl  = east_sl
        self.west_sl  = west_sl
        self.seaLevel = 0.0
        self.g = 9.80665
        self.configfile=None
        self.Boundary_shift=BoundaryShift
        if self.topodata.datatype=='celeris':
            with open(os.path.join(self.topodata.path, 'config.json'),'r') as uf:
                    self.configfile = json.load(uf)
            if checjson('WIDTH',self.configfile)==1:
                self.Nx = int(self.configfile['WIDTH'])
            else:
                self.Nx = Nx
            if checjson('HEIGHT',self.configfile)==1:
                self.Ny = int(self.configfile['HEIGHT'])
            else:
                self.Ny = Ny
            if checjson('dx',self.configfile)==1:
                self.dx = float(self.configfile['dx'])
            else:
                self.dx = (self.x2 - self.x1)/self.Nx
            if checjson('dy',self.configfile)==1:
                self.dy = float(self.configfile['dy'])
            else:
                self.dy = (self.y2 - self.y1)/self.Ny
            if checjson('Courant_num',self.configfile)==1:
                self.Courant = float(self.configfile['Courant_num'])
            else:
                self.Courant = Courant
            if checjson('base_depth',self.configfile)==1:
                self.base_depth_ = float(self.configfile['base_depth'])
            else:
                self.base_depth_ = base_depth
            if checjson('friction',self.configfile)==1:
                self.friction = float(self.configfile['friction'])
            else:
                self.friction = friction
            if checjson('isManning',self.configfile)==1:
                self.isManning = int(self.configfile['isManning'])
            else:
                self.isManning = isManning
        if self.topodata.datatype=='xyz':
            self.dy = (self.y2 - self.y1)/self.Ny
            self.dx = (self.x2 - self.x1)/self.Nx
            self.isManning = isManning
            self.friction = friction
            self.Courant = Courant
            self.base_depth_ = base_depth


        self.pixels = ti.field(float, shape=(self.Nx,self.Ny))

    def topofield(self):
        if self.topodata.datatype=='celeris':
            x_out, y_out = np.meshgrid( np.arange( 0.0, self.Nx*self.dx , self.dx),np.arange(0, self.Ny*self.dy,self.dy))
            foo = self.topodata.z()
            return x_out, y_out,foo.T
        if self.topodata.datatype=='xyz':
            dum = self.topodata.z()
            x_out, y_out = np.meshgrid( np.arange( self.x1, self.x2, self.dx),np.arange(self.y1, self.y2, self.dy))
            dem = griddata(dum[:,:2], dum[:,2], (x_out, y_out), method='nearest')
            return x_out.T, y_out.T, dem.T

    def bottom(self):
        nbottom = np.zeros((4,self.Nx,self.Ny),dtype=ti2np(self.precision))
        nbottom[2] =-1.0* self.topofield()[2]
        nbottom[3] = 99.   # To be used in neardry

        bottom = ti.field(self.precision,shape=(4,self.Nx,self.Ny,))
        bottom.from_numpy(nbottom)
        return bottom

    def grid(self):
        xx = self.topofield()[0]
        yy = self.topofield()[1]
        zz = self.topofield()[2]
        return xx,yy,zz

    def maxdepth(self):
        if self.base_depth_ ==None:
            return np.max(self.topofield()[2])
        else:
            return self.base_depth_

    def maxtopo(self):
        return np.min(self.topofield()[2])

    def dt(self):
        maxdepth = self.maxdepth()
        return self.Courant*self.dx/np.sqrt(self.g*self.maxdepth())

    def reflect_x(self):
        return 2*(self.Nx-3)

    def reflect_y(self):
        return 2*(self.Ny-3)

    def states(self):
        foo = ti.Vector.field(4,self.precision,shape=(self.Nx,self.Ny,))
        return foo

    def states_one(self):
        foo = ti.Vector.field(1,self.precision,shape=(self.Nx,self.Ny,))
        return foo


if __name__ == "__main__":
    print('Domain module used in Celeris')
