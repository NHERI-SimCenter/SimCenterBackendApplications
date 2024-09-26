import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as clr
from scipy.interpolate import griddata
import taichi as ti
import taichi.math as tm


def celeris_matplotlib(water='seismic',land='terrain',sediment='default',SedTrans=False):
    if SedTrans==False:
        ## Water Color
        water_cmap = cm.get_cmap(water,256)
        water_cmap._init()
        water_cmap = water_cmap._lut
        N = water_cmap.shape[0]
        ramp_water = np.linspace(0,0.75,N)
        clist = []
        for i in range(N):
            clist.append((ramp_water[i],water_cmap[i]))
        ## Topo Color
        land_cmap = cm.get_cmap(land,256)
        land_cmap._init()
        land_cmap = land_cmap._lut
        N = land_cmap.shape[0]
        ramp_land = np.linspace(0.75,1,N)
        for i in range(N):
            clist.append((ramp_land[i],land_cmap[i]))
        cmap =clr.LinearSegmentedColormap.from_list("", clist,N=256)
    if SedTrans==True:
        ## Water Color
        water_cmap = cm.get_cmap(water,256)
        water_cmap._init()
        water_cmap = water_cmap._lut
        N = water_cmap.shape[0]
        ramp_water = np.linspace(0,0.5,N)
        clist = []
        for i in range(N):
            clist.append((ramp_water[i],water_cmap[i]))
        
        
        if sediment=='default':
            clist.append((0.5, 'skyblue'))
            clist.append((0.51, 'tan'))
            clist.append((0.6, 'peru'))
            clist.append((0.75, 'saddlebrown'))
        else:
            ## Sediment Color
            sediment_cmap = cm.get_cmap(sediment,256)
            sediment_cmap._init()
            sediment_cmap = sediment_cmap._lut
            N = sediment_cmap.shape[0]
            ramp_sediment = np.linspace(0.5,0.75,N)
            for i in range(N):
                clist.append((ramp_sediment[i],sediment_cmap[i]))
        ## Topo Color
        land_cmap = cm.get_cmap(land,256)
        land_cmap._init()
        land_cmap = land_cmap._lut
        N = land_cmap.shape[0]
        ramp_land = np.linspace(0.75,1,N)
        for i in range(N):
            clist.append((ramp_land[i],land_cmap[i]))
        cmap =clr.LinearSegmentedColormap.from_list("", clist,N=256)
    
    return cmap


def celeris_waves():
    clist = [(0, 'dodgerblue'),
            (0.125, 'lightsteelblue'),
            (0.25, 'lightskyblue'),
            (0.375, 'aliceblue'),
            (0.5, 'skyblue'),
            (0.625, 'cornflowerblue'),
            (0.75, 'royalblue'),
            (0.8, 'mediumseagreen'),
            (0.87, 'yellowgreen'),
            (0.95, 'greenyellow'), 
            (1, 'yellow')]
    cmap =clr.LinearSegmentedColormap.from_list("", clist,N=256)
    return cmap

@ti.func
def MinMod(a,b,c):
    # return the minimum of the positive values if all inputs are positive, 
    # the maximum of the negative values if all inputs are negative, and zero otherwise.
    a,b,c =float(a),float(b),float(c)
    min_value = 0.0
    if (a>0.0) and (b>0.0) and (c>0.0):
        min_value = min(a,b,c)
    elif (a<0.0) and (b<0.0) and (c<0.0):
        min_value = max(a,b,c)
    return min_value
@ti.func
def sineWave(x,y,t,d,amplitude,period,theta,phase):
    g = 9.81
    omega = 2.0 * np.pi / period
    k = omega * omega / (g * tm.sqrt( tm.tanh(omega * omega * d / g)))
    c = omega / k
    kx = tm.cos(theta) * x * k
    ky = tm.sin(theta) * y * k
    eta = amplitude * tm.sin(omega * t - kx - ky + phase)*min(1, t / period)
    speed = g * eta / (c * k) * tm.tanh(k * d)
    hu = speed * tm.cos(theta)
    hv = speed * tm.sin(theta)
    return eta, hu , hv
@ti.func
def Reconstruct(west, here, east, TWO_THETA):
    # west, here, east = values of U_bar at j-1, j, j+1 (or k-1, k, k+1)
    # out_west, out_east = reconstructed values of U_west and U_east at (j,k)
    z1 = TWO_THETA * (here - west)
    z2 = (east - west)
    z3 = TWO_THETA * (east - here)

    dx_grad_over_two = 0.25 * MinMod(z1, z2, z3)
    
    #mu = 0.5 * (z1 + z2 + z3) / 3.0
    #standard_deviation = tm.sqrt(((z1-mu)**2 + (z2-mu)**2 + (z3-mu)**2) / 3.0)
    
    out_east = here + dx_grad_over_two
    out_west = here - dx_grad_over_two
    return out_west, out_east

@ti.func
def CalcUV(h, hu, hv,hc,epsilon,dB_max):
    # in:  [hN, hE, hS, hW],  [huN, huE, huS, huW],  [hvN, hvE, hvS, hvW]
    # out: [uN, uE, uS, uW],  [vN, vE, vS, vW]
    h2 = h * h
    #h4 = h2 * h2
    #h4=h*h*h*h
    epsilon_c = ti.max(epsilon, dB_max)
    divide_by_h = 2.0 * h / (h2 + ti.max(h2, epsilon_c))
    #this is important - the local depth used for the edges should not be less than the difference in water depth across the edge
    #divide_by_h = tm.sqrt(2.0) * h / tm.sqrt(h4 + tm.max(h4, epsilon))
    # divide_by_h = h / np.maximum(h2, epsilon)
    u = divide_by_h * hu
    v = divide_by_h * hv
    c = divide_by_h * hc
    return u, v,c
@ti.func
def CalcUV_Sed(h, hc1, hc2, hc3, hc4, epsilon,dB_max):
    h2 = h*h
    epsilon_c = ti.max(epsilon, dB_max)
    #h4=h*h*h*h
    divide_by_h = 2.0 * h / (h2 + ti.max(h2, epsilon_c))
    c1 = divide_by_h * hc1
    c2 = divide_by_h * hc2
    c3 = divide_by_h * hc3
    c4 = divide_by_h * hc4
    return c1,c2,c3,c4

@ti.func
def CorrectW(B_west, B_east, w_bar, w_west, w_east):
    if w_east < B_east:
        w_east = B_east
        w_west = max(B_west, 2 * w_bar - B_east)
    elif w_west < B_west:
        w_east = max(B_east, 2 * w_bar - B_west)
        w_west = B_west
    return w_west, w_east

@ti.func
def NumericalFlux(aplus, aminus, Fplus, Fminus, Udifference):
    numerical_flux = 0.0
    if (aplus - aminus) != 0.0:
        numerical_flux = (aplus * Fminus - aminus * Fplus + aplus * aminus * Udifference) / (aplus - aminus)
    return numerical_flux

@ti.func
def ScalarAntiDissipation(uplus, uminus, aplus, aminus, epsilon):
    R = 0.0   # Defaul return if none of the conditions are met
    if (aplus!=0.0) and (aminus!=0.0) :
        Fr = abs(uplus) / aplus if abs(uplus) >= abs(uminus) else abs(uminus) / aminus
        R = (Fr + epsilon ) / (Fr + 1.0)
    elif aplus==0.0 or aminus==0.0:
        R = epsilon
    return R


@ti.func
def FrictionCalc(hu, hv, h, base_depth,delta,epsilon, isManning, g, friction): 
    h_scaled = h / base_depth
    h2 = h_scaled*h_scaled
    h4 = h2**2
    divide_by_h2 = 2.0 * h2 / (h4 + max(h4, 1.e-6)) / base_depth / base_depth
    divide_by_h = 1. / max(h, delta)
    
    f = friction
    if isManning == 1:
        f = g * friction**2 * abs(divide_by_h)**(1./3.)
    f = min(f,0.5)
    f = f * tm.sqrt(hu**2 + hv**2) * divide_by_h2    
    return f

@ti.func
def Thomas(b, a, c, f):
    """
    %  Solve the  n x n  tridiagonal system for y:
    %
    %  [ a(1)  c(1)                                  ] [  y(1)  ]   [  f(1)  ]
    %  [ b(2)  a(2)  c(2)                            ] [  y(2)  ]   [  f(2)  ]
    %  [       b(3)  a(3)  c(3)                      ] [        ]   [        ]
    %  [            ...   ...   ...                  ] [  ...   ] = [  ...   ]
    %  [                    ...    ...    ...        ] [        ]   [        ]
    %  [                        b(n-1) a(n-1) c(n-1) ] [ y(n-1) ]   [ f(n-1) ]
    %  [                                 b(n)  a(n)  ] [  y(n)  ]   [  f(n)  ]
    %
    %  f must be a vector (row or column) of length n
    %  a, b, c must be vectors of length n (note that b(1) and c(n) are not used)
    """
    # Initialize vectors
    n = b.shape
    print(n)
    y = ti.var(ti.f32, shape=n)
    v = ti.var(ti.f32, shape=n)
    w = a[0]
    y[0] = f[0] / w

    # Forward elimination
    for i in range(1, n):
        v[i - 1] = c[i - 1] / w
        w = a[i] - b[i] * v[i - 1]
        y[i] = (f[i] - b[i] * y[i - 1]) / w

    # Backward substitution
    for j in range(n - 2, -1, -1):
        y[j] -= v[j] * y[j + 1]

    return y

if __name__ == "__main__":
    baty = Topodata('dummy.xyz','xyz')
    d = Domain(1.0,100.0,1.0,50.0,200,100)
    
