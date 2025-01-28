import os  # noqa: INP001, D100

import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import taichi.math as tm
from matplotlib import cm
from scipy.interpolate import griddata

Vpi = np.pi  # 3.141592653589793


def ColorsfromMPL(cmap='Blues'):  # noqa: N802, D103
    cm_cmap = cm.get_cmap(cmap, 16)
    cm_cmap._init()  # noqa: SLF001
    cm_cmap = cm_cmap._lut  # noqa: SLF001
    cm_cmap = cm_cmap.astype(np.float16)
    return cm_cmap  # noqa: RET504


def celeris_matplotlib(  # noqa: D103
    water='seismic',
    land='terrain',
    sediment='default',
    SedTrans=False,  # noqa: FBT002, N803
):
    if SedTrans == False:  # noqa: E712
        ## Water Color
        water_cmap = cm.get_cmap(water, 256)
        water_cmap._init()  # noqa: SLF001
        water_cmap = water_cmap._lut  # noqa: SLF001
        N = water_cmap.shape[0]  # noqa: N806
        ramp_water = np.linspace(0, 0.75, N)
        clist = []
        for i in range(N):
            clist.append((ramp_water[i], water_cmap[i]))  # noqa: PERF401
        ## Topo Color
        land_cmap = cm.get_cmap(land, 256)
        land_cmap._init()  # noqa: SLF001
        land_cmap = land_cmap._lut  # noqa: SLF001
        N = land_cmap.shape[0]  # noqa: N806
        ramp_land = np.linspace(0.75, 1, N)
        for i in range(N):
            clist.append((ramp_land[i], land_cmap[i]))  # noqa: PERF401
        cmap = clr.LinearSegmentedColormap.from_list('', clist, N=256)
    if SedTrans == True:  # noqa: E712
        ## Water Color
        water_cmap = cm.get_cmap(water, 256)
        water_cmap._init()  # noqa: SLF001
        water_cmap = water_cmap._lut  # noqa: SLF001
        N = water_cmap.shape[0]  # noqa: N806
        ramp_water = np.linspace(0, 0.5, N)
        clist = []
        for i in range(N):
            clist.append((ramp_water[i], water_cmap[i]))

        if sediment == 'default':
            clist.append((0.5, 'skyblue'))
            clist.append((0.51, 'tan'))
            clist.append((0.6, 'peru'))
            clist.append((0.75, 'saddlebrown'))
        else:
            ## Sediment Color
            sediment_cmap = cm.get_cmap(sediment, 256)
            sediment_cmap._init()  # noqa: SLF001
            sediment_cmap = sediment_cmap._lut  # noqa: SLF001
            N = sediment_cmap.shape[0]  # noqa: N806
            ramp_sediment = np.linspace(0.5, 0.75, N)
            for i in range(N):
                clist.append((ramp_sediment[i], sediment_cmap[i]))
        ## Topo Color
        land_cmap = cm.get_cmap(land, 256)
        land_cmap._init()  # noqa: SLF001
        land_cmap = land_cmap._lut  # noqa: SLF001
        N = land_cmap.shape[0]  # noqa: N806
        ramp_land = np.linspace(0.75, 1, N)
        for i in range(N):
            clist.append((ramp_land[i], land_cmap[i]))
        cmap = clr.LinearSegmentedColormap.from_list('', clist, N=256)

    return cmap


def celeris_waves():  # noqa: D103
    clist = [
        (0, 'dodgerblue'),
        (0.125, 'lightsteelblue'),
        (0.25, 'lightskyblue'),
        (0.375, 'aliceblue'),
        (0.5, 'skyblue'),
        (0.625, 'cornflowerblue'),
        (0.75, 'royalblue'),
        (0.8, 'mediumseagreen'),
        (0.87, 'yellowgreen'),
        (0.95, 'greenyellow'),
        (1, 'yellow'),
    ]
    cmap = clr.LinearSegmentedColormap.from_list('', clist, N=256)
    return cmap  # noqa: RET504


@ti.func
def MinMod(a, b, c):  # noqa: N802, D103
    # return the minimum of the positive values if all inputs are positive,
    # the maximum of the negative values if all inputs are negative, and zero otherwise.
    a, b, c = float(a), float(b), float(c)
    min_value = 0.0
    if (a > 0.0) and (b > 0.0) and (c > 0.0):
        min_value = ti.min(a, b, c)
    elif (a < 0.0) and (b < 0.0) and (c < 0.0):
        min_value = ti.max(a, b, c)
    return min_value


@ti.func
def cosh(x):  # noqa: D103
    return 0.5 * (ti.exp(x) + ti.exp(-1 * x))


@ti.func
def sineWave(x, y, t, d, amplitude, period, theta, phase, g, wave_type):  # noqa: N802, D103
    omega = 2.0 * Vpi / period
    k = omega * omega / (g * ti.sqrt(ti.tanh(omega * omega * d / g)))
    c = omega / k
    kx = ti.cos(theta) * x * k
    ky = ti.sin(theta) * y * k
    eta = amplitude * ti.sin(omega * t - kx - ky + phase) * ti.min(1.0, t / period)
    ### Check this is only valid for sinewves/irregualr
    num_waves = 0
    if wave_type == 2:  # noqa: PLR2004
        num_waves = 4
    if num_waves > 0:
        eta = eta * ti.max(
            0.0, ti.min(1.0, ((float(num_waves) * period - t) / period))
        )

    speed = g * eta / (c * k) * ti.tanh(k * d)
    hu = speed * tm.cos(theta)
    hv = speed * tm.sin(theta)
    return ti.Vector([eta, hu, hv])


@ti.func
def Reconstruct(west, here, east, TWO_THETAc):  # noqa: N802, N803, D103
    # west, here, east = values of U_bar at j-1, j, j+1 (or k-1, k, k+1)
    # out_west, out_east = reconstructed values of U_west and U_east at (j,k)
    z1 = TWO_THETAc * (here - west)
    z2 = east - west
    z3 = TWO_THETAc * (east - here)
    min_value = 0.0
    if (z1 > 0.0) and (z2 > 0.0) and (z3 > 0.0):
        min_value = ti.min(ti.min(z1, z2), z3)
    elif (z1 < 0.0) and (z2 < 0.0) and (z3 < 0.0):
        min_value = ti.max(ti.max(z1, z2), z3)

    dx_grad_over_two = 0.25 * min_value
    return ti.Vector([here - dx_grad_over_two, here + dx_grad_over_two])


@ti.func
def CalcUV(h, hu, hv, hc, epsilon, dB_max):  # noqa: N802, N803, D103
    # in:  [hN, hE, hS, hW]
    # out: [uN, uE, uS, uW]
    epsilon_c = ti.max(epsilon, dB_max)
    divide_by_h = 2.0 * h / (h * h + ti.max(h * h, epsilon_c))
    # this is important - the local depth used for the edges should not be less than the difference in water depth across the edge
    # divide_by_h = h / np.maximum(h2, epsilon)  #u = divide_by_h * hu #v = divide_by_h * hv #c = divide_by_h * hc
    return divide_by_h * hu, divide_by_h * hv, divide_by_h * hc


@ti.func
def CalcUV_Sed(h, hc1, hc2, hc3, hc4, epsilon, dB_max):  # noqa: N802, N803, D103
    epsilon_c = ti.max(epsilon, dB_max)
    # h4=h*h*h*h
    divide_by_h = ti.sqrt(2.0) * h / (h * h + ti.max(h * h, epsilon_c))
    c1 = divide_by_h * hc1
    c2 = divide_by_h * hc2
    c3 = divide_by_h * hc3
    c4 = divide_by_h * hc4
    return ti.Vector([c1, c2, c3, c4])


@ti.func
def NumericalFlux(aplus, aminus, Fplus, Fminus, Udifference):  # noqa: N802, N803, D103
    numerical_flux = 0.0
    if (aplus - aminus) != 0.0:
        numerical_flux = (
            aplus * Fminus - aminus * Fplus + aplus * aminus * Udifference
        ) / (aplus - aminus)
    return numerical_flux


@ti.func
def ScalarAntiDissipation(uplus, uminus, aplus, aminus, epsilon):  # noqa: N802, D103
    R = 0.0  # Default return if none of the conditions are met  # noqa: N806
    if aplus != 0.0 and aminus != 0.0:
        if ti.abs(uplus) >= ti.abs(uminus):
            Fr = ti.abs(uplus) / aplus  # noqa: N806
        else:
            Fr = ti.abs(uminus) / aminus  # noqa: N806
        R = (Fr + epsilon) / (Fr + 1.0)  # noqa: N806
    elif aplus == 0.0 or aminus == 0.0:
        R = epsilon  # noqa: N806
    return R


@ti.func
def FrictionCalc(hu, hv, h, base_depth, delta, isManning, g, friction):  # noqa: N802, N803, D103
    h_scaled = h / base_depth
    h2 = h_scaled * h_scaled
    h4 = h2 * h2
    divide_by_h2 = 2.0 * h2 / (h4 + ti.max(h4, 1.0e-6)) / base_depth / base_depth
    divide_by_h = 1.0 / ti.max(h, delta)

    f = friction
    if isManning == 1:
        f = g * ti.pow(friction, 2) * ti.pow(ti.abs(divide_by_h), 1.0 / 3.0)
    f = ti.min(f, 0.5)
    f = f * ti.sqrt(hu * hu + hv * hv) * divide_by_h2
    return f  # noqa: RET504


def main():
    """Define utility functions used in Celeris."""
    return


if __name__ == '__main__':
    """Define utility functions used in Celeris."""
    print('Module of functions used in Celeris')  # noqa: T201
    main()
