#!/usr/bin/env python3

"""
Compute inline/total hydrodynamic force and moments on a monopile using Morison's equation
"""

from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fractions import Fraction
import matplotlib as mpl
import os, sys
import re
import json
import argparse

# Local
from welib.tools.figure import defaultRC

defaultRC()
from welib.tools.colors import python_colors
from welib.hydro.wavekin import *
from welib.hydro.morison import *


# --- Parameters
g = 9.81  # gravity [m/s^2]
h = 30.0  # water depth [m]
rho = 1000  # water density
D = 6  # monopile diameter [m]
CD = 1  # given
CM = 2  #
a = 3  # wave peak amplitude [m]
T = 12.0  # period [s]
eps = 0  # phase shift [rad]
f = 1.0 / T
k = wavenumber(f, h, g)

nz = 30  # number of points used in the z direction to compute loads

z_ref = -h  # reference point for moment calculation

# --------------------------------------------------------------------------------}
# --- Inline force and moments as function of time, with or without Wheeler stretching
# --------------------------------------------------------------------------------{
time = np.linspace(0, T, 9)

fig1, axes1 = plt.subplots(
    2, 4, sharex=True, sharey=True, figsize=(12.8, 4.8)
)  # (6.4,4.8)
fig1.subplots_adjust(
    left=0.05, right=0.99, top=0.95, bottom=0.09, hspace=0.26, wspace=0.11
)

fig2, axes2 = plt.subplots(
    2, 4, sharex=True, sharey=True, figsize=(12.8, 4.8)
)  # (6.4,4.8)
fig2.subplots_adjust(
    left=0.05, right=0.99, top=0.95, bottom=0.09, hspace=0.26, wspace=0.11
)

XLIM = [-75, 75]  # For inline force
XLIMM = [-2500, 2500]  # For inline moment

for it, t in enumerate(time[:-1]):
    # Wave kinematics
    eta = elevation2d(a, f, k, eps, t, x=0)
    z = np.linspace(-h, eta, nz)
    u, du = kinematics2d(a, f, k, eps, h, t, z, Wheeler=True, eta=eta)
    u0, du0 = kinematics2d(a, f, k, eps, h, t, z)
    # Wave loads with wheeler
    p_tot = inline_load(u, du, D, CD, CM, rho)
    p_inertia = inline_load(u, du, D, CD * 0, CM, rho)
    p_drag = inline_load(u, du, D, CD, CM * 0, rho)
    dM = p_tot * (z - z_ref)  # [Nm/m]

    # Wave loads without Wheeler
    p_tot0 = inline_load(u0, du0, D, CD, CM, rho)
    p_inertia0 = inline_load(u0, du0, D, CD * 0, CM, rho)
    p_drag0 = inline_load(u0, du0, D, CD, CM * 0, rho)
    dM0 = p_tot0 * (z - z_ref)  # [Nm/m]

    # Plot inline force
    ax = axes1[int(it / 4), np.mod(it, 4)]
    ax.plot(p_inertia / 1000, z, '-', c=python_colors(0), label=r'$f_{inertia}$')
    ax.plot(p_drag / 1000, z, '-', c=python_colors(3), label=r'$f_{drag}$')
    ax.plot(p_tot / 1000, z, 'k-', label=r'$f_{tot}$')
    ax.plot(p_inertia0 / 1000, z, '+', c=python_colors(0))
    ax.plot(p_drag0 / 1000, z, '+', c=python_colors(3))
    ax.plot(p_tot0 / 1000, z, 'k+')
    ax.set_title('t/T={}'.format(Fraction(t / T)))
    if it == 0:
        ax.legend()
    ax.plot(XLIM, [0, 0], 'k')
    ax.plot(XLIM, [a, a], 'k--')
    ax.plot(XLIM, [-a, -a], 'k--')

    # Plot inline moment
    ax = axes2[int(it / 4), np.mod(it, 4)]
    ax.plot(dM / 1000, z, 'k-', label=r'$dM_{tot}$ with Wheeler')
    ax.plot(dM0 / 1000, z, 'k+', label=r'$dM_{tot}$ no-correction')
    ax.set_title('t/T={}'.format(Fraction(t / T)))
    if it == 0:
        ax.legend()
    ax.plot(XLIMM, [0, 0], 'k')
    ax.plot(XLIMM, [a, a], 'k--')
    ax.plot(XLIMM, [-a, -a], 'k--')


axes1[0, 0].set_xlim(XLIM)
axes1[0, 0].set_ylim([-h, a + 1])
axes1[0, 0].set_ylabel('Depth z [m]')
axes1[1, 0].set_ylabel('Depth z [m]')
axes1[1, 0].set_xlabel('Inline force [kN/m]')
axes1[1, 1].set_xlabel('Inline force [kN/m]')
axes1[1, 2].set_xlabel('Inline force [kN/m]')
axes1[1, 3].set_xlabel('Inline force [kN/m]')

fig1.savefig('forces.png')
# fig1.savefig('forces.webp')
# fig1.show()

axes2[0, 0].set_xlim(XLIMM)
axes2[0, 0].set_ylim([-h, a + 1])
axes2[0, 0].set_ylabel('Depth z [m]')
axes2[1, 0].set_ylabel('Depth z [m]')
axes2[1, 0].set_xlabel('Inline moment [kNm/m]')
axes2[1, 1].set_xlabel('Inline moment [kNm/m]')
axes2[1, 2].set_xlabel('Inline moment [kNm/m]')
axes2[1, 3].set_xlabel('Inline moment [kNm/m]')

fig2.savefig('moments.png')
# fig2.savefig('moments.webp')
# fig2.show()
# --------------------------------------------------------------------------------}
# --- Integrated force and sea bed moment over a period
# --------------------------------------------------------------------------------{
time = np.linspace(0, 60.0, 6001)

veta = np.zeros(time.shape)
vF = np.zeros(time.shape)
vM = np.zeros(time.shape)
vF0 = np.zeros(time.shape)
vM0 = np.zeros(time.shape)

XLIM = [-75, 75]  # For inline force
XLIMM = [-2500, 2500]  # For inline moment

# a=6 # NOTE: increased amplitude here to see Effect of Wheeler
elevation = np.zeros((len(time), nz))
velocity = np.zeros((len(time), nz))
accel = np.zeros((len(time), nz))
force = np.zeros((len(time), nz))

for it, t in enumerate(time):
    # Wave kinematics
    veta[it] = elevation2d(a, f, k, eps, t, x=0)
    z = np.linspace(-h, veta[it], nz)
    u, du = kinematics2d(a, f, k, eps, h, t, z, Wheeler=True, eta=veta[it])
    u0, du0 = kinematics2d(a, f, k, eps, h, t, z)
    # Wave loads with Wheeler
    p_tot = inline_load(u, du, D, CD, CM, rho)
    vF[it] = np.trapz(p_tot, z)  # [N]
    vM[it] = np.trapz(p_tot * (z - z_ref), z)  # [Nm]
    # Wave loads without Wheeler
    p_tot0 = inline_load(u0, du0, D, CD, CM, rho)
    vF0[it] = np.trapz(p_tot0, z)  # [N]
    vM0[it] = np.trapz(p_tot0 * (z - z_ref), z)  # [Nm]

    elevation[it, :] = z.copy()
    velocity[it, :] = u.copy()
    accel[it, :] = du.copy()
    force[it, :] = p_tot.copy()

# Plot
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(6.4, 4.8))  # (6.4,4.8)
fig.subplots_adjust(
    left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.14, wspace=0.20
)
# axes[0] = axes[0]
axes[0].plot(time / T, veta, 'k-')
axes[0].set_ylabel('Elevation [m]')
axes[0].grid(True)
# axes[1] = axes[1]
axes[1].plot(time / T, vF0 / 1e6, label='Standard')
axes[1].plot(time / T, vF / 1e6, 'k-', label='Wheeler Correction')
axes[1].set_ylabel('Streamwise Load, Cumulative [MN]')
axes[1].legend()
axes[1].grid(True)
# axes[2] = axes[2]
axes[2].plot(time / T, vM0 / 1e6, label='Standard')
axes[2].plot(time / T, vM / 1e6, 'k-', label='Wheeler Correction')
axes[2].set_ylabel('Sea-Bed Moment [MNm]')
axes[2].set_xlabel('Dimensionless Time, t/T [-]')
axes[2].legend()
axes[2].grid(True)

# fig.savefig('IntegratedPileLoads.png')
# fig.savefig('IntegratedPileLoads.webp')
fig.savefig('IntegratedPileLoads.png')
# fig.show()

# now save csv of the velocity, acceleration, force, and moment
veta_df = pd.DataFrame()
u_df = pd.DataFrame()
du_df = pd.DataFrame()
for i in range(nz):
    dof = 1
    name = 'Disp_' + str(i + 1) + '_' + str(dof)
    veta_df[name] = elevation[:, i]
    name = 'Vel_' + str(i + 1) + '_' + str(dof)
    u_df[name] = velocity[:, i]
    name = 'RMSA_' + str(i + 1) + '_' + str(dof)
    du_df[name] = accel[:, i]

    # transpose the dataframe so one recorder occupies a row, not a column (which are timesteps)
    # veta_df = veta_df.T
    # u_df = u_df.T
    # du_df = du_df.T

# add column per each force recorder
result_df = pd.DataFrame()
for i in range(nz):
    dof = 1
    # name = 'Node_' + str(i+1) + '_' + str(dof)
    name = 'Force_' + str(i + 1) + '_' + str(dof)
    result_df[name] = force[:, i]
    # transpose the dataframe
    # result_df = result_df.T

    # make sure there are no headers or indices


# write columns to columns in csv files
(veta_df.T).to_csv('disp.evt', sep=' ', encoding='utf-8', index=False, header=False)
(u_df.T).to_csv('vel.evt', sep=' ', encoding='utf-8', index=False, header=False)
(du_df.T).to_csv('accel.evt', sep=' ', encoding='utf-8', index=False, header=False)
(result_df.T).to_csv(
    'forces.evt', sep=' ', encoding='utf-8', index=False, header=False
)


# write columns to columns in csv files
(veta_df.T).to_csv('disp.out', sep=' ', encoding='utf-8', index=False, header=False)
(u_df.T).to_csv('vel.out', sep=' ', encoding='utf-8', index=False, header=False)
(du_df.T).to_csv('accel.out', sep=' ', encoding='utf-8', index=False, header=False)
(result_df.T).to_csv(
    'forces.out', sep=' ', encoding='utf-8', index=False, header=False
)
(result_df.T).to_csv(
    'node.out', sep=' ', encoding='utf-8', index=False, header=False
)

# make results.out dataframe with 3 columns and one row, no header. Each element is separated by a space

# results_df = pd.DataFrame({'total_impulse':vF[-1], 'max_force':vM[-1], 'total_disp':vF0[-1]}, index=[0])
# results_df.to_csv('results.out', sep=' ', encoding='utf-8', header=False, index=False)


def main(df=None):
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute inline/total hydrodynamic force and moments on a monopile using Morisons equation'
    )

    parser.add_argument(
        '-hw', '--water_depth', type=float, default=30.0, help='Water depth [m]'
    )
    parser.add_argument(
        '-Tp', '--peak_period', type=float, default=12.7, help='Wave period [s]'
    )
    parser.add_argument(
        '-Hs',
        '--significant_wave_height',
        type=float,
        default=5.0,
        help='Significant wave height [m]',
    )
    parser.add_argument(
        '-Dp',
        '--pile_diameter',
        type=float,
        default=1.0,
        help='Monopile diameter [m]',
    )
    parser.add_argument(
        '-Cd', '--drag_coefficient', type=float, default=2.1, help='Drag coefficient'
    )
    parser.add_argument(
        '-Cm', '--mass_coefficient', type=float, default=2.0, help='Mass coefficient'
    )
    parser.add_argument(
        '-nz',
        '--number_of_recorders_z',
        type=int,
        default=4,
        help='Number of points used in the z direction to compute loads',
    )
    parser.add_argument('-t', '--time', type=float, default=1.0, help='Time [s]')

    arguments, unknowns = parser.parse_known_args()

    # hw = arguments.water_depth
    # Tp = arguments.peak_period
    # Hs = arguments.significant_wave_height
    # # D  = arguments.pile_diameter
    # # CD = arguments.drag_coefficient
    # # CM = arguments.mass_coefficient
    # nz = arguments.number_of_recorders_z
    # t  = arguments.time

    # # --- Derived parameters
    # h = hw
    # # T = Tp
    # f  = 1./T
    # g  = 9.81
    # k  = wavenumber(f, h, g)
    # a  = Hs
    # z_ref = -h # reference point for moment calculation
    # eps = 0   # phase shift [rad]
    # rho = 1000 # water density

    # --- Wave kinematics

    # fig.show()

    # --------------------------------------------------------------------------------}

    # plt.suptitle('Hydro - Morison loads on monopile')

    # plt.savefig('MorisonLoads.png')

    # plt.show()
    print('End of __main__ in Ex4_WaveLoads.py')
    main()


if __name__ == '__test__':
    pass

if __name__ == '__export__':
    # plt.close(fig1)
    # plt.close(fig2)
    # plt.close(fig)
    from welib.tools.repo import export_figs_callback

    export_figs_callback(__file__)
