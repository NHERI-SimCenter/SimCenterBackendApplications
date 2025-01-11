#!/usr/bin/env python3  # noqa: EXE001, D100

"""Compute inline/total hydrodynamic force and moments on a monopile using Morison's equation"""  # noqa: D400

import argparse
import json
import os
from fractions import Fraction

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import seed, uniform
from welib.tools.figure import defaultRC

defaultRC()

from welib.hydro.morison import *  # noqa: E402, F403
from welib.hydro.morison import inline_load  # noqa: E402
from welib.hydro.spectra import jonswap  # noqa: E402
from welib.hydro.wavekin import *  # noqa: E402, F403
from welib.hydro.wavekin import elevation2d, kinematics2d, wavenumber  # noqa: E402
from welib.tools.colors import python_colors  # noqa: E402
from welib.tools.spectral import fft_wrap  # noqa: E402

"""
Plot the wave kinematics (elevation, velocity, acceleration) for linear waves
Different locations, times and superposition of frequencies can be used.
Plot the JONSWAP spectrum for a given sea state
Generate wave time series based on the Jonswap spectrum
Compute inline/total hydrodynamic force and moments on a monopile using Morison's equation
"""


def ReadAIM(filenameAIM, getRV):  # noqa: C901, N802, N803, D103, PLR0915
    with open(filenameAIM, encoding='utf-8') as BIMFile:  # noqa: PTH123, N806
        bim = json.load(BIMFile)
        for event_id in range(1):
            if getRV is True:
                print(  # noqa: T201
                    'Running a sample simulation with default input parameters due to getRV flag'
                )
                g = 9.80665  # gravity [m/s^2]
                rho = 1000  # water density

                hw = 30.0  # water depth [ft]
                hc = 1.0
                hs = 2.0
                ht = 3.0

                Hs = 8.1  # noqa: N806
                Tp = 12.7  # noqa: N806

                D = 6  # monopile diameter [ft]  # noqa: N806
                # D = bim['Events'][i]['pileDiameter']
                AD = 1.0  # noqa: N806
                CD = 1.0  # noqa: N806
                CM = 2.0  # noqa: N806
                # CM = bim['Events'][i]['massCoefficient']

                # D = bim['Events'][i]['pileDiameter']
                nz = int(bim['Events'][event_id]['recorderCountZ'])
                x = 0.0
                duration = 360.0
                dt = 1.0
                randomSeed = 1  # noqa: N806

            else:
                print('Running a sample simulation with JSON input parameters')  # noqa: T201
                g = 9.80665  # gravity [m/s^2]
                rho = 1000.0  # water density [kg/m^3]

                hw = float(bim['Events'][event_id]['waterDepth'])
                hc = float(bim['Events'][event_id]['climateChangeSLR'])
                hs = float(bim['Events'][event_id]['stormSurgeSLR'])
                ht = float(bim['Events'][event_id]['tidalSLR'])

                Hs = float(bim['Events'][event_id]['significantWaveHeight'])  # noqa: N806
                Tp = float(bim['Events'][event_id]['peakPeriod'])  # noqa: N806

                D = 6.0  # monopile diameter [m]  # noqa: N806
                # D = bim['Events'][i]['pileDiameter']
                AD = float(bim['Events'][event_id]['dragArea'])  # noqa: N806, F841
                CD = float(bim['Events'][event_id]['dragCoefficient'])  # noqa: N806
                CM = 2.0  # noqa: N806
                # CM = bim['Events'][i]['massCoefficient']

                # D = bim['Events'][i]['pileDiameter']
                nz = int(bim['Events'][event_id]['recorderCountZ'])
                x = float(bim['Events'][event_id]['recorderOriginX'])

                duration = float(bim['Events'][event_id]['timeDuration'])
                dt = float(bim['Events'][event_id]['timeStep'])

                if (
                    bim['Events'][event_id].get('seed') is None
                    or bim['Events'][event_id]['seed'] == 'None'
                ):
                    randomSeed = 1  # noqa: N806
                else:
                    randomSeed = int(bim['Events'][event_id]['seed'])  # noqa: N806

            seed(randomSeed)

            h = hw + hc + hs + ht  # water depth [ft]
            z_ref = -h  # reference point for moment calculation, i.e. seabed [ft]

            time = np.arange(0, duration + dt, dt)  # time vector [s]
            df = 1 / np.max(time)  # step size for frequency  # noqa: PD901
            fHighCut = (  # noqa: N806
                1 / (dt) / 2.0
            )  # Highest frequency in calculations
            freq = np.arange(df, fHighCut, df)
            # --- Solve dispersion relation
            k = wavenumber(freq, h, g)

            # --- Plots
            fig, axes = plt.subplots(
                2, 1, sharey=False, figsize=(6.4, 4.8)
            )  # (6.4,4.8)
            fig.subplots_adjust(
                left=0.12,
                right=0.95,
                top=0.95,
                bottom=0.11,
                hspace=0.30,
                wspace=0.20,
            )

            # --- Jonswap spectrum
            Tp_list = [Tp]  # noqa: N806
            for Tp_i in Tp_list:  # noqa: N806
                Hs_list = [Hs]  # noqa: N806
                for Hs_i in Hs_list:  # noqa: N806
                    S = jonswap(freq, Hs_i, Tp=Tp_i, g=g)  # noqa: N806

                    seeds = [randomSeed]
                    for si in seeds:
                        seed(si)
                        eps = uniform(
                            0, 2 * np.pi, len(freq)
                        )  # random phases between 0 and 2pi
                        a = np.sqrt(2 * S * df)  # wave amplitudes based on spectrum
                        # --- Compute wave elevation based on amplitudes and random phases
                        eta = elevation2d(a, freq, k, eps, time, x=x)

                        # --- Compute FFT of wave elevation
                        f_fft, S_fft, Info = fft_wrap(  # noqa: N806
                            time, eta, output_type='PSD', averaging='none'
                        )

                        ax = axes[0]
                        ax.plot(time, eta)
                        ax.tick_params(direction='in')
                        ax.autoscale(enable=True, axis='both', tight=True)
                        ax.set_xlabel('Time [s]')
                        ax.set_ylabel(r'Wave elevation [m]')
                        ax.set_title('Hydro - wave generation')

                        ax = axes[1]
                        ax.plot(
                            f_fft,
                            S_fft,
                            '-',
                            label='Sampled: Hs=' + str(Hs_i) + ' Tp=' + str(Tp_i),
                        )
                        ax.plot(
                            freq,
                            S,
                            'k',
                            label='JONSWAP: Hs=' + str(Hs_i) + ' Tp=' + str(Tp_i),
                        )
                        ax.legend()
                        ax.set_xlabel('Frequency [Hz]')
                        ax.set_ylabel(r'Spectral density [m$^2$ s]')
                        ax.tick_params(direction='in')
                        ax.autoscale(enable=True, axis='both', tight=True)

            fig.savefig('StochasticWaveLoads_JONSWAP.png')

            vF = np.zeros(time.shape)  # noqa: N806
            vM = np.zeros(time.shape)  # noqa: N806
            vF0 = np.zeros(time.shape)  # noqa: N806, F841
            vM0 = np.zeros(time.shape)  # noqa: N806, F841

            elevation = np.zeros((len(time), nz))
            velocity = np.zeros((len(time), nz))
            accel = np.zeros((len(time), nz))
            force = np.zeros((len(time), nz))

            # veta = elevation2d(a, freq, k, eps, time, x=x)
            VFs = np.zeros((len(time), nz))  # noqa: N806
            VMs = np.zeros((len(time), nz))  # noqa: N806, F841
            for it, t in enumerate(time):
                # Wave kinematics
                # veta[it] = elevation2d(a, freq, k, eps, t, x=x)
                z = np.linspace(-h, eta[it], nz)
                u, du = kinematics2d(
                    a, freq, k, eps, h, t, z, Wheeler=True, eta=eta[it]
                )
                u0, du0 = kinematics2d(a, freq, k, eps, h, t, z)
                # Wave loads with Wheeler
                p_tot = inline_load(u, du, D, CD, CM, rho)
                vF[it] = np.trapz(p_tot, z)  # [N]  # noqa: NPY201
                vM[it] = np.trapz(p_tot * (z - z_ref), z)  # [Nm]  # noqa: NPY201

                for zi in range(nz):
                    VFs[it, zi] = np.trapz(p_tot[: (zi + 1)], z[: (zi + 1)])  # noqa: NPY201

                # Wave loads without Wheeler
                # p_tot0 = inline_load(u0, du0, D, CD, CM, rho)
                # vF0[it] = np.trapz(p_tot0, z)  # [N]
                # vM0[it] = np.trapz(p_tot0 * (z - z_ref), z)  # [Nm]

                elevation[it, :] = z.copy()
                velocity[it, :] = u.copy()
                accel[it, :] = du.copy()
                force[it, :] = p_tot.copy()

            if getRV == False or getRV == True:  # noqa: E712, PLR1714
                # Plot
                fig, axes = plt.subplots(
                    3, 1, sharex=True, figsize=(6.4, 4.8)
                )  # (6.4,4.8)
                fig.subplots_adjust(
                    left=0.12,
                    right=0.95,
                    top=0.95,
                    bottom=0.11,
                    hspace=0.14,
                    wspace=0.20,
                )
                # axes[0] = axes[0]
                axes[0].plot(time, eta, 'k-')
                axes[0].set_ylabel('Wave Elevation [m]')
                axes[0].grid(True)  # noqa: FBT003

                for zi in range(nz):
                    axes[1].plot(time, VFs[:, zi] / 1e3, label='Recorder ' + str(zi))
                # axes[1].plot(time, vFs[:, zi] / 1e3, 'k-', label='Wheeler Correction')
                axes[1].set_ylabel('Streamwise Load, Cumulative [kN]')
                axes[1].legend()
                axes[1].grid(True)  # noqa: FBT003

                # axes[2].plot(time, vM0 / 1e6, label='Standard')
                # for zi in range(nz):
                # axes[2].plot(time, VMs[:, zi] / 1e6, label='Standard')
                axes[2].plot(time, vM / 1e6, 'k-', label='Wheeler Correction')
                axes[2].set_ylabel('Sea-Bed Moment [MNm]')
                axes[2].set_xlabel('Time [s]')
                axes[2].legend()
                axes[2].grid(True)  # noqa: FBT003

                fig.savefig('StochasticWaveLoads_IntegratedLoads.png')

                # Plot elevation, velocity, accleleration, and force for each recorder
                fig, axes = plt.subplots(
                    4, 1, sharex=True, figsize=(6.4, 4.8)
                )  # (6.4,4.8)
                fig.subplots_adjust(
                    left=0.12,
                    right=0.95,
                    top=0.95,
                    bottom=0.11,
                    hspace=0.14,
                    wspace=0.20,
                )
                for zi in range(nz):
                    axes[0].plot(time, elevation, label='Recorder ' + str(zi))
                axes[0].set_ylabel('Recorder Elevation [m]')
                axes[0].grid(True)  # noqa: FBT003
                for zi in range(nz):
                    axes[1].plot(
                        time, force[:, zi] / 1e3, label='Recorder ' + str(zi)
                    )

                axes[1].set_ylabel('Streamwise Load [kN]')
                axes[1].legend()

                for zi in range(nz):
                    axes[2].plot(time, velocity[:, zi], label='Recorder ' + str(zi))
                axes[2].plot(time, velocity)
                axes[2].set_ylabel('Velocity [m/s]')
                axes[2].grid(True)  # noqa: FBT003

                for zi in range(nz):
                    axes[3].plot(time, accel[:, zi], label='Recorder ' + str(zi))
                axes[3].plot(time, accel)
                axes[3].set_ylabel('Acceleration [m/s^2]')
                axes[3].set_xlabel('Time [s]')
                fig.savefig('StochasticWaveLoads_Recorders.png')
            # plt.show()

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

            # add column per each force recorder
            result_df = pd.DataFrame()
            for i in range(nz):
                dof = 1
                # name = 'Node_' + str(i+1) + '_' + str(dof)
                name = 'Force_' + str(i + 1) + '_' + str(dof)
                result_df[name] = force[:, i]

            # write columns to columns in csv files, if file exists, overwrite
            if getRV == False or getRV == True:  # noqa: E712, PLR1714
                (veta_df.T).to_csv(
                    'disp.evt', sep=' ', encoding='utf-8', index=False, header=False
                )
                (u_df.T).to_csv(
                    'vel.evt', sep=' ', encoding='utf-8', index=False, header=False
                )
                (du_df.T).to_csv(
                    'accel.evt', sep=' ', encoding='utf-8', index=False, header=False
                )
                (result_df.T).to_csv(
                    'forces.evt',
                    sep=' ',
                    encoding='utf-8',
                    index=False,
                    header=False,
                )

                # # write columns to columns in csv files
                # (veta_df.T).to_csv('disp.out', sep=' ', encoding='utf-8', index=False, header=False)
                # (u_df.T).to_csv('vel.out', sep=' ', encoding='utf-8', index=False, header=False)
                # (du_df.T).to_csv('accel.out', sep=' ', encoding='utf-8', index=False, header=False)
                # (result_df.T).to_csv('forces.out', sep=' ', encoding='utf-8', index=False, header=False)
                # (result_df.T).to_csv('node.out', sep=' ', encoding='utf-8', index=False, header=False)

                # # make results.out dataframe with 3 columns and one row, no header. Each element is separated by a space
                # results_df = pd.DataFrame({'total_impulse': vF[-1], 'max_force': vM[-1], 'total_disp': vF0[-1]}, index=[0])
                # results_df.to_csv('results.out', sep=' ', encoding='utf-8', header=False, index=False)

            # else:

            # (veta_df.T).to_csv('disp.evt', sep=' ', encoding='utf-8', index=False, header=False)
            # (u_df.T).to_csv('vel.evt', sep=' ', encoding='utf-8', index=False, header=False)
            # (du_df.T).to_csv('accel.evt', sep=' ', encoding='utf-8', index=False, header=False)
            # (result_df.T).to_csv(
            #     'forces.evt', sep=' ', encoding='utf-8', index=False, header=False
            # )

            # # write columns to columns in csv files
            # (veta_df.T).to_csv('disp.out', sep=' ', encoding='utf-8', index=False, header=False)
            # (u_df.T).to_csv('vel.out', sep=' ', encoding='utf-8', index=False, header=False)
            # (du_df.T).to_csv('accel.out', sep=' ', encoding='utf-8', index=False, header=False)
            # (result_df.T).to_csv(
            #     'forces.out', sep=' ', encoding='utf-8', index=False, header=False
            # )
            # (result_df.T).to_csv(
            #     'node.out', sep=' ', encoding='utf-8', index=False, header=False
            # )


def main():
    """Main function."""  # noqa: D401
    parser = argparse.ArgumentParser(
        description='Compute inline/total hydrodynamic force and moments on a monopile using Morisons equation'
    )

    parser.add_argument(
        '-b',
        '--filenameAIM',
        help='BIM File',
        required=True,
        default='AIM.json',
    )
    parser.add_argument(
        '-e',
        '--filenameEVENT',
        help='Event File',
        required=True,
        default='EVENT.json',
    )
    parser.add_argument(
        '--getRV',
        help='getRV',
        required=False,
        action='store_true',
    )

    arguments, unknowns = parser.parse_known_args()

    filenameAIM = arguments.filenameAIM  # noqa: N806, F841
    filenameEVENT = arguments.filenameEVENT  # noqa: N806, F841
    getRV = arguments.getRV  # noqa: N806, F841

    ReadAIM(arguments.filenameAIM, arguments.getRV)

    return 0


if __name__ == '__main__':
    """Entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Compute inline/total hydrodynamic force and moments on a monopile using Morisons equation'
    )

    parser.add_argument(
        '-b',
        '--filenameAIM',
        help='BIM File',
        required=True,
        default='AIM.json',
    )
    parser.add_argument(
        '-e',
        '--filenameEVENT',
        help='Event File',
        required=True,
        default='EVENT.json',
    )
    parser.add_argument(
        '--getRV',
        help='getRV',
        required=False,
        action='store_true',
    )

    arguments, unknowns = parser.parse_known_args()

    filenameAIM = arguments.filenameAIM  # noqa: N816
    filenameEVENT = arguments.filenameEVENT  # noqa: N816
    getRV = arguments.getRV  # noqa: N816

    # ReadAIM(arguments.filenameAIM)

    main()


if __name__ == '__test__':
    pass

if __name__ == '__export__':
    from welib.tools.repo import export_figs_callback

    export_figs_callback(__file__)
