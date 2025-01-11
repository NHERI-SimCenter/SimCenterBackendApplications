#!/usr/bin/env python3

"""
Plot the JONSWAP spectrum for a given sea state and generate wave time series based on the Jonswap spectrum
"""  # noqa: D200, D400

import argparse
import sys
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from welib.tools.figure import defaultRC

defaultRC()
import matplotlib as mpl  # noqa: E402
from numpy.random import seed, uniform  # noqa: E402
from welib.hydro.morison import *  # noqa: E402, F403
from welib.hydro.morison import inline_load  # noqa: E402
from welib.hydro.spectra import jonswap  # noqa: E402
from welib.hydro.wavekin import *  # noqa: E402, F403
from welib.hydro.wavekin import elevation2d, kinematics2d, wavenumber  # noqa: E402
from welib.tools.colors import python_colors  # noqa: E402
from welib.tools.spectral import fft_wrap  # noqa: E402

if __name__ == '__main__':
    """Entry point for the script."""

    input_args = sys.argv[1:]

    print(  # noqa: T201
        'Jonswap.py - Backend-script post_process_sensors.py running: '
        + str(sys.argv[0])
    )
    print(  # noqa: T201
        'post_process_sensors.py - Backend-script post_process_sensors.py received input args: '
        + str(input_args)
    )

    inputFile = sys.argv[1]  # noqa: N816
    outputPath = sys.argv[2]  # noqa: N816

    import json
    import os

    with open(inputFile, encoding='utf-8') as BIMFile:  # noqa: PTH123
        bim = json.load(BIMFile)
        for event_id in range(1):
            print('Running a sample simulation with JSON input parameters')  # noqa: T201
            g = 9.80665  # gravity [m/s^2]
            rho = 1000.0  # water density

            hw = float(bim['Events'][event_id]['waterDepth'])
            hc = float(bim['Events'][event_id]['climateChangeSLR'])
            hs = float(bim['Events'][event_id]['stormSurgeSLR'])
            ht = float(bim['Events'][event_id]['tidalSLR'])

            Hs = float(bim['Events'][event_id]['significantWaveHeight'])
            Tp = float(bim['Events'][event_id]['peakPeriod'])

            D = 6.0  # monopile diameter [m]
            # D = bim['Events'][i]['pileDiameter']
            AD = float(bim['Events'][event_id]['dragArea'])
            CD = float(bim['Events'][event_id]['dragCoefficient'])
            CM = 2.0
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
                randomSeed = 1  # noqa: N816
            else:
                randomSeed = int(bim['Events'][event_id]['seed'])  # noqa: N816

            seed(randomSeed)

            h = hw + hc + hs + ht
            z_ref = -h  # reference point for moment calculation
            time = np.arange(0, duration + dt, dt)

            df = 1 / np.max(time)  # step size for frequency  # noqa: PD901
            fHighCut = (  # noqa: N816
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
            # dt       = t[1]-t[0]                   # timestep [s]
            Tp_list = [Tp]
            for Tp_i in Tp_list:
                Hs_list = [Hs]
                for Hs_i in Hs_list:
                    S = jonswap(freq, Hs_i, Tp=Tp_i, g=g)

                    seeds = [randomSeed + 2, randomSeed + 1, randomSeed]
                    for si in seeds:
                        seed(si)
                        eps = uniform(
                            0, 2 * np.pi, len(freq)
                        )  # random phases between 0 and 2pi
                        a = np.sqrt(2 * S * df)  # wave amplitudes based on spectrum
                        # --- Compute wave elevation based on amplitudes and random phases
                        eta = elevation2d(a, freq, k, eps, time, x=x)

                        # --- Compute FFT of wave elevation
                        f_fft, S_fft, Info = fft_wrap(
                            time, eta, output_type='PSD', averaging='none'
                        )

                        os.makedirs(outputPath, exist_ok=True)  # noqa: PTH103
                        os.chdir(outputPath)
                        # --- Plot
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
                        # if (getRV == False):
                        fig.savefig('StochasticWaveLoads_JONSWAP.png')

                        # if (si == seeds[-1]):

                        print('Saving wave spectra and time series')  # noqa: T201
                        spectra_df = pd.DataFrame()

                        spectra_df['Frequency'] = freq
                        spectra_df['SpectralDensity'] = S

                        generated_df = pd.DataFrame()
                        generated_df['Frequency'] = f_fft
                        generated_df['SpectralDensity'] = S_fft

                        time_df = pd.DataFrame()
                        time_df['Time'] = time
                        time_df['WaveElevation'] = eta

                        # Save csv files of the spectra and time series
                        np.savetxt('WaveSpectra.csv', spectra_df, delimiter=',')
                        np.savetxt(
                            'WaveSpectraGenerated.csv', generated_df, delimiter=','
                        )
                        np.savetxt('WaveTimeSeries.csv', time_df, delimiter=',')
