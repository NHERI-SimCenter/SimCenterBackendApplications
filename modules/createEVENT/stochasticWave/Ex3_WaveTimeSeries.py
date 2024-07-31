#!/usr/bin/env python3  # noqa: EXE001

"""Generate wave time series based on the Jonswap spectrum"""  # noqa: D400, D415

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed, uniform

# Local
from welib.tools.figure import defaultRC

defaultRC()
from welib.hydro.morison import *  # noqa: E402, F403
from welib.hydro.spectra import jonswap  # noqa: E402
from welib.hydro.wavekin import *  # noqa: E402, F403
from welib.hydro.wavekin import elevation2d, wavenumber  # noqa: E402
from welib.tools.spectral import fft_wrap  # noqa: E402

# --- Random seed
seed(None)  # noqa: NPY002

# --- Parameters
# t  = np.linspace(0,600,601) # time vector  [s]  # noqa: ERA001
t = np.linspace(0, 60.0, 6001)  # time vector  [s]
Hs = 8.1  # Significant wave height [m]
Tp = 12.7  # Peak period [s]
h = 30.0  # Water depth [m]
g = 9.80665  # Gravity[m/s2]

# --- Jonswap spectrum
dt = t[1] - t[0]  # timestep [s]
df = 1 / np.max(t)  # step size for frequency  # noqa: PD901
fHighCut = 1 / (dt) / 2.0  # Highest frequency in calculations  # noqa: N816
freq = np.arange(df, fHighCut, df)
S = jonswap(freq, Hs, Tp=Tp, g=9.80665)

# --- Solve dispersion relation
k = wavenumber(freq, h, g)

# --- Compute wave elevation based on amplitudes and random phases
eps = uniform(0, 2 * np.pi, len(freq))  # random phases between 0 and 2pi  # noqa: NPY002
a = np.sqrt(2 * S * df)  # wave amplitudes based on spectrum
x = 0  # longitudinal distance where wave is evaluated [m]
eta = elevation2d(a, freq, k, eps, t, x)

# --- Compute FFT of wave elevation
f_fft, S_fft, Info = fft_wrap(t, eta, output_type='PSD', averaging='none')


# --- Plots
fig, axes = plt.subplots(2, 1, sharey=False, figsize=(6.4, 4.8))  # (6.4,4.8)
fig.subplots_adjust(
    left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.30, wspace=0.20
)

ax = axes[0]
ax.plot(t, eta)
ax.tick_params(direction='in')
ax.autoscale(enable=True, axis='both', tight=True)
ax.set_xlabel('Time [s]')
ax.set_ylabel(r'Wave elevation [m]')
ax.set_title('Wave Generation')

ax = axes[1]
ax.plot(f_fft, S_fft, '-', label='Generated')
ax.plot(freq, S, 'k', label='Jonswap')
ax.legend()
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel(r'Spectral density [m$^2$ s]')
ax.tick_params(direction='in')
ax.autoscale(enable=True, axis='both', tight=True)
fig.savefig('WaveTimeSeries.png')
# fig.savefig('WaveTimeSeries.webp')  # noqa: ERA001
# plt.show()  # noqa: ERA001

if __name__ == '__main__':
    pass
if __name__ == '__test__':
    pass
if __name__ == '__export__':
    from welib.tools.repo import export_figs_callback

    export_figs_callback(__file__)
