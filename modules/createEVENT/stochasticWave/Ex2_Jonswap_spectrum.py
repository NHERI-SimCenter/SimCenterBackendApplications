#!/usr/bin/env python3  # noqa: CPY001, EXE001

"""Plot the JONSWAP spectrum for a given sea state"""  # noqa: D400

import matplotlib.pyplot as plt
import numpy as np

# Local
from welib.tools.figure import defaultRC

defaultRC()
from welib.hydro.morison import *  # noqa: E402, F403
from welib.hydro.spectra import jonswap  # noqa: E402
from welib.hydro.wavekin import *  # noqa: E402, F403

# --- Parameters
t = np.arange(0, 3600.1, 1)  # time vector  [s]
dt = t[1] - t[0]  # timestep [s]
Hs = 8.1  # Significant wave height [m]
Tp = 12.7  # Peak period [s]

# --- Derived parameters
df = 1.0 / np.max(t)  # Step size for frequency  # noqa: PD901
fMax = (1.0 / dt) / 2  # Highest frequency  # noqa: N816
freq = np.arange(df, fMax + df / 2, df)

# --- Spectrum and amplitude
S = jonswap(freq, Hs, Tp)  # Spectral density [m^2.s]
ap = np.sqrt(2 * S * df)  # Wave amplitude [m]

# Find location of maximum energy
iMax = np.argmax(S)  # noqa: N816

# --- Plots
fig, ax = plt.subplots(1, 1, sharey=False, figsize=(6.4, 4.8))  # (6.4,4.8)
fig.subplots_adjust(
    left=0.12, right=0.95, top=0.95, bottom=0.11, hspace=0.20, wspace=0.20
)
ax.plot(freq, S)
ax.plot(freq[iMax], S[iMax], 'ko')
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel(r'Spectral density [m^2 s]')
ax.set_title('Hydro - Jonswap spectrum')
ax.tick_params(direction='in')
# fig.savefig('JonswapSpectrum.png')
# fig.savefig('JonswapSpectrum.webp')

# plt.show()

if __name__ == '__main__':
    pass
if __name__ == '__test__':
    np.testing.assert_almost_equal(S[iMax], 113.8770176)
if __name__ == '__export__':
    from welib.tools.repo import export_figs_callback

    export_figs_callback(__file__)
