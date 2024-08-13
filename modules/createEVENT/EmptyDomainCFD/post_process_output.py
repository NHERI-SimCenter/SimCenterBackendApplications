# Copyright (c) 2016-2017, The Regents of the University of California (Regents).  # noqa: D100, INP001
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the FreeBSD Project.
#
# REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
# THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS
# PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
# UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

#
# Contributors:
# Abiy Melaku


#
# This script reads OpenFOAM output and plot the characteristics of the
# approaching wind. For now, it read and plots only velocity field data and
# pressure on predicted set of probes.
#

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal


def readPressureProbes(fileName):  # noqa: N802, N803
    """Created on Wed May 16 14:31:42 2018

    Reads pressure probe data from OpenFOAM and return the probe location, time, and the pressure
    for each time step.

    @author: Abiy
    """  # noqa: D400, D401
    probes = []
    p = []
    time = []

    with open(fileName) as f:  # noqa: PLW1514, PTH123
        for line in f:
            if line.startswith('#'):
                if line.startswith('# Probe'):
                    line = line.replace('(', '')  # noqa: PLW2901
                    line = line.replace(')', '')  # noqa: PLW2901
                    line = line.split()  # noqa: PLW2901
                    probes.append([float(line[3]), float(line[4]), float(line[5])])
                else:
                    continue
            else:
                line = line.split()  # noqa: PLW2901
                time.append(float(line[0]))
                p_probe_i = np.zeros([len(probes)])
                for i in range(len(probes)):
                    p_probe_i[i] = float(line[i + 1])
                p.append(p_probe_i)

    probes = np.asarray(probes, dtype=np.float32)
    time = np.asarray(time, dtype=np.float32)
    p = np.asarray(p, dtype=np.float32)

    return probes, time, p


def read_pressure_data(file_names):
    """This functions takes names of different OpenFOAM pressure measurements and connect
    them into one file removing overlaps if any. All the probes must be in the same
    location, otherwise an error might show up.

    Parameters
    ----------
    *args
        List of file pashes of pressure data to be connected together.

    Returns
    -------
    time, pressure
        Returns the pressure time and pressure data of the connected file.

    """  # noqa: D205, D401, D404
    no_files = len(file_names)
    connected_time = []  # Connected array of time
    connected_p = []  # connected array of pressure.

    time1 = []
    p1 = []
    time2 = []
    p2 = []
    probes = []

    for i in range(no_files):
        probes, time2, p2 = readPressureProbes(file_names[i])

        if i == 0:
            connected_time = time2
            connected_p = p2
        else:
            try:
                index = np.where(time2 > time1[-1])[0][0]
                # index += 1

            except:  # noqa: E722
                # sys.exit('Fatal Error!: the pressure files have time gap')
                index = 0  # Joint them even if they have a time gap

            connected_time = np.concatenate((connected_time, time2[index:]))
            connected_p = np.concatenate((connected_p, p2[index:]))

        time1 = time2
        p1 = p2  # noqa: F841
    return probes, connected_time, connected_p


class PressureData:
    """A class that holds a pressure data and performs the following operations:
    - mean and rms pressure coefficients
    - peak pressure coefficients
    """  # noqa: D205, D400

    def __init__(
        self,
        path,
        u_ref=0.0,
        rho=1.25,
        p_ref=0.0,
        start_time=None,
        end_time=None,
    ):
        self.path = path
        self.u_ref = u_ref
        self.p_ref = p_ref
        self.rho = rho
        self.start_time = start_time
        self.end_time = end_time
        self.__read_cfd_data()
        self.__set_time()
        self.Nt = len(self.time)
        self.T = self.time[-1]
        self.z = self.probes[:, 2]
        self.y = self.probes[:, 1]
        self.x = self.probes[:, 0]
        self.dt = np.mean(np.diff(self.time))
        self.probe_count = np.shape(self.probes)[0]

    def __read_cfd_data(self):
        if os.path.isdir(self.path):  # noqa: PTH112
            print('Reading from path : %s' % (self.path))  # noqa: T201, UP031
            time_names = os.listdir(self.path)
            sorted_index = np.argsort(np.float64(time_names)).tolist()
            # print(sorted_index)
            # print("\tTime directories: %s" %(time_names))
            file_names = []

            for i in range(len(sorted_index)):
                file_name = os.path.join(self.path, time_names[sorted_index[i]], 'p')  # noqa: PTH118
                file_names.append(file_name)

            # print(file_names)
            self.probes, self.time, self.p = read_pressure_data(file_names)
            self.p = self.rho * np.transpose(self.p)  # OpenFOAM gives p/rho

            # self.p = np.transpose(self.p) # OpenFOAM gives p/rho
        else:
            print('Cannot find the file path: %s' % (self.path))  # noqa: T201, UP031

    def __set_time(self):
        if self.start_time != None:  # noqa: E711
            start_index = int(np.argmax(self.time > self.start_time))
            self.time = self.time[start_index:]
            # self.cp = self.cp[:,start_index:]
            try:  # noqa: SIM105
                self.p = self.p[:, start_index:]
            except:  # noqa: S110, E722
                pass

        if self.end_time != None:  # noqa: E711
            end_index = int(np.argmax(self.time > self.end_time))
            self.time = self.time[:end_index]
            # self.cp = self.cp[:,:end_index]
            try:  # noqa: SIM105
                self.p = self.p[:, :end_index]
            except:  # noqa: S110, E722
                pass


def von_karman_spectrum(f, Uav, I, L, comp=0):  # noqa: N803, E741, D103
    psd = np.zeros(len(f))  # noqa: F841

    if comp == 0:
        return (
            4.0
            * np.power(I * Uav, 2.0)
            * (L / Uav)
            / np.power(1.0 + 70.8 * np.power(f * L / Uav, 2.0), 5.0 / 6.0)
        )

    if comp == 1 or comp == 2:  # noqa: RET503, PLR1714, PLR2004
        return (
            4.0
            * np.power(I * Uav, 2.0)
            * (L / Uav)
            * (1.0 + 188.4 * np.power(2.0 * f * L / Uav, 2.0))
            / np.power(1.0 + 70.8 * np.power(2.0 * f * L / Uav, 2.0), 11.0 / 6.0)
        )


def psd(x, dt, nseg):
    """Calculates the power spectral density of a given signal using the welch
    method.

    Parameters
    ----------
    x
        The time history of the signal.
    dt
        The time step .
    nseg
        The the number of segments to average the time series.

    Returns
    -------
    freq, spectra
        Returns the frequency and spectra of the signal

    """  # noqa: D205, D401
    x_no_mean = x - np.mean(x)
    freq, spectra = signal.welch(
        x_no_mean, fs=1.0 / dt, nperseg=len(x_no_mean) / nseg
    )

    return freq[1:], spectra[1:]


def write_open_foam_vector_field(p, file_name):
    """Writes a given vector-field (n x 3) array to OpenFOAM 'vectorField'
    format.

    """  # noqa: D205, D401
    f = open(file_name, 'w+')  # noqa: PLW1514, PTH123, SIM115
    f.write('%d' % len(p[:, 2]))
    f.write('\n(')
    for i in range(len(p[:, 2])):
        f.write(f'\n ({p[i, 0]:.7e} {p[i, 1]:.7e} {p[i, 2]:.7e})')

    f.write('\n);')
    f.close()


def read_openFoam_scalar_field(file_name):  # noqa: N802
    """Reads a given vectorField OpenFOAM into numpy (n x 3) array format."""  # noqa: D401
    sField = []  # noqa: N806

    with open(file_name) as f:  # noqa: PLW1514, PTH123
        itrf = iter(f)
        next(itrf)
        for line in itrf:
            if line.startswith('(') or line.startswith(')'):  # noqa: PIE810
                continue
            else:  # noqa: RET507
                line = line.split()  # noqa: PLW2901
                sField.append(float(line[0]))

    sField = np.asarray(sField, dtype=np.float32)  # noqa: N806

    return sField  # noqa: RET504


def read_openFoam_vector_field(file_name):  # noqa: N802
    """Reads a given vectorField OpenFOAM into numpy (n x 3) array format."""  # noqa: D401
    vField = []  # noqa: N806

    with open(file_name) as f:  # noqa: PLW1514, PTH123
        for line in f:
            if line.startswith('('):
                line = line.replace('(', '')  # noqa: PLW2901
                line = line.replace(')', '')  # noqa: PLW2901
                line = line.split()  # noqa: PLW2901

                if len(line) < 3:  # noqa: PLR2004
                    continue

                vField.append([float(line[0]), float(line[1]), float(line[2])])

    vField = np.asarray(vField, dtype=np.float32)  # noqa: N806

    return vField  # noqa: RET504


def read_openFoam_tensor_field(file_name):  # noqa: N802
    """Reads a given vectorField OpenFOAM into numpy (n x 3) array format."""  # noqa: D401
    vField = []  # noqa: N806

    row_count = 9

    with open(file_name) as f:  # noqa: PLW1514, PTH123
        for line in f:
            if line.startswith('('):
                line = line.replace('(', '')  # noqa: PLW2901
                line = line.replace(')', '')  # noqa: PLW2901
                line = line.split()  # noqa: PLW2901

                if len(line) < row_count:
                    continue

                row = np.zeros(row_count)

                for i in range(row_count):
                    row[i] = float(line[i])

                vField.append(row)

    vField = np.asarray(vField, dtype=np.float32)  # noqa: N806

    return vField  # noqa: RET504


def read_openFoam_symmetric_tensor_field(file_name):  # noqa: N802
    """Reads a given vectorField OpenFOAM into numpy (n x 3) array format."""  # noqa: D401
    vField = []  # noqa: N806

    row_count = 6

    with open(file_name) as f:  # noqa: PLW1514, PTH123
        for line in f:
            if line.startswith('('):
                line = line.replace('(', '')  # noqa: PLW2901
                line = line.replace(')', '')  # noqa: PLW2901
                line = line.split()  # noqa: PLW2901

                if len(line) < row_count:
                    continue

                row = np.zeros(row_count)
                for i in range(row_count):
                    row[i] = float(line[i])

                vField.append(row)

    vField = np.asarray(vField, dtype=np.float32)  # noqa: N806

    return vField  # noqa: RET504


def read_velocity_data(path):
    """This functions takes names of different OpenFOAM velocity measurements and connect
    them into one file removing overlaps if any. All the probes must be in the same
    location, otherwise an error might showup.

    Parameters
    ----------
    *args
        List of file paths of velocity data to be connected together.

    Returns
    -------
    time, pressure
        Returns the velocity time and velocity data of the connected file.

    """  # noqa: D205, D401, D404
    num_files = len(path)
    connected_time = []  # Connected array of time
    connected_U = []  # connected array of pressure.  # noqa: N806

    time1 = []
    U1 = []  # noqa: N806
    time2 = []
    U2 = []  # noqa: N806
    probes = []

    for i in range(num_files):
        probes, time2, U2 = read_velocity_probes(path[i])  # noqa: N806
        if i != 0:
            try:
                index = np.where(time2 > time1[-1])[0][0]
            except:  # noqa: E722
                # sys.exit('Fatal Error!: the pressure files have time gap')
                index = 0  # Join them even if they have a time gap
            connected_time = np.concatenate((connected_time, time2[index:]))
            connected_U = np.concatenate((connected_U, U2[index:]))  # noqa: N806
        else:
            connected_time = time2
            connected_U = U2  # noqa: N806

        time1 = time2
        U1 = U2  # noqa: N806, F841
    shape = np.shape(connected_U)
    U = np.zeros((shape[1], shape[2], shape[0]))  # noqa: N806

    for i in range(shape[1]):
        for j in range(shape[2]):
            U[i, j, :] = connected_U[:, i, j]
    return probes, connected_time, U


def read_velocity_probes(fileName):  # noqa: N803
    """Created on Wed May 16 14:31:42 2018

    Reads velocity probe data from OpenFOAM and return the probe location, time,
    and the velocity vector for each time step.
    """  # noqa: D400, D401
    probes = []
    U = []  # noqa: N806
    time = []

    with open(fileName) as f:  # noqa: PLW1514, PTH123
        for line in f:
            if line.startswith('#'):
                if line.startswith('# Probe'):
                    line = line.replace('(', '')  # noqa: PLW2901
                    line = line.replace(')', '')  # noqa: PLW2901
                    line = line.split()  # noqa: PLW2901
                    probes.append([float(line[3]), float(line[4]), float(line[5])])
                else:
                    continue
            else:
                line = line.replace('(', '')  # noqa: PLW2901
                line = line.replace(')', '')  # noqa: PLW2901
                line = line.split()  # noqa: PLW2901
                try:
                    time.append(float(line[0]))
                except:  # noqa: S112, E722
                    continue
                u_probe_i = np.zeros([len(probes), 3])
                for i in range(len(probes)):
                    u_probe_i[i, :] = [
                        float(line[3 * i + 1]),
                        float(line[3 * i + 2]),
                        float(line[3 * i + 3]),
                    ]
                U.append(u_probe_i)

    probes = np.asarray(probes, dtype=np.float32)
    time = np.asarray(time, dtype=np.float32)
    U = np.asarray(U, dtype=np.float32)  # noqa: N806

    return probes, time, U


def calculate_length_scale(u, uav, dt, min_corr=0.0):
    """Calculates the length scale of a velocity time history given."""  # noqa: D401
    u = u - np.mean(u)  # noqa: PLR6104

    corr = signal.correlate(u, u, mode='full')

    u_std = np.std(u)

    corr = corr[int(len(corr) / 2) :] / (u_std**2 * len(u))

    loc = np.argmax(corr < min_corr)

    corr = corr[:loc]

    L = uav * np.trapz(corr, dx=dt)  # noqa: NPY201, N806

    return L  # noqa: RET504


def psd(x, dt, nseg):  # noqa: F811
    """Calculates the power spectral density of a given signal using the welch
    method.

    Parameters
    ----------
    x
        The time history of the signal.
    dt
        The time step .
    nseg
        The the number of segments to average the time series.

    Returns
    -------
    freq, spectra
        Returns the frequency and spectra of the signal

    """  # noqa: D205, D401
    x_no_mean = x - np.mean(x)
    freq, spectra = signal.welch(
        x_no_mean, fs=1.0 / dt, nperseg=len(x_no_mean) / nseg
    )

    return freq[1:], spectra[1:]


class VelocityData:
    """A class that holds a velocity data and performs the following operations:
    - mean velocity profile
    - turbulence intensity profiles
    - integral scale of turbulence profiles
    """  # noqa: D205, D400

    def __init__(
        self,
        path,
        sampling_rate=400,
        filter_data=False,  # noqa: FBT002
        filter_freq=400,
        start_time=None,
        end_time=None,
        resample_dt=None,
    ):
        self.path = path
        self.sampling_rate = sampling_rate
        self.filter_data = filter_data
        self.filter_freq = filter_freq
        self.start_time = start_time
        self.end_time = end_time
        self.component_count = 3
        self.resample_dt = resample_dt
        self.__read_cfd_data()
        self.__set_time()
        self.Nt = len(self.time)
        self.T = self.time[-1]
        self.dt = np.mean(np.diff(self.time))
        self.f_max = 1.0 / (2.0 * self.dt)
        self.probe_count = np.shape(self.probes)[0]
        self.Np = self.probe_count
        self.z = self.probes[:, 2]
        self.y = self.probes[:, 1]
        self.x = self.probes[:, 0]
        self.__filter_signal()
        self.__calculate_all()

    def __read_cfd_data(self):
        if os.path.isdir(self.path):  # noqa: PTH112
            print('Reading from path : %s' % (self.path))  # noqa: T201, UP031
            time_names = os.listdir(self.path)
            sorted_index = np.argsort(np.float64(time_names)).tolist()
            file_names = []

            for i in range(len(sorted_index)):
                file_name = os.path.join(self.path, time_names[sorted_index[i]], 'U')  # noqa: PTH118
                file_names.append(file_name)

            self.probes, self.time, self.U = read_velocity_data(file_names)

            # Distance along the path of the profile

            n_points = np.shape(self.probes)[0]
            self.dist = np.zeros(n_points)

            for i in range(n_points - 1):
                self.dist[i + 1] = self.dist[i] + np.linalg.norm(
                    self.probes[i + 1, :] - self.probes[i, :]
                )

            # Coefficient of variation
            cv = np.std(np.diff(self.time)) / np.mean(np.diff(self.time))

            if cv > 1.0e-4:  # noqa: PLR2004
                self.__adjust_time_step()

        else:
            print('Cannot find the file path: %s' % (self.path))  # noqa: T201, UP031

    def __adjust_time_step(self):
        if self.resample_dt == None:  # noqa: E711
            dt = np.mean(np.diff(self.time))
        else:
            dt = self.resample_dt

        time = np.arange(start=self.time[0], stop=self.time[-1], step=dt)

        shape = np.shape(self.U)

        U = np.zeros((shape[0], shape[1], len(time)))  # noqa: N806

        for i in range(shape[0]):
            for j in range(shape[1]):
                U[i, j, :] = np.interp(time, self.time, self.U[i, j, :])

        self.time = time
        self.U = U

    def __filter_signal(self):
        if self.filter_data:
            low_pass = signal.butter(
                10, self.filter_freq, 'lowpass', fs=self.sampling_rate, output='sos'
            )
            for i in range(self.probe_count):
                for j in range(self.component_count):
                    self.U[i, j, :] = signal.sosfilt(low_pass, self.U[i, j, :])

    def __set_time(self):
        if self.start_time != None:  # noqa: E711
            start_index = int(np.argmax(self.time > self.start_time))
            self.time = self.time[start_index:]
            self.U = self.U[:, :, start_index:]

        if self.end_time != None:  # noqa: E711
            end_index = int(np.argmax(self.time > self.end_time))
            self.time = self.time[:end_index]
            self.U = self.U[:, :, :end_index]

    def __calculate_all(self):
        self.u = np.zeros((self.probe_count, self.component_count, self.Nt))

        # Calculate the mean velocity profile.

        self.Uav = np.mean(self.U[:, 0, :], axis=1)

        # Calculate the turbulence intensity.
        self.I = np.std(self.U, axis=2)  # gets the standard deviation
        self.Ru = np.var(self.U[:, 0, :], axis=1)  # gets reynolds stress
        self.Rv = np.var(self.U[:, 1, :], axis=1)  # gets reynolds stress
        self.Rw = np.var(self.U[:, 2, :], axis=1)  # gets reynolds stress

        for i in range(self.component_count):
            self.I[:, i] = self.I[:, i] / self.Uav  # noqa: PLR6104

        # Calculate the length scale profiles.
        self.L = np.zeros((self.probe_count, self.component_count))
        for i in range(self.probe_count):
            for j in range(self.component_count):
                self.u[i, j, :] = self.U[i, j, :] - np.mean(self.U[i, j, :])
                self.L[i, j] = calculate_length_scale(
                    self.u[i, j, :], self.Uav[i], self.dt, 0.05
                )

        # Calculate the shear stress profiles.
        self.uv_bar = np.zeros(self.Np)
        self.uw_bar = np.zeros(self.Np)

        for i in range(self.Np):
            self.uv_bar[i] = np.cov(self.U[i, 0, :], self.U[i, 1, :])[0, 1]
            self.uw_bar[i] = np.cov(self.U[i, 0, :], self.U[i, 2, :])[0, 1]

    def get_Uav(self, z):  # noqa: N802, D102
        from scipy import interpolate  # noqa: PLC0415

        f = interpolate.interp1d(self.z, self.Uav)

        return f(z)


def copy_vtk_planes_and_order(input_path, output_path, field):
    """This code reads VTK sample plane data from OpenFOAM case directory and
    copies them into other directory with all vtks files ordered in their
    respective time sequence in one directory.

    input_path: path of the vtk files in the postProcessing directory
    ouput_path: path to write the vtk files in order

    """  # noqa: D205, D401, D404
    if not os.path.isdir(input_path):  # noqa: PTH112
        print(f'Cannot find the path for: {input_path}')  # noqa: T201
        return

    if not os.path.isdir(output_path):  # noqa: PTH112
        print(f'Cannot find the path for: {output_path}')  # noqa: T201
        return

    print(f'Reading from path: {input_path}')  # noqa: T201
    time_names = os.listdir(input_path)
    times = np.float64(time_names)
    sorted_index = np.argsort(times).tolist()

    n_times = len(times)

    print(f'\tNumber of time directories: {n_times} ')  # noqa: T201
    print(f'\tTime step: {np.mean(np.diff(times)):.4f} s')  # noqa: T201
    print(  # noqa: T201
        f'\tTotal duration: {times[sorted_index[-1]] - times[sorted_index[0]]:.4f} s'
    )

    for i in range(n_times):
        index = sorted_index[i]
        pathi = os.path.join(input_path, time_names[index])  # noqa: PTH118
        os.listdir(pathi)

        new_name = f'{field}_T{i + 1:04d}.vtk'
        for f in os.listdir(pathi):
            if f.endswith('.vtk'):
                new_path = os.path.join(output_path, new_name)  # noqa: PTH118
                old_path = os.path.join(pathi, f)  # noqa: PTH118
                shutil.copyfile(old_path, new_path)
                print(f'Copied path: {old_path}')  # noqa: T201


def plot_wind_profiles_and_spectra(case_path, output_path, prof_name):  # noqa: D103
    # Read JSON data
    json_path = os.path.join(  # noqa: PTH118
        case_path, 'constant', 'simCenter', 'input', 'EmptyDomainCFD.json'
    )
    with open(json_path) as json_file:  # noqa: PLW1514, PTH123
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    wc_data = json_data['windCharacteristics']

    ref_h = wc_data['referenceHeight']

    prof_path = os.path.join(case_path, 'postProcessing', prof_name)  # noqa: PTH118

    prof = VelocityData(prof_path, start_time=None, end_time=None)

    # Create wind profile data profile z, Uav, Iu ..., Lu ...,
    prof_np = np.zeros((len(prof.z), 9))
    prof_np[:, 0] = prof.z
    prof_np[:, 1] = prof.Uav
    prof_np[:, 2] = prof.I[:, 0]
    prof_np[:, 3] = prof.I[:, 1]
    prof_np[:, 4] = prof.I[:, 2]
    prof_np[:, 5] = prof.uw_bar
    prof_np[:, 6] = prof.L[:, 0]
    prof_np[:, 7] = prof.L[:, 1]
    prof_np[:, 8] = prof.L[:, 2]

    # Read the target wind profile data
    tar_path = os.path.join(case_path, 'constant', 'boundaryData', 'inlet')  # noqa: PTH118

    tar_p = read_openFoam_vector_field(os.path.join(tar_path, 'points'))  # noqa: PTH118
    tar_U = read_openFoam_scalar_field(os.path.join(tar_path, 'U'))  # noqa: PTH118, N806
    tar_R = read_openFoam_symmetric_tensor_field(os.path.join(tar_path, 'R'))  # noqa: PTH118, N806
    tar_L = read_openFoam_tensor_field(os.path.join(tar_path, 'L'))  # noqa: PTH118, N806

    tar_U_ref = np.interp(ref_h, tar_p[:, 2], tar_U)  # noqa: N806, F841

    tar_Iu = np.sqrt(tar_R[:, 0]) / tar_U  # noqa: N806
    tar_Iv = np.sqrt(tar_R[:, 3]) / tar_U  # noqa: N806
    tar_Iw = np.sqrt(tar_R[:, 5]) / tar_U  # noqa: N806
    tar_uw = tar_R[:, 2]

    tar_Lu = tar_L[:, 0]  # noqa: N806
    tar_Lv = tar_L[:, 3]  # noqa: N806
    tar_Lw = tar_L[:, 6]  # noqa: N806

    tar_I = np.zeros((3, len(tar_Iu)))  # noqa: N806
    tar_L = np.zeros((3, len(tar_Lu)))  # noqa: N806

    tar_I[0, :] = tar_Iu
    tar_I[1, :] = tar_Iv
    tar_I[2, :] = tar_Iw

    tar_L[0, :] = tar_Lu
    tar_L[1, :] = tar_Lv
    tar_L[2, :] = tar_Lw

    subplot_titles = (
        'Mean Velocity',
        'Turbulence Intensity, Iu',
        'Turbulence Intensity, Iv',
        'Turbulence Intensity, Iw',
        'Shear Stress',
        'Length Scale, Lu',
        'Length Scale, Lv',
        'Length Scale, Lw',
    )

    fig = make_subplots(
        rows=2,
        cols=4,
        start_cell='top-left',
        subplot_titles=subplot_titles,
        vertical_spacing=0.15,
    )

    fig.add_trace(
        go.Scatter(
            x=tar_U,
            y=tar_p[:, 2],
            line=dict(color='black', width=3.0, dash='dot'),  # noqa: C408
            mode='lines',
            name='Target',
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=prof_np[:, 1],
            y=prof_np[:, 0],
            line=dict(color='firebrick', width=2.5),  # noqa: C408
            mode='lines+markers',
            name=prof_name,
        ),
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text='$U_{av} [m/s]$',
        range=[0, 1.25 * np.max(prof_np[:, 1])],
        showline=True,
        linewidth=1.5,
        linecolor='black',
        ticks='outside',
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text='$z [m]$',
        range=[0, 1.01 * np.max(prof_np[:, 0])],
        showline=True,
        linewidth=1.5,
        linecolor='black',
        ticks='outside',
        row=1,
        col=1,
    )

    # Turbulence Intensity Iu
    fig.add_trace(
        go.Scatter(
            x=tar_Iu,
            y=tar_p[:, 2],
            line=dict(color='black', width=3.0, dash='dot'),  # noqa: C408
            mode='lines',
            name='Target',
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=prof_np[:, 2],
            y=prof_np[:, 0],
            line=dict(color='firebrick', width=2.5),  # noqa: C408
            mode='lines+markers',
            name=prof_name,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(
        title_text='$I_{u}$',
        range=[0, 1.3 * np.max(prof_np[:, 2])],
        showline=True,
        linewidth=1.5,
        linecolor='black',
        ticks='outside',
        row=1,
        col=2,
    )
    fig.update_yaxes(
        title_text='',
        range=[0, 1.01 * np.max(prof_np[:, 0])],
        showline=True,
        linewidth=1.5,
        linecolor='black',
        ticks='outside',
        row=1,
        col=2,
    )

    # Turbulence Intensity Iv
    fig.add_trace(
        go.Scatter(
            x=tar_Iw,
            y=tar_p[:, 2],
            line=dict(color='black', width=3.0, dash='dot'),  # noqa: C408
            mode='lines',
            name='Target',
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=prof_np[:, 3],
            y=prof_np[:, 0],
            line=dict(color='firebrick', width=2.5),  # noqa: C408
            mode='lines+markers',
            name=prof_name,
        ),
        row=1,
        col=3,
    )
    fig.update_xaxes(
        title_text='$I_{v}$',
        range=[0, 1.3 * np.max(prof_np[:, 3])],
        showline=True,
        linewidth=1.5,
        linecolor='black',
        ticks='outside',
        row=1,
        col=3,
    )
    fig.update_yaxes(
        title_text='',
        range=[0, 1.01 * np.max(prof_np[:, 0])],
        showline=True,
        linewidth=1.5,
        linecolor='black',
        ticks='outside',
        row=1,
        col=3,
    )

    # Turbulence Intensity Iw
    fig.add_trace(
        go.Scatter(
            x=tar_Iw,
            y=tar_p[:, 2],
            line=dict(color='black', width=3.0, dash='dot'),  # noqa: C408
            mode='lines',
            name='Target',
        ),
        row=1,
        col=4,
    )
    fig.add_trace(
        go.Scatter(
            x=prof_np[:, 4],
            y=prof_np[:, 0],
            line=dict(color='firebrick', width=2.5),  # noqa: C408
            mode='lines+markers',
            name=prof_name,
        ),
        row=1,
        col=4,
    )
    fig.update_xaxes(
        title_text='$I_{w}$',
        range=[0, 1.3 * np.max(prof_np[:, 4])],
        showline=True,
        linewidth=1.5,
        linecolor='black',
        ticks='outside',
        row=1,
        col=4,
    )
    fig.update_yaxes(
        title_text='',
        range=[0, 1.01 * np.max(prof_np[:, 0])],
        showline=True,
        linewidth=1.5,
        linecolor='black',
        ticks='outside',
        row=1,
        col=4,
    )

    # Shear Stress Profile
    fig.add_trace(
        go.Scatter(
            x=tar_uw,
            y=tar_p[:, 2],
            line=dict(color='black', width=3.0, dash='dot'),  # noqa: C408
            mode='lines',
            name='Target',
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=prof_np[:, 5],
            y=prof_np[:, 0],
            line=dict(color='firebrick', width=2.5),  # noqa: C408
            mode='lines+markers',
            name=prof_name,
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(
        title_text=r'$\overline{uw}$',
        range=[1.3 * np.min(prof_np[:, 5]), 1.5 * np.max(prof_np[:, 5])],
        showline=True,
        linewidth=1.5,
        linecolor='black',
        ticks='outside',
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text='$z [m]$',
        range=[0, 1.01 * np.max(prof_np[:, 0])],
        showline=True,
        linewidth=1.5,
        linecolor='black',
        ticks='outside',
        row=2,
        col=1,
    )

    # Length scale Lu
    fig.add_trace(
        go.Scatter(
            x=tar_Lu,
            y=tar_p[:, 2],
            line=dict(color='black', width=3.0, dash='dot'),  # noqa: C408
            mode='lines',
            name='Target',
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=prof_np[:, 6],
            y=prof_np[:, 0],
            line=dict(color='firebrick', width=2.5),  # noqa: C408
            mode='lines+markers',
            name=prof_name,
        ),
        row=2,
        col=2,
    )
    fig.update_xaxes(
        title_text='$L_{u} [m]$',
        range=[0, 1.5 * np.max(prof_np[:, 6])],
        showline=True,
        linewidth=1.5,
        linecolor='black',
        ticks='outside',
        row=2,
        col=2,
    )
    fig.update_yaxes(
        title_text='',
        range=[0, 1.01 * np.max(prof_np[:, 0])],
        showline=True,
        linewidth=1.5,
        linecolor='black',
        ticks='outside',
        row=2,
        col=2,
    )

    # Length scale Lv
    fig.add_trace(
        go.Scatter(
            x=tar_Lv,
            y=tar_p[:, 2],
            line=dict(color='black', width=3.0, dash='dot'),  # noqa: C408
            mode='lines',
            name='Target',
        ),
        row=2,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=prof_np[:, 7],
            y=prof_np[:, 0],
            line=dict(color='firebrick', width=2.5),  # noqa: C408
            mode='lines+markers',
            name=prof_name,
        ),
        row=2,
        col=3,
    )
    fig.update_xaxes(
        title_text='$L_{v} [m]$',
        range=[0, 1.5 * np.max(prof_np[:, 7])],
        showline=True,
        linewidth=1.5,
        linecolor='black',
        ticks='outside',
        row=2,
        col=3,
    )
    fig.update_yaxes(
        title_text='',
        range=[0, 1.01 * np.max(prof_np[:, 0])],
        showline=True,
        linewidth=1.5,
        linecolor='black',
        ticks='outside',
        row=2,
        col=3,
    )

    # Length scale Lw
    fig.add_trace(
        go.Scatter(
            x=tar_Lw,
            y=tar_p[:, 2],
            line=dict(color='black', width=3.0, dash='dot'),  # noqa: C408
            mode='lines',
            name='Target',
        ),
        row=2,
        col=4,
    )
    fig.add_trace(
        go.Scatter(
            x=prof_np[:, 8],
            y=prof_np[:, 0],
            line=dict(color='firebrick', width=2.5),  # noqa: C408
            mode='lines+markers',
            name=prof_name,
        ),
        row=2,
        col=4,
    )
    fig.update_xaxes(
        title_text='$L_{w} [m]$',
        range=[0, 1.5 * np.max(prof_np[:, 8])],
        showline=True,
        linewidth=1.5,
        linecolor='black',
        ticks='outside',
        row=2,
        col=4,
    )
    fig.update_yaxes(
        title_text='',
        range=[0, 1.01 * np.max(prof_np[:, 0])],
        showline=True,
        linewidth=1.5,
        linecolor='black',
        ticks='outside',
        row=2,
        col=4,
    )

    fig.update_layout(height=850, width=1200, title_text='', showlegend=False)
    fig.show()
    fig.write_html(
        os.path.join(output_path, prof_name + '.html'),  # noqa: PTH118
        include_mathjax='cdn',
    )

    # Plot the spectra at four locations

    spec_h = ref_h * np.array([0.25, 0.50, 1.00, 2.00])

    n_spec = len(spec_h)
    nseg = 5
    ncomp = 3
    ylabel = [
        r'$fS_{u}/\sigma^2_{u}$',
        r'$fS_{v}/\sigma^2_{v}$',
        r'$fS_{w}/\sigma^2_{w}$',
    ]

    for i in range(n_spec):
        loc = np.argmin(np.abs(prof_np[:, 0] - spec_h[i]))

        loc_tar = np.argmin(np.abs(tar_p[:, 2] - spec_h[i]))

        subplot_titles = ('u-component', 'v-component', 'w-component')
        fig = make_subplots(
            rows=1,
            cols=3,
            start_cell='top-left',
            subplot_titles=subplot_titles,
            vertical_spacing=0.15,
        )

        U_ref_prof = np.interp(spec_h[i], prof_np[:, 0], prof_np[:, 1])  # noqa: N806
        U_ref_tar = np.interp(spec_h[i], tar_p[:, 2], tar_U)  # noqa: N806

        # Plot each component
        for j in range(ncomp):
            freq, spec = psd(prof.u[loc, j, :], prof.dt, nseg)

            f_min = np.min(freq) / 1.5
            f_max = 1.5 * np.max(freq)

            u_var = np.var(prof.u[loc, j, :])

            spec = freq * spec / u_var
            freq = freq * spec_h[i] / U_ref_prof

            tar_Iz = tar_I[j, loc_tar]  # noqa: N806
            tar_Lz = tar_L[j, loc_tar]  # noqa: N806

            vonk_f = np.logspace(np.log10(f_min), np.log10(f_max), 200)
            vonk_psd = von_karman_spectrum(vonk_f, U_ref_tar, tar_Iz, tar_Lz, j)

            vonk_psd = vonk_f * vonk_psd / np.square(U_ref_tar * tar_Iz)
            vonk_f = vonk_f * spec_h[i] / U_ref_tar

            fig.add_trace(
                go.Scatter(
                    x=freq,
                    y=spec,
                    line=dict(color='firebrick', width=1.5),  # noqa: C408
                    mode='lines',
                    name=prof_name,
                ),
                row=1,
                col=1 + j,
            )
            fig.add_trace(
                go.Scatter(
                    x=vonk_f,
                    y=vonk_psd,
                    line=dict(color='black', width=3.0, dash='dot'),  # noqa: C408
                    mode='lines',
                    name='Target(von Karman)',
                ),
                row=1,
                col=1 + j,
            )
            fig.update_xaxes(
                type='log',
                title_text='$fz/U$',
                showline=True,
                linewidth=1.5,
                linecolor='black',
                ticks='outside',
                row=1,
                col=1 + j,
            )
            fig.update_yaxes(
                type='log',
                title_text=ylabel[j],
                showline=True,
                linewidth=1.5,
                linecolor='black',
                ticks='outside',
                row=1,
                col=1 + j,
            )

        fig.update_layout(height=450, width=1500, title_text='', showlegend=False)
        fig.show()
        fig.write_html(
            os.path.join(  # noqa: PTH118
                output_path, 'spectra_' + prof_name + '_H' + str(1 + i) + '.html'
            ),
            include_mathjax='cdn',
        )


def plot_pressure_profile(case_path, output_path, prof_name):  # noqa: D103
    prof_path = os.path.join(case_path, 'postProcessing', prof_name)  # noqa: PTH118

    prof = PressureData(
        prof_path, start_time=1.0, end_time=None, u_ref=0.0, rho=1.25, p_ref=0.0
    )

    std_p = np.std(prof.p, axis=1)

    subplot_titles = ('Pressure Fluctuation',)

    fig = make_subplots(
        rows=1,
        cols=1,
        start_cell='top-left',
        subplot_titles=subplot_titles,
        vertical_spacing=0.15,
    )

    # Plot pressure fluctuation Velocity
    fig.add_trace(
        go.Scatter(
            x=prof.x - np.min(prof.x),
            y=std_p,
            line=dict(color='firebrick', width=2.5),  # noqa: C408
            mode='lines+markers',
            name=prof_name,
        ),
        row=1,
        col=1,
    )

    fig.update_xaxes(
        title_text='Distance from inlet (x) [m]',
        range=[np.min(prof.x - np.min(prof.x)), np.max(prof.x - np.min(prof.x))],
        showline=True,
        linewidth=1.5,
        linecolor='black',
        ticks='outside',
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text=r'Pressure R.M.S',
        range=[0, 1.15 * np.max(std_p)],
        showline=True,
        linewidth=1.5,
        linecolor='black',
        ticks='outside',
        row=1,
        col=1,
    )

    fig.update_layout(height=400, width=800, title_text='', showlegend=False)
    fig.show()
    fig.write_html(
        os.path.join(output_path, 'pressure_' + prof_name + '.html'),  # noqa: PTH118
        include_mathjax='cdn',
    )


if __name__ == '__main__':
    """"
    Entry point to read the simulation results from OpenFOAM case and post-process it.
    """

    # CLI parser
    parser = argparse.ArgumentParser(
        description='Get EVENT file from OpenFOAM output'
    )
    parser.add_argument(
        '-c', '--case', help='OpenFOAM case directory', required=True
    )

    arguments, unknowns = parser.parse_known_args()

    case_path = arguments.case

    print('Case full path: ', case_path)  # noqa: T201

    # prof_name = sys.argv[2]

    # Read JSON data
    json_path = os.path.join(  # noqa: PTH118
        case_path, 'constant', 'simCenter', 'input', 'EmptyDomainCFD.json'
    )
    with open(json_path) as json_file:  # noqa: PLW1514, PTH123
        json_data = json.load(json_file)

    # Returns JSON object as a dictionary
    rm_data = json_data['resultMonitoring']

    wind_profiles = rm_data['windProfiles']
    vtk_planes = rm_data['vtkPlanes']

    prof_output_path = os.path.join(  # noqa: PTH118
        case_path, 'constant', 'simCenter', 'output', 'windProfiles'
    )

    # Check if it exists and remove files
    if os.path.exists(prof_output_path):  # noqa: PTH110
        shutil.rmtree(prof_output_path)

    # Create new path
    Path(prof_output_path).mkdir(parents=True, exist_ok=True)

    # Plot velocity and pressure profiles
    for prof in wind_profiles:
        name = prof['name']
        field = prof['field']
        print(name)  # noqa: T201
        print(field)  # noqa: T201

        if field == 'Velocity':
            plot_wind_profiles_and_spectra(case_path, prof_output_path, name)

        if field == 'Pressure':
            plot_pressure_profile(case_path, prof_output_path, name)

    # Copy the VTK files renamed
    for pln in vtk_planes:
        name = pln['name']
        field = pln['field']

        vtk_path = os.path.join(case_path, 'postProcessing', name)  # noqa: PTH118
        vtk_path_renamed = os.path.join(  # noqa: PTH118
            case_path, 'postProcessing', name + '_renamed'
        )

        Path(vtk_path_renamed).mkdir(parents=True, exist_ok=True)

        copy_vtk_planes_and_order(vtk_path, vtk_path_renamed, field)

        # Check if it exists and remove files
        if os.path.exists(vtk_path):  # noqa: PTH110
            shutil.rmtree(vtk_path)
