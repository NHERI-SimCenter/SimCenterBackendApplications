"""
Simple Python Script to integrate a strong motion record using
the Newmark-Beta method
"""

import numpy as np
from math import sqrt
from scipy.integrate import cumtrapz
from scipy.constants import g

def convert_accel_units(acceleration, from_, to_='cm/s/s'): 
    """
    Converts acceleration from/to different units
    :param acceleration: the acceleration (numeric or numpy array)
    :param from_: unit of `acceleration`: string in "g", "m/s/s", "m/s**2",
        "m/s^2", "cm/s/s", "cm/s**2" or "cm/s^2"
    :param to_: new unit of `acceleration`: string in "g", "m/s/s", "m/s**2",
        "m/s^2", "cm/s/s", "cm/s**2" or "cm/s^2". When missing, it defaults
        to "cm/s/s"
    :return: acceleration converted to the given units (by default, 'cm/s/s')
    """
    m_sec_square = ("m/s/s", "m/s**2", "m/s^2")
    cm_sec_square = ("cm/s/s", "cm/s**2", "cm/s^2")
    acceleration = np.asarray(acceleration)
    if from_ == 'g':
        if to_ == 'g':
            return acceleration
        if to_ in m_sec_square:
            return acceleration * g
        if to_ in cm_sec_square:
            return acceleration * (100 * g)
    elif from_ in m_sec_square:
        if to_ == 'g':
            return acceleration / g
        if to_ in m_sec_square:
            return acceleration
        if to_ in cm_sec_square:
            return acceleration * 100
    elif from_ in cm_sec_square:
        if to_ == 'g':
            return acceleration / (100 * g)
        if to_ in m_sec_square:
            return acceleration / 100
        if to_ in cm_sec_square:
            return acceleration

    raise ValueError("Unrecognised time history units. "
                     "Should take either ''g'', ''m/s/s'' or ''cm/s/s''")

def get_velocity_displacement(time_step, acceleration, units="cm/s/s",
                              velocity=None, displacement=None):
    """
    Returns the velocity and displacment time series using simple integration
    :param float time_step:
        Time-series time-step (s)
    :param numpy.ndarray acceleration:
        Acceleration time-history
    :returns:
        velocity - Velocity Time series (cm/s)
        displacement - Displacement Time series (cm)
    """
    acceleration = convert_accel_units(acceleration, units)
    if velocity is None:
        velocity = time_step * cumtrapz(acceleration, initial=0.)
    if displacement is None:
        displacement = time_step * cumtrapz(velocity, initial=0.)
    return velocity, displacement

class NewmarkBeta:
    """
    Evaluates the response spectrum using the Newmark-Beta methodology
    """

    def __init__(self, acceleration, time_step, periods, damping=0.05,
            units="g"):
        """
        Setup the response spectrum calculator
        :param numpy.ndarray time_hist:
            Acceleration time history
        :param numpy.ndarray periods:
            Spectral periods (s) for calculation
        :param float damping:
            Fractional coefficient of damping
        :param str units:
            Units of the acceleration time history {"g", "m/s", "cm/s/s"}
        """
        self.periods = periods
        self.num_per = len(periods)
        self.acceleration = convert_accel_units(acceleration, units)
        self.damping = damping
        self.d_t = time_step
        self.velocity, self.displacement = get_velocity_displacement(
            self.d_t, self.acceleration)
        self.num_steps = len(self.acceleration)
        self.omega = (2. * np.pi) / self.periods
        self.response_spectrum = None

    def run(self):
        """
        Evaluates the response spectrum
        :returns:
            Response Spectrum - Dictionary containing all response spectrum
                                data
                'Time' - Time (s)
                'Acceleration' - Acceleration Response Spectrum (cm/s/s)
                'Velocity' - Velocity Response Spectrum (cm/s)
                'Displacement' - Displacement Response Spectrum (cm)
                'Pseudo-Velocity' - Pseudo-Velocity Response Spectrum (cm/s)
                'Pseudo-Acceleration' - Pseudo-Acceleration Response Spectrum
                                       (cm/s/s)
            Time Series - Dictionary containing all time-series data
                'Time' - Time (s)
                'Acceleration' - Acceleration time series (cm/s/s)
                'Velocity' - Velocity time series (cm/s)
                'Displacement' - Displacement time series (cm)
                'PGA' - Peak ground acceleration (cm/s/s)
                'PGV' - Peak ground velocity (cm/s)
                'PGD' - Peak ground displacement (cm)
            accel - Acceleration response of Single Degree of Freedom Oscillator
            vel - Velocity response of Single Degree of Freedom Oscillator
            disp - Displacement response of Single Degree of Freedom Oscillator
        """
        omega = (2. * np.pi) / self.periods
        cval = self.damping * 2. * omega
        kval = ((2. * np.pi) / self.periods) ** 2.
        # Perform Newmark - Beta integration
        accel, vel, disp, a_t = self._newmark_beta(omega, cval, kval)
        self.response_spectrum = {
            'Period': self.periods,
            'Acceleration': np.max(np.fabs(a_t), axis=0),
            'Velocity': np.max(np.fabs(vel), axis=0),
            'Displacement': np.max(np.fabs(disp), axis=0)}
        self.response_spectrum['Pseudo-Velocity'] =  omega * \
            self.response_spectrum['Displacement']
        self.response_spectrum['Pseudo-Acceleration'] =  (omega ** 2.) * \
            self.response_spectrum['Displacement'] / g / 100.0
        time_series = {
            'Time-Step': self.d_t,
            'Acceleration': self.acceleration,
            'Velocity': self.velocity,
            'Displacement': self.displacement,
            'PGA': np.max(np.fabs(self.acceleration))/g/100.0,
            'PGV': np.max(np.fabs(self.velocity)),
            'PGD': np.max(np.fabs(self.displacement))}
        return self.response_spectrum, time_series, accel, vel, disp

    def _newmark_beta(self, omega, cval, kval):
        """
        Newmark-beta integral
        :param numpy.ndarray omega:
            Angular period - (2 * pi) / T
        :param numpy.ndarray cval:
            Damping * 2 * omega
        :param numpy.ndarray kval:
            ((2. * pi) / T) ** 2.
        :returns:
            accel - Acceleration time series
            vel - Velocity response of a SDOF oscillator
            disp - Displacement response of a SDOF oscillator
            a_t - Acceleration response of a SDOF oscillator
        """
        # Pre-allocate arrays
        accel = np.zeros([self.num_steps, self.num_per], dtype=float)
        vel = np.zeros([self.num_steps, self.num_per], dtype=float)
        disp = np.zeros([self.num_steps, self.num_per], dtype=float)
        a_t = np.zeros([self.num_steps, self.num_per], dtype=float)
        # Initial line
        accel[0, :] =(-self.acceleration[0] - (cval * vel[0, :])) - \
                      (kval * disp[0, :])
        a_t[0, :] = accel[0, :] + accel[0, :]
        for j in range(1, self.num_steps):
            disp[j, :] = disp[j-1, :] + (self.d_t * vel[j-1, :]) + \
                (((self.d_t ** 2.) / 2.) * accel[j-1, :])
                         
            accel[j, :] = (1./ (1. + self.d_t * 0.5 * cval)) * \
                (-self.acceleration[j] - kval * disp[j, :] - cval *
                (vel[j-1, :] + (self.d_t * 0.5) * accel[j-1, :]))
            vel[j, :] = vel[j - 1, :] + self.d_t * (0.5 * accel[j - 1, :] +
                0.5 * accel[j, :])
            a_t[j, :] = self.acceleration[j] + accel[j, :]
        return accel, vel, disp, a_t
