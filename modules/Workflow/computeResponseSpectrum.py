"""Simple Python Script to integrate a strong motion record using
the Newmark-Beta method
"""  # noqa: CPY001, D205, D400, INP001

import numpy as np
from scipy.constants import g
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d


def convert_accel_units(acceleration, from_, to_='cm/s/s'):  # noqa: C901
    """Converts acceleration from/to different units
    :param acceleration: the acceleration (numeric or numpy array)
    :param from_: unit of `acceleration`: string in "g", "m/s/s", "m/s**2",
        "m/s^2", "cm/s/s", "cm/s**2" or "cm/s^2"
    :param to_: new unit of `acceleration`: string in "g", "m/s/s", "m/s**2",
        "m/s^2", "cm/s/s", "cm/s**2" or "cm/s^2". When missing, it defaults
        to "cm/s/s"
    :return: acceleration converted to the given units (by default, 'cm/s/s')
    """  # noqa: D205, D400, D401
    m_sec_square = ('m/s/s', 'm/s**2', 'm/s^2')
    cm_sec_square = ('cm/s/s', 'cm/s**2', 'cm/s^2')
    acceleration = np.asarray(acceleration)
    if from_ == 'g':
        if to_ == 'g':
            return acceleration  # noqa: DOC201
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

    raise ValueError(  # noqa: DOC501, TRY003
        'Unrecognised time history units. '  # noqa: EM101
        "Should take either ''g'', ''m/s/s'' or ''cm/s/s''"
    )


def get_velocity_displacement(
    time_step,
    acceleration,
    units='cm/s/s',
    velocity=None,
    displacement=None,
):
    """Returns the velocity and displacement time series using simple integration
    :param float time_step:
        Time-series time-step (s)
    :param numpy.ndarray acceleration:
        Acceleration time-history
    :returns:
        velocity - Velocity Time series (cm/s)
        displacement - Displacement Time series (cm)
    """  # noqa: D205, D400, D401
    acceleration = convert_accel_units(acceleration, units)
    if velocity is None:
        velocity = time_step * cumtrapz(acceleration, initial=0.0)
    if displacement is None:
        displacement = time_step * cumtrapz(velocity, initial=0.0)
    return velocity, displacement  # noqa: DOC201


class NewmarkBeta:
    """Evaluates the response spectrum using the Newmark-Beta methodology"""  # noqa: D400

    def __init__(
        self,
        acceleration,
        time_step,
        periods,
        damping=0.05,
        dt_disc=0.002,
        units='g',
    ):
        """Setup the response spectrum calculator
        :param numpy.ndarray time_hist:
            Acceleration time history
        :param numpy.ndarray periods:
            Spectral periods (s) for calculation
        :param float damping:
            Fractional coefficient of damping
        :param float dt_disc:
            Sampling rate of the acceleration
        :param str units:
            Units of the acceleration time history {"g", "m/s", "cm/s/s"}
        """  # noqa: D205, D400, D401
        self.periods = periods
        self.num_per = len(periods)
        self.acceleration = convert_accel_units(acceleration, units)
        self.damping = damping
        self.d_t = time_step
        self.velocity, self.displacement = get_velocity_displacement(
            self.d_t, self.acceleration
        )
        self.num_steps = len(self.acceleration)
        self.omega = (2.0 * np.pi) / self.periods
        self.response_spectrum = None
        self.dt_disc = dt_disc

    def run(self):
        """Evaluates the response spectrum
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
        """  # noqa: D205, D400, D401
        omega = (2.0 * np.pi) / self.periods
        cval = self.damping * 2.0 * omega
        kval = ((2.0 * np.pi) / self.periods) ** 2.0
        # Perform Newmark - Beta integration
        accel, vel, disp, a_t = self._newmark_beta(omega, cval, kval)
        self.response_spectrum = {
            'Period': self.periods,
            'Acceleration': np.max(np.fabs(a_t), axis=0),
            'Velocity': np.max(np.fabs(vel), axis=0),
            'Displacement': np.max(np.fabs(disp), axis=0),
        }
        self.response_spectrum['Pseudo-Velocity'] = (
            omega * self.response_spectrum['Displacement']
        )
        self.response_spectrum['Pseudo-Acceleration'] = (
            (omega**2.0) * self.response_spectrum['Displacement'] / g / 100.0
        )
        time_series = {
            'Time-Step': self.d_t,
            'Acceleration': self.acceleration,
            'Velocity': self.velocity,
            'Displacement': self.displacement,
            'PGA': np.max(np.fabs(self.acceleration)) / g / 100.0,
            'PGV': np.max(np.fabs(self.velocity)),
            'PGD': np.max(np.fabs(self.displacement)),
        }
        return self.response_spectrum, time_series, accel, vel, disp  # noqa: DOC201

    def _newmark_beta(self, omega, cval, kval):  # noqa: ARG002
        """Newmark-beta integral
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
        """  # noqa: D205, D400
        # Parameters
        dt = self.d_t
        ground_acc = self.acceleration
        num_steps = self.num_steps
        dt_disc = self.dt_disc
        # discritize
        num_steps_disc = int(np.floor(num_steps * dt / dt_disc))
        f = interp1d(
            [dt * x for x in range(num_steps)],
            ground_acc,
            bounds_error=False,
            fill_value=(ground_acc[0], ground_acc[-1]),
        )
        tmp_time = [dt_disc * x for x in range(num_steps_disc)]
        ground_acc = f(tmp_time)
        # Pre-allocate arrays
        accel = np.zeros([num_steps_disc, self.num_per], dtype=float)
        vel = np.zeros([num_steps_disc, self.num_per], dtype=float)
        disp = np.zeros([num_steps_disc, self.num_per], dtype=float)
        a_t = np.zeros([num_steps_disc, self.num_per], dtype=float)
        # Initial line
        accel[0, :] = (-ground_acc[0] - (cval * vel[0, :])) - (kval * disp[0, :])
        for j in range(1, num_steps_disc):
            delta_acc = ground_acc[j] - ground_acc[j - 1]
            delta_d2u = (
                -delta_acc
                - dt_disc * cval * accel[j - 1, :]
                - dt_disc * kval * (vel[j - 1, :] + 0.5 * dt_disc * accel[j - 1, :])
            ) / (1.0 + 0.5 * dt_disc * cval + 0.25 * dt_disc**2 * kval)
            delta_du = dt_disc * accel[j - 1, :] + 0.5 * dt_disc * delta_d2u
            delta_u = (
                dt_disc * vel[j - 1, :]
                + 0.5 * dt_disc**2 * accel[j - 1, :]
                + 0.25 * dt_disc**2 * delta_d2u
            )
            accel[j, :] = delta_d2u + accel[j - 1, :]
            vel[j, :] = delta_du + vel[j - 1, :]
            disp[j, :] = delta_u + disp[j - 1, :]
            a_t[j, :] = ground_acc[j] + accel[j, :]

        return accel, vel, disp, a_t  # noqa: DOC201
