#  # noqa: INP001, D100
# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
#
# This file is part of the SimCenter Backend Applications
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# this file. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarnóczay
#

import warnings
from datetime import datetime


# Monkeypatch warnings to get prettier messages
def _warning(message, category, filename, lineno, file=None, line=None):  # noqa: ARG001
    if '\\' in filename:
        file_path = filename.split('\\')
    elif '/' in filename:
        file_path = filename.split('/')
    python_file = '/'.join(file_path[-3:])
    print(f'WARNING in {python_file} at line {lineno}\n{message}\n')  # noqa: T201


warnings.showwarning = _warning


def show_warning(warning_msg):  # noqa: D103
    warnings.warn(UserWarning(warning_msg))  # noqa: B028


def log_msg(msg='', prepend_timestamp=True):  # noqa: FBT002
    """Print a message to the screen with the current time as prefix

    The time is in ISO-8601 format, e.g. 2018-06-16T20:24:04Z

    Parameters
    ----------
    msg: string
       Message to print.

    """  # noqa: D400
    if prepend_timestamp:
        formatted_msg = '{} {}'.format(
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S:%fZ')[:-4],  # noqa: DTZ003
            msg,
        )
    else:
        formatted_msg = msg

    print(formatted_msg)  # noqa: T201

    if globals().get('log_file', None) is not None:
        with open(globals()['log_file'], 'a') as f:  # noqa: PLW1514, PTH123
            f.write('\n' + formatted_msg)


# Constants for unit conversion to standard units

unit_types = {
    'time': ['sec', 'minute', 'h', 'day'],
    'length': ['m', 'mm', 'cm', 'km', 'inch', 'ft', 'mile'],
    'area': ['m2', 'mm2', 'cm2', 'km2', 'inch2', 'ft2', 'mile2'],
    'volume': ['m3', 'mm3', 'cm3', 'km3', 'inch3', 'ft3', 'mile3'],
    'speed': ['cmps', 'mps', 'mph', 'inchps', 'ftps', 'kph', 'fps', 'kts'],
    'acceleration': ['mps2', 'cmps2', 'inchps2', 'ftps2', 'g'],
    'mass': ['kg', 'ton', 'lb'],
    'force': ['N', 'kN', 'lbf', 'kip', 'kips'],
    'pressure': ['Pa', 'kPa', 'MPa', 'GPa', 'psi', 'ksi', 'Mpsi'],
}

# time
sec = 1.0

minute = 60.0 * sec
h = 60.0 * minute
day = 24.0 * h

sec2 = sec**2.0

# distance, area, volume
m = 1.0

mm = 0.001 * m
cm = 0.01 * m
km = 1000.0 * m

inch = 0.0254
ft = 12.0 * inch
mile = 5280.0 * ft

# area
m2 = m**2.0

mm2 = mm**2.0
cm2 = cm**2.0
km2 = km**2.0

inch2 = inch**2.0
ft2 = ft**2.0
mile2 = mile**2.0

# volume
m3 = m**3.0

mm3 = mm**3.0
cm3 = cm**3.0
km3 = km**3.0

inch3 = inch**3.0
ft3 = ft**3.0
mile3 = mile**3.0

# speed / velocity
cmps = cm / sec
mps = m / sec
mph = mile / h

inchps = inch / sec
ftps = ft / sec
kph = km / h
fps = ft / sec
kts = 1.15078 * mph

# acceleration
mps2 = m / sec2
cmps2 = cm / sec2
inchps2 = inch / sec2
ftps2 = ft / sec2

g = 9.80665 * mps2

# mass
kg = 1.0

ton = 1000.0 * kg

lb = 0.453592 * kg

# force
N = kg * m / sec2

kN = 1e3 * N  # noqa: N816

lbf = lb * g
kip = 1000.0 * lbf
kips = kip

# pressure / stress
Pa = N / m2

kPa = 1e3 * Pa  # noqa: N816
MPa = 1e6 * Pa
GPa = 1e9 * Pa

psi = lbf / inch2
ksi = 1e3 * psi
Mpsi = 1e6 * psi

# KZ: unit bases decouple
unit_bases = {
    'm2': {'length': 'm'},
    'mm2': {'length': 'mm'},
    'cm2': {'length': 'cm'},
    'km2': {'length': 'km'},
    'inch2': {'length': 'in'},
    'ft2': {'length': 'ft'},
    'mile2': {'length': 'mile'},
    'm3': {'length': 'm'},
    'mm3': {'length': 'mm'},
    'cm3': {'length': 'cm'},
    'km3': {'length': 'km'},
    'inch3': {'length': 'in'},
    'ft3': {'length': 'ft'},
    'mile3': {'length': 'mile'},
    'cmps': {'length': 'cm', 'time': 'sec'},
    'mps': {'length': 'm', 'time': 'sec'},
    'mph': {'length': 'mile', 'time': 'h'},
    'inchps': {'length': 'inch', 'time': 'sec'},
    'ftps': {'length': 'ft', 'time': 'sec'},
    'mps2': {'length': 'm', 'time': 'sec'},
    'cmps2': {'length': 'cm', 'time': 'sec'},
    'inchps2': {'length': 'in', 'time': 'sec'},
    'ftps2': {'length': 'ft', 'time': 'sec'},
    'g': {},
}
unit_decoupling_type_list = ['TH_file']


def get_scale_factors(input_units, output_units):  # noqa: C901
    """Determine the scale factor to convert input event to internal event data"""  # noqa: D400
    # special case: if the input unit is not specified then do not do any scaling
    if input_units is None:
        scale_factors = {'ALL': 1.0}

    else:
        # parse output units:

        # if no length unit is specified, 'inch' is assumed
        unit_length = output_units.get('length', 'inch')
        if unit_length == 'in':
            unit_length = 'inch'
        f_length = globals().get(unit_length, None)
        if f_length is None:
            raise ValueError(f'Specified length unit not recognized: {unit_length}')  # noqa: DOC501, EM102, TRY003

        # if no time unit is specified, 'sec' is assumed
        unit_time = output_units.get('time', 'sec')
        f_time = globals().get(unit_time, None)
        if f_time is None:
            raise ValueError(f'Specified time unit not recognized: {unit_time}')  # noqa: DOC501, EM102, TRY003

        scale_factors = {}

        for input_name, input_unit in input_units.items():
            # exceptions
            if input_name == 'factor':
                f_scale = 1.0

            else:
                # get the scale factor to standard units
                if input_unit == 'in':
                    input_unit = 'inch'  # noqa: PLW2901

                f_in = globals().get(input_unit, None)
                if f_in is None:
                    raise ValueError(f'Input unit not recognized: {input_unit}')  # noqa: DOC501, EM102, TRY003

                unit_type = None
                for base_unit_type, unit_set in globals()['unit_types'].items():
                    if input_unit in unit_set:
                        unit_type = base_unit_type

                if unit_type is None:
                    raise ValueError(f'Failed to identify unit type: {input_unit}')  # noqa: DOC501, EM102, TRY003

                # the output unit depends on the unit type
                if unit_type == 'acceleration':
                    f_out = f_time**2.0 / f_length

                elif unit_type == 'speed':
                    f_out = f_time / f_length

                elif unit_type == 'length':
                    f_out = 1.0 / f_length

                else:
                    raise ValueError(  # noqa: DOC501, TRY003
                        f'Unexpected unit type in workflow: {unit_type}'  # noqa: EM102
                    )

                # the scale factor is the product of input and output scaling
                f_scale = f_in * f_out

            scale_factors.update({input_name: f_scale})

    return scale_factors


def get_unit_bases(input_units):
    """Decouple input units"""  # noqa: D400
    # special case: if the input unit is not specified then do nothing
    if input_units is None:  # noqa: PLR1702
        input_unit_bases = {}

    else:
        input_unit_bases = {}
        unit_bases_dict = globals()['unit_bases']
        for unit_type, input_unit in input_units.items():
            if unit_type in globals()['unit_decoupling_type_list']:
                cur_unit_bases = {'length': 'm', 'force': 'N', 'time': 'sec'}
                for unit_name, unit_bases in unit_bases_dict.items():
                    if unit_name == input_unit:
                        for x, y in unit_bases.items():
                            cur_unit_bases.update({x: y})
                        break
                input_unit_bases = cur_unit_bases
                break

    return input_unit_bases
