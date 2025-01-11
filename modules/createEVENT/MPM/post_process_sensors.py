#!/usr/bin/env python3  # noqa: EXE001, D100

# -*- coding: utf-8 -*-
# Copyright (c) 2016-2017, The Regents of the University of California (Regents).
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
# Justin Bonus

"""This script reads HydroUQ MPM output from sensors and then plots the data."""  # noqa: D404

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


def main():
    """Main function for post-processing sensor data from MPM simulations."""  # noqa: D401
    return 0


if __name__ == '__main__':
    """Post-process sensor data from MPM simulations and save the plots."""
    # CLI parser
    input_args = sys.argv[1:]
    print(  # noqa: T201
        'post_process_sensors.py - Backend-script post_process_sensors.py reached main. Starting...'
    )
    print(  # noqa: T201
        'post_process_sensors.py - Backend-script post_process_sensors.py running: '
        + str(sys.argv[0])
    )
    print(  # noqa: T201
        'post_process_sensors.py - Backend-script post_process_sensors.py received input args: '
        + str(input_args)
    )

    # parser = argparse.ArgumentParser(description="Get sensor measurements from output, process them, plot them, and then save the figures.")
    # parser.add_argument('-i', '--input_directory', help="Sensor Measurement Input Directory", required=True)
    # parser.add_argument('-o', '--output_directory', help="Sensor Plot Output Directory", required=True)
    # parser.add_argument('-f', '--files', help="Sensor Measurement Files", required=True)
    # arguments, unknowns = parser.parse_known_args()

    sensor_data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    sensor_files = sys.argv[3].split(',')

    print('Raw Sensor Input Directory: ', sensor_data_dir)  # noqa: T201
    print('Processed Sensor Output Directory: ', output_dir)  # noqa: T201
    print('Raw Sensor Filenames: ', sensor_files)  # noqa: T201

    # Get the list of sensor names
    sensor_names = [
        (sensor_file.split('.')[0]).lstrip('/').strip()
        for sensor_file in sensor_files
    ]

    # Load the sensor data
    sensor_data = {}
    for sensor_file in sensor_files:
        # Remove any leading '/' from the sensor file
        sensor_file = sensor_file.lstrip('/')  # noqa: PLW2901
        # Remove whitespace from the sensor file
        sensor_file = sensor_file.strip()  # noqa: PLW2901
        sensor_file = sensor_file.split(  # noqa: PLW2901
            '.'
        )  # Split the sensor file by the '.' character
        if sensor_file[-1] != 'csv':
            print(  # noqa: T201
                'Error: Sensor file is not a csv file. Please provide a csv file. Will skip this file: '
                + sensor_file[0]
                + '.'
                + sensor_file[-1]
            )
            continue
        sensor_file = sensor_file[  # noqa: PLW2901
            0
        ]  # Get the first part of the sensor file, which is the sensor name
        sensor_data[sensor_file] = pd.read_csv(
            os.path.join(sensor_data_dir, sensor_file + '.csv'),  # noqa: PTH118
            header=None,
            skiprows=1,
            delimiter=',',
            usecols=[0, 1],
        )

        # Assume that the header is row 0, and that the time is in the first column, and the value is in the second column
        sensor_data[sensor_file].columns = ['time', 'value']

        please_convert_to_date_time = False  # May want to use this later, as wave-flumes tend to report time in date-time formats
        if (
            please_convert_to_date_time == True  # noqa: E712
            and sensor_data[sensor_file]['time'].dtype != 'datetime64[ns]'
        ):
            sensor_data[sensor_file]['time'] = pd.to_datetime(
                sensor_data[sensor_file]['time']
            )

    # Make sure the output directory exists, and save the sensor raw data to the output directory if they aren't already there
    if not os.path.exists(output_dir):  # noqa: PTH110
        print(  # noqa: T201
            'Output directory not found. Creating output directory: '
            + output_dir
            + '.'
        )
        os.makedirs(output_dir)  # noqa: PTH103
    if output_dir != sensor_data_dir:
        for sensor_name in sensor_names:
            print(  # noqa: T201
                'Save '
                + os.path.join(output_dir, sensor_name)  # noqa: PTH118
                + '.csv to output directory.'
            )
            sensor_data[sensor_name].to_csv(
                os.path.join(output_dir, sensor_name + '.csv'),  # noqa: PTH118
                index=False,
            )

    # Plot the sensor data, and save the plots to the output directory (html and png files)
    for sensor_name in sensor_names:
        sensor_name_png = sensor_name + '.png'
        sensor_name_html = sensor_name + '.html'
        fig, axes = plt.subplots(1, 1)
        axes.plot(
            sensor_data[sensor_name]['time'], sensor_data[sensor_name]['value']
        )
        axes.set_title(sensor_name)
        axes.set_xlabel('Time [s]')
        axes.set_ylabel('Sensor Measurement')
        # Save the plot as a png file
        plt.savefig(
            os.path.join(output_dir, sensor_name_png),  # noqa: PTH118
            dpi=300,
            bbox_inches='tight',
        )
        plt.close(fig)

        figure_as_html = (
            "<img src='" + os.path.join(output_dir, sensor_name_png) + "'>"  # noqa: PTH118
        )

        # Save the plot as an html file
        with open(os.path.join(output_dir, sensor_name_html), 'w') as f:  # noqa: PTH118, PTH123
            f.write(figure_as_html)

    sys.exit(main())
