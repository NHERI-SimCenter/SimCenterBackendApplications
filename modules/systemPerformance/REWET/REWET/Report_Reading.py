# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 16:07:24 2022

@author: snaeimi
"""

import io
import datetime


def parseTimeStamp(time_stamp):
    striped_time_stamp = time_stamp.split(':')
    hour = striped_time_stamp[0]
    minute = striped_time_stamp[1]
    second = striped_time_stamp[2]

    hour = int(hour)
    minute = int(minute)
    second = int(second)

    return (hour, minute, minute)


class Report_Reading:
    def __init__(self, file_addr):
        self.file_data = {}
        self.maximum_trial_time = []
        with io.open(file_addr, 'r', encoding='utf-8') as f:
            lnum = 0
            for line in f:
                # self.file_data[lnum] = line
                if 'Maximum trials exceeded at' in line:
                    time_str = (
                        line.split('WARNING: Maximum trials exceeded at ')[1]
                        .split(' hrs')[0]
                        .split(',')[0]
                    )

                    x = parseTimeStamp(time_str)
                    time_sec = datetime.timedelta(
                        hours=x[0], minutes=x[1], seconds=x[2]
                    ).total_seconds()
                    time_sec = int(time_sec)
                    self.maximum_trial_time.append(time_sec)
                elif 'System unbalanced at' in line:
                    time_str = (
                        line.split('System unbalanced at ')[1]
                        .split(' hrs')[0]
                        .split(',')[0]
                    )
                    x = parseTimeStamp(time_str)
                    time_sec = datetime.timedelta(
                        hours=x[0], minutes=x[1], seconds=x[2]
                    ).total_seconds()
                    time_sec = int(time_sec)
                    self.maximum_trial_time.append(time_sec)
                lnum += 1
