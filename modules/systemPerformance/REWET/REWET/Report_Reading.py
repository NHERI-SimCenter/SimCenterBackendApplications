"""Created on Tue Oct  4 16:07:24 2022

@author: snaeimi
"""  # noqa: N999, D400, D415

import datetime


def parseTimeStamp(time_stamp):  # noqa: ANN001, ANN201, N802, D103
    striped_time_stamp = time_stamp.split(':')
    hour = striped_time_stamp[0]
    minute = striped_time_stamp[1]
    second = striped_time_stamp[2]

    hour = int(hour)
    minute = int(minute)
    second = int(second)

    return (hour, minute, minute)


class Report_Reading:  # noqa: N801, D101
    def __init__(self, file_addr):  # noqa: ANN001, ANN204, D107
        self.file_data = {}
        self.maximum_trial_time = []
        with open(file_addr, encoding='utf-8') as f:  # noqa: PTH123
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
                lnum += 1  # noqa: SIM113
