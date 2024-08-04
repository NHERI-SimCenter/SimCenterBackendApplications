# written: fmk, adamzs  # noqa: CPY001, D100, INP001

# import functions for Python 2.X support
import sys

if sys.version.startswith('2'):
    range = xrange  # noqa: A001, F821
    string_types = basestring  # noqa: F821
else:
    string_types = str

import os
import subprocess  # noqa: S404
from time import gmtime, strftime


class WorkFlowInputError(Exception):  # noqa: D101
    def __init__(self, value):
        self.value = value

    def __str__(self):  # noqa: D105
        return repr(self.value)


try:
    basestring  # noqa: B018
except NameError:
    basestring = str


def workflow_log(msg):  # noqa: D103
    # ISO-8601 format, e.g. 2018-06-16T20:24:04Z
    print('%s %s' % (strftime('%Y-%m-%dT%H:%M:%SZ', gmtime()), msg))  # noqa: T201, UP031


# function to return result of invoking an application
def runApplication(application_plus_args):  # noqa: N802, D103
    if application_plus_args[0] == 'python':
        command = f'python "{application_plus_args[1]}" ' + ' '.join(
            application_plus_args[2:]
        )
    else:
        command = f'"{application_plus_args[0]}" ' + ' '.join(
            application_plus_args[1:]
        )

    try:
        result = subprocess.check_output(  # noqa: S602
            command, stderr=subprocess.STDOUT, shell=True
        )
        # for line in result.split('\n'):
        # pass
        # print(line)
        returncode = 0
    except subprocess.CalledProcessError as e:
        result = e.output
        returncode = e.returncode

    if returncode != 0:
        workflow_log('NON-ZERO RETURN CODE: %s' % returncode)  # noqa: UP031
    return command, result, returncode


def add_full_path(possible_filename):  # noqa: D103
    if not isinstance(possible_filename, basestring):
        return possible_filename
    if os.path.exists(possible_filename):  # noqa: PTH110
        if os.path.isdir(possible_filename):  # noqa: PTH112
            return os.path.abspath(possible_filename) + '/'  # noqa: PTH100
        else:  # noqa: RET505
            return os.path.abspath(possible_filename)  # noqa: PTH100
    else:
        return possible_filename


def recursive_iter(obj):  # noqa: D103
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, basestring):
                obj[k] = add_full_path(v)
            else:
                recursive_iter(v)
    elif any(isinstance(obj, t) for t in (list, tuple)):
        for idx, item in enumerate(obj):
            if isinstance(item, basestring):
                obj[idx] = add_full_path(item)
            else:
                recursive_iter(item)


def relative2fullpath(json_object):  # noqa: D103
    recursive_iter(json_object)
