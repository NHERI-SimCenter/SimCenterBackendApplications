"""This script contains functions for reading and writing OpenFoam dictionaries."""  # noqa: INP001, D404

import os

import numpy as np


def find_keyword_line(dict_lines, keyword):  # noqa: D103
    start_line = -1

    count = 0
    for line in dict_lines:
        l = line.lstrip(' ')  # noqa: E741

        if l.startswith(keyword):
            start_line = count
            break

        count += 1  # noqa: SIM113

    return start_line


def write_foam_field(field, file_name):
    """Writes a given numpy two dimensional array to OpenFOAM
    field format. It can handle the following formats:
    pointField,
    vectorField,
    tensorField,
    symmTensorField
    """  # noqa: D205, D400, D401
    if os.path.exists(file_name):  # noqa: PTH110
        os.remove(file_name)  # noqa: PTH107

    foam_file = open(file_name, 'w+')  # noqa: SIM115, PTH123

    size = np.shape(field)

    foam_file.write(f'{size[0]}')
    foam_file.write('\n(')

    for i in range(size[0]):
        line = '\n('
        for j in range(size[1]):
            line += f' {field[i, j]:.6e}'
        line += ')'
        foam_file.write(line)

    foam_file.write('\n);')
    foam_file.close()


def write_scalar_field(field, file_name):
    """Writes a given one dimensional numpy array to OpenFOAM
    scalar field format.
    """  # noqa: D205, D401
    if os.path.exists(file_name):  # noqa: PTH110
        os.remove(file_name)  # noqa: PTH107

    foam_file = open(file_name, 'w+')  # noqa: SIM115, PTH123

    size = np.shape(field)

    foam_file.write(f'{size[0]}')
    foam_file.write('\n(')

    for i in range(size[0]):
        foam_file.write(f'\n {field.flatten()[i]:.6e}')

    foam_file.write('\n);')
    foam_file.close()
