# This script contains functions for reading and writing
# OpenFoam dictionaries and filses.
#
import numpy as np
import os


def find_keyword_line(dict_lines, keyword):
    start_line = -1

    count = 0
    for line in dict_lines:
        l = line.lstrip(' ')

        if l.startswith(keyword):
            start_line = count
            break

        count += 1

    return start_line


def write_foam_field(field, file_name):
    """
    Writes a given numpy two dimensional array to OpenFOAM
    field format. It can handel the following formats:
        pointField,
        vectorField,
        tensorField,
        symmTensorField
    """
    if os.path.exists(file_name):
        os.remove(file_name)

    foam_file = open(file_name, 'w+')

    size = np.shape(field)

    foam_file.write('{}'.format(size[0]))
    foam_file.write('\n(')

    for i in range(size[0]):
        line = '\n('
        for j in range(size[1]):
            line += ' {:.6e}'.format(field[i, j])
        line += ')'
        foam_file.write(line)

    foam_file.write('\n);')
    foam_file.close()


def write_scalar_field(field, file_name):
    """
    Writes a given one dimensional numpy array to OpenFOAM
    scalar field format.
    """
    if os.path.exists(file_name):
        os.remove(file_name)

    foam_file = open(file_name, 'w+')

    size = np.shape(field)

    foam_file.write('{}'.format(size[0]))
    foam_file.write('\n(')

    for i in range(size[0]):
        foam_file.write('\n {:.6e}'.format(field.flatten()[i]))

    foam_file.write('\n);')
    foam_file.close()
