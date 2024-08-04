#!/usr/bin/python  # noqa: CPY001, D100, EXE001

# written: fmk, adamzs 01/18

# import functions for Python 2.X support
import sys

if sys.version.startswith('2'):
    range = xrange  # noqa: A001, F821
    string_types = basestring  # noqa: F821
else:
    string_types = str

import sys


def process_results(inputArgs):  # noqa: ANN001, ANN201, N803, D103
    #
    # process output file "node.out" for nodal displacements
    #

    with open('node.out') as inFile:  # noqa: N806, PLW1514, PTH123
        line = inFile.readline()
        line = inFile.readline()
        line = inFile.readline()
        displ = line.split()
        numNode = len(displ)  # noqa: N806

    inFile.close  # noqa: B018

    # now process the input args and write the results file

    outFile = open('results.out', 'w')  # noqa: N806, PLW1514, PTH123, SIM115

    # note for now assuming no ERROR in user data
    for i in inputArgs:
        theList = i.split('_')  # noqa: N806

        if len(theList) == 4:  # noqa: PLR2004
            dof = int(theList[3])
        else:
            dof = 1

        if theList[0] == 'Node':
            nodeTag = int(theList[1])  # noqa: N806

            if nodeTag > 0 and nodeTag <= numNode:
                if theList[2] == 'Disp':
                    nodeDisp = abs(float(displ[((nodeTag - 1) * 2) + dof - 1]))  # noqa: N806
                    outFile.write(str(nodeDisp))
                    outFile.write(' ')
                else:
                    outFile.write('0. ')
            else:
                outFile.write('0. ')
        else:
            outFile.write('0. ')

    outFile.close  # noqa: B018


if __name__ == '__main__':
    n = len(sys.argv)
    responses = []
    for i in range(1, n):
        responses.append(sys.argv[i])  # noqa: PERF401

    process_results(responses)
