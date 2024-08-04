import json  # noqa: CPY001, D100, EXE002, INP001
import sys


def main(inputFile, outputFile):  # noqa: N803, D103
    extraArgs = sys.argv[3:]  # noqa: N806

    # initialize the log file
    with open(inputFile) as f:  # noqa: PLW1514, PTH123
        data = json.load(f)

    for k, val in zip(extraArgs[0::2], extraArgs[1::2]):
        data[k] = val

    with open(outputFile, 'w') as outfile:  # noqa: PLW1514, PTH123
        json.dump(data, outfile)


if __name__ == '__main__':
    main(inputFile=sys.argv[1], outputFile=sys.argv[2])
