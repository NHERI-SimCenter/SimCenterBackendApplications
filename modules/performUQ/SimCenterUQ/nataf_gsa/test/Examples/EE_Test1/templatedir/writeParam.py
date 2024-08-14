import os  # noqa: CPY001, D100, INP001
import sys


def main():  # noqa: D103
    paramsIn = sys.argv[1]  # noqa: N806
    paramsOut = sys.argv[2]  # noqa: N806

    if not os.path.isfile(paramsIn):  # noqa: PTH113
        print(f'Input param file {paramsIn} does not exist. Exiting...')  # noqa: T201
        sys.exit()

    outFILE = open(paramsOut, 'w')  # noqa: N806, PLW1514, PTH123, SIM115

    with open(paramsIn) as inFILE:  # noqa: N806, PLW1514, PTH123
        line = inFILE.readline()
        splitLine = line.split()  # noqa: N806
        numRV = int(splitLine[3])  # noqa: N806
        print(numRV, file=outFILE)

        for i in range(numRV):  # noqa: B007
            line = inFILE.readline()
            splitLine = line.split()  # noqa: N806
            nameRV = splitLine[1]  # noqa: N806
            valueRV = splitLine[3]  # noqa: N806
            print(f'{nameRV} {valueRV}', file=outFILE)

    outFILE.close  # noqa: B018
    inFILE.close  # noqa: B018


if __name__ == '__main__':
    main()
