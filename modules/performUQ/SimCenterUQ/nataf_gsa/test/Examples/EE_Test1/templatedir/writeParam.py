import os
import sys


def main():
    paramsIn = sys.argv[1]
    paramsOut = sys.argv[2]

    if not os.path.isfile(paramsIn):
        print(f'Input param file {paramsIn} does not exist. Exiting...')
        sys.exit()

    outFILE = open(paramsOut, 'w')

    with open(paramsIn) as inFILE:
        line = inFILE.readline()
        splitLine = line.split()
        numRV = int(splitLine[3])
        print(numRV, file=outFILE)

        for i in range(numRV):
            line = inFILE.readline()
            splitLine = line.split()
            nameRV = splitLine[1]
            valueRV = splitLine[3]
            print(f'{nameRV} {valueRV}', file=outFILE)

    outFILE.close
    inFILE.close


if __name__ == '__main__':
    main()
