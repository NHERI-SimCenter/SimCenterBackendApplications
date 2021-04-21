import sys, os, json

def main(inputFile, outputFile):

    extraArgs = sys.argv[3:]
    
    # initialize the log file
    with open(inputFile, 'r') as f:
        data = json.load(f)

    for k,val in zip(extraArgs[0::2],extraArgs[1::2]):
        data[k]=val

    with open(outputFile, 'w') as outfile:
        json.dump(data, outfile)


if __name__ == "__main__":
    main(inputFile=sys.argv[1], outputFile=sys.argv[2])

