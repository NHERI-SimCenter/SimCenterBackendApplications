# written: Michael Gardner @ UNR

# import sys


def prepareUQ(paramsFile, inputFile, outputFile, rvSpecifier):
    # These are the delimiter choices, which can expanded as more UQ programs are added. Remember to also
    # extend the factory in rvDelimiter to handle addtional cases
    rvDelimiterChoices = ['SimCenterDelimiter', 'UQpyDelimiter']

    if rvSpecifier not in rvDelimiterChoices:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Tue Jul 30 05:16:30 PM PDT 2024
        # JVM
        # Commenting out the following, since it is invalid.

        # except IOError:
        #     print("ERROR: preProcessUQ.py: Symbol identifying value as random variable not recognized : ", rvSpecifier)

        pass
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Open parameters file and read parameter settings
    numRVs = 0
    lineCount = 0
    rvNames = []
    rvSettings = []

    try:
        with open(paramsFile) as params:
            for line in params:
                if lineCount == 0:
                    rvNames = [i.strip() for i in line.split(',')]
                    numRVs = len(rvNames)
                    # Replace RV names based on delimiter
                    for i, rv in enumerate(rvNames):
                        rvNames[i] = rvSpecifier.replaceRV(rv)

                else:
                    rvSettings = [i.strip() for i in line.split(',')]

                lineCount = lineCount + 1

    except OSError:
        print('ERROR: preProcessUQ.py could not open parameters file: ' + paramsFile)

    # Next, open input file and search for random variables that need to be replaced by parameter realizations
    inputTemplate = 'inputTemplate'
    realizationOutput = 'outputFile'
    try:
        inputTemplate = open(inputFile)
    except OSError:
        print(
            'ERROR: preProcessUQ.py could not open input template file: ' + inputFile
        )

    try:
        realizationOutput = open(outputFile, 'w')
    except OSError:
        print('ERROR: preProcessUQ.py could not open output file: ' + outputFile)

    # Iterate over all lines in input template
    for line in inputTemplate:
        # Iterate over all RVs to check they need to be replaced
        for i, rv in enumerate(rvNames):
            try:
                line = line.replace(rv, rvSettings[i])
            except:
                pass

        realizationOutput.write(line)

    realizationOutput.close()
