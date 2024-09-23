from pathlib import Path  # noqa: INP001, D100


def createTemplate(variableNames, templateName):  # noqa: N802, N803, D103
    filePath = Path('./' + templateName)  # noqa: N806

    with open(filePath, 'w') as f:  # noqa: PTH123
        f.write(f'{len(variableNames)}\n')

        for name in variableNames:
            f.write(f'{name} <{name}>\n')

        f.close()
