from pathlib import Path  # noqa: CPY001, D100, INP001


def createTemplate(variableNames, templateName):  # noqa: N802, N803, D103
    filePath = Path('./' + templateName)  # noqa: N806

    with open(filePath, 'w') as f:  # noqa: PLW1514, PTH123
        f.write(f'{len(variableNames)}\n')

        for name in variableNames:
            f.write(f'{name} <{name}>\n')

        f.close()
