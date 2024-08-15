from pathlib import Path


def createTemplate(variableNames, templateName):
    filePath = Path('./' + templateName)

    with open(filePath, 'w') as f:
        f.write(f'{len(variableNames)}\n')

        for name in variableNames:
            f.write(f'{name} <{name}>\n')

        f.close()
