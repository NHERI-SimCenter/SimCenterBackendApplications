"""Created on Wed Jan 10 14:23:09 2024

@author: snaeimi

This is the main final to run REWET. This file supersedes initial.py to run
REWET. In order to keep the backward compatibility, the initial.py is kepy,
so one can run initial.py to run REWET. currently, REWET's GUI still works with
initial.py. Main.py is going to be the most developed tool.

"""

import argparse
import os
import sys

from initial import Starter

if __name__ == '__main__':
    argParser = argparse.ArgumentParser(
        prog='REWET V0.2',
        description='REstoration tool for Restoration of Water after Event Tool is a package for modeling damages and restoration in water network. You can specify settings in with providing a JSON. An example JSON file is provided in example folder. Modify the example folder and provide its path as an input. If not provided, the default settings values from the input/settings.py will be ran. thus, you can alternatively modify values in settings for a single run.',
    )

    argParser.add_argument('--json', '-j', default=None, help='json settings file')

    argParser.add_argument(
        '--project', '-p', default=None, help='REWET project file'
    )

    parse_namespace = argParser.parse_args()

    starter = Starter()
    # No file is pacified, so the default values in settings file is going to
    # be ran.

    if parse_namespace.json == None and parse_namespace.project == None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            starter.run()
        sys.exit(0)
    elif parse_namespace.json != None and parse_namespace.project == None:
        if parse_namespace.json.split('.')[-1].upper() != 'JSON':
            print(
                'ERROR in json file name: ',
                parse_namespace.json,
                'The json file must have json extention',
            )
            sys.exit(0)
        elif not os.path.exists(parse_namespace.json):
            print('ERROR in json file: ', parse_namespace.json, 'does not exist')
        else:
            starter.run(parse_namespace.json)

    elif parse_namespace.json == None and parse_namespace.project != None:
        if parse_namespace.project.split('.')[-1].upper() != 'PRJ':
            print(
                'ERROR in project file name: ',
                parse_namespace.project,
                'The project file must have PRJ extention',
            )
            sys.exit(0)
        elif not os.path.exists(parse_namespace.project):
            print(
                'ERROR in project file: ', parse_namespace.project, 'does not exist'
            )
        else:
            starter.run(parse_namespace.project)

    else:
        print(
            'ERROR in arguments\n',
            'Either of the json or project file arguments must be used',
        )

else:
    print(
        'Main File has been ran with not being the main module (i.e.,\
          __name__ is not "__main__"'
    )
