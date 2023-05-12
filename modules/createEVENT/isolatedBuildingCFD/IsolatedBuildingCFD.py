from __future__ import print_function
import os, sys
import re
import json
import argparse

	
if __name__ == "__main__":
    """
    Entry point to generate event file using IsolatedBuildindCFD
    """
    #CLI parser
    #parser = argparse.ArgumentParser(description="Get sample EVENT file produced by CFD")
    #parser.add_argument('-b', '--filenameAIM', help="BIM File", required=True)
    #parser.add_argument('-e', '--filenameEVENT', help= "Event File", required=True)
    #parser.add_argument('--getRV', help= "getRV", required=False, action='store_true')

    #parsing arguments
    #arguments, unknowns = parser.parse_known_args()

    #if arguments.getRV == True:
    #    #Read the number of floors
    #    floorsCount = GetFloorsCount(arguments.filenameAIM)
    #    forces = []
    #    for i in range(floorsCount):
    #        forces.append(FloorForces())

    #    writeEVENT(forces, arguments.filenameEVENT)
    print('IsolatedBuildingCFD - Done')
    
    
    
