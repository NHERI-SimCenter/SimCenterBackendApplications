# SimCenterBackendApplications

[![Build status](https://ci.appveyor.com/api/projects/status/2fitp6tm5sj00qwr?svg=true)](https://ci.appveyor.com/project/fmckenna/simcenterbackendapplications)

This repo contains the backend applications that make up SimCeA (SimCenter Aquarium). SimCeA is a collection of applications developed by SimCenter that are utilized when the research applications (quoFEM, EE-UQ, WE-UQ, PBE, and RDT) perform the numerical computations. These computations are performed using scientific workflows; the software that runs the workflows are found in the modules/Workflow directory. The applications these workflows invoke when running are found in the other directories under modules. 


## Building the Applications

This is OS dependent. Have a look at the appvey.yml file included to see what needs to be installed (in addition to Qt) and what the terminal or powershell commands are to build the applications. The instructions are posted in the documentation for each of the research applications.

## Acknowledgments

This material is based upon work supported by the National Science Foundation under grants #1612843 and #2131111

### Contact

NHERI-SimCenter nheri-simcenter@berkeley.edu
