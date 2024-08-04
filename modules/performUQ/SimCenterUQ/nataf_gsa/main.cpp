
/* *****************************************************************************
Copyright (c) 2016-2017, The Regents of the University of California (Regents).
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED HEREUNDER IS
PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

*************************************************************************** */

/**
 *  @author  Sang-ri Yi
 *  @date    8/2021
 *  @section DESCRIPTION
 *  Backend application to run SimCenterUQ engine in the quoFEM application developed by SimCenter. 
 */

#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
#include <chrono>

#include "jsonInput.h"
#include "ERADist.h"
#include "ERANataf.h"
#include "runGSA.h"
#include "runForward.h"
#include "runMFMC.h"
#include <regex>

#include "writeErrors.h"

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/thread/thread.hpp>
//#include <boost/lockfree/queue.hpp>

#ifdef MPI_RUN
	#include <mpi.h>
#endif

writeErrors theErrorFile; // Error log

int main(int argc, char** argv) {
    int a = 0;
    int nprocs, procno;
#ifdef MPI_RUN
    //Running Remote
    MPI_Comm comm;
    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &procno);
    MPI_Comm_size(comm, &nprocs);
    if (procno == 0) std::cerr << "Main - running MPI" << std::endl;
    if (procno == 0) std::cerr << "num proc = " << nprocs << std::endl;
#else
    //Running Local
    procno = 0;
    nprocs = 1;
    if (procno == 0) std::cerr << "running OpenMP" << std::endl;
#if !defined(ARMA_USE_OPENMP)
#define ARMA_USE_OPENMP
#endif
#endif

    if ((argc != 6) && (procno == 0)) {
        std::string errMsg = "Number of the additional command line arguments is " + std::to_string(argc - 1) +
                             ", but 5 is required. The arguments should always include the working directory / input file name / workflow driver name / os type / run type";
        std::cerr << errMsg << std::endl;
        theErrorFile.abort();
    }

    std::string workDir = argv[1];
    std::string inpFile = argv[2];
    std::string workflowDriver = argv[3];
    std::string osType = argv[4];
    std::string runType = argv[5];

    if (procno == 0) std::cerr << "* WORKDIR: " << workDir << "\n";
    if (procno == 0) std::cerr << "* INPFILE: " << inpFile << "\n";
    if (procno == 0) std::cerr << "* WORKDRIVER: " << workflowDriver << "\n";
    if (procno == 0) std::cerr << "* OS: " << osType << "\n";
    if (procno == 0) std::cerr << "* RUN: " << runType << "\n\n";

    std::string errorDir = workDir + "/dakota.err";
    theErrorFile.getFileName(errorDir, procno);

    auto elapseStart = std::chrono::high_resolution_clock::now();
    double elapsedTime;

    //
    //  (1) read JSON file
    //

    jsonInput inp(workDir, inpFile, procno);

    //
    //	(2) Construct Nataf Object
    //
    ERANataf T(inp, procno);

    //
    //	(3) Run UQ analysis
    //

    if (!inp.uqType.compare("Sensitivity Analysis")) {

        //
        //	(3-1) Global sensitivity analysis - (parallel)
        //
        runGSA myGSA(workflowDriver, osType, runType, inp, T, procno, nprocs);

        elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - elapseStart).count() / 1.e3;
        myGSA.writeOutputs(inp, elapsedTime, procno); //	Write dakota.out
        myGSA.writeTabOutputs(inp, procno); //	Write dakota.out

    } else if (!inp.uqType.compare("Forward Propagation")) {


        if (!inp.uqMethod.compare("Monte Carlo")) {
            std::cout << "Running Forward Propagation" << std::endl;

            //
            //	(3-2) Forward analysis
            //
            runForward myForward(workflowDriver, osType, runType, inp, T, procno, nprocs);

            myForward.writeTabOutputs(inp, procno);    //	Write dakotaTab.out
            elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - elapseStart).count() / 1.e3;
        } else if (!inp.uqMethod.compare("Multi-fidelity Monte Carlo")) {

            std::cout << "Running MFMC" << std::endl;

            //
            //	(3-2) MFMC
            //

            runMFMC myMFMC(workflowDriver, osType, runType, inp, T, procno, nprocs);

            myMFMC.writeTabOutputs();    //	Write dakotaTab.out
            //myMFMC.writeInfo();    //	Write dakotaTab.out

            elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - elapseStart).count() / 1.e3;
            myMFMC.writeOutputs(elapsedTime);

        }
    }

	if (procno == 0) std::cout << "Elapsed TOTAL ANALYSIS time: " << elapsedTime << " s\n";

	#ifdef MPI_RUN
		MPI_Finalize();
		theErrorFile.print("MPI done");
	#else
		theErrorFile.print("OpenMP done");
	#endif	
		
	theErrorFile.close();

	return 0;
}

