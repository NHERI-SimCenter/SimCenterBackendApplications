
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
#include <regex>

#include "writeErrors.h"

#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/thread/thread.hpp>
//#include <boost/lockfree/queue.hpp>

#ifdef MPI_RUN
	#include <mpi.h>
#endif

writeErrors theErrorFile; // Error log

int main(int argc, char** argv)
{

	int nprocs, procno;
	#ifdef MPI_RUN
		MPI_Comm comm;
		MPI_Init(&argc, &argv);
		comm = MPI_COMM_WORLD;
		MPI_Comm_rank(comm, &procno);
		MPI_Comm_size(comm, &nprocs);		
		if (procno == 0) std::cerr << "Main - running MPI" << std::endl;
        if (procno == 0) std::cerr << "num proc = " << nprocs << std::endl;

	#else
		procno = 0;
		nprocs = 1;
		if (procno == 0) std::cerr << "running OpenMP" << std::endl;

		#if !defined(ARMA_USE_OPENMP)
			#define ARMA_USE_OPENMP
		#endif
	#endif


		//std::regex re(R"(\{([^}]+)\})");
		//std::cerr << "testing\n";

		if ((argc != 6) && (procno == 0)) {
			std::string errMsg = "Number of the additional commend line arguments is " + std::to_string(argc - 1) + ", but 5 is required. The arguments should always include the working directory / input file name / workflow driver name / os type / run type";
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

	//std::string workDir = "C:/Users/yisan/Documents/quoFEM/LocalWorkDir/tmp.SimCenter";
	//std::string osType = "Windows";
	//std::string runType = "runningLocal";

	std::string  errorDir = workDir + "/dakota.err" ;
	theErrorFile.getFileName(errorDir, procno);

	//auto elapseStart = std::chrono::high_resolution_clock::now();
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
	//	(3-1) Random number generator Gaussian(mean=0,var=1) - (batch samples)
	//	(4-1) FE Analysis - (parallel)
	//

	/*
	auto readStart = std::chrono::high_resolution_clock::now();
	mat C;
	field<std::string> header;
	C.load(csv_name("C:/Users/SimCenter/Dropbox/SimCenterPC/GSAPCA/Y.txt"));
	auto readEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - readStart).count() / 1.e3;
	std::cout << "  - reading took " << readEnd << " s\n";
	std::cout << C.size() << " s\n";

	for (int i=0; i < 10; i++) {
		std::cout << C(0,i) << " s\n";
	}
	readStart = std::chrono::high_resolution_clock::now();
	std::vector< std::vector<double> >  V(C.n_rows);
	for (size_t i = 0; i < C.n_rows; ++i) {
		V[i] = arma::conv_to< std::vector<double> >::from(C.row(i));
	};
	std::cout << "  - conversion took " << readEnd << " s\n";
	*/


	if (inp.femAppName.compare("SurrogateGP") == 0) {

		T.sample(inp, procno);
		T.simulateAppBatchSurrogate(workflowDriver, osType, runType, inp, procno, nprocs);

	} else if (inp.UQmethod.compare("Monte Carlo") == 0) {

		T.sample(inp, procno);
		T.simulateAppBatch(workflowDriver, osType, runType, inp, procno, nprocs);

	}


	//
	//	(3-1)(4-1) Read dataset

	else if (inp.UQmethod.compare("Import Data Files") == 0) {
		if (runType.compare("runningLocal") != 0) {
			std::string errMsg = "Error running SimCenterUQ: No need to run remotely when the data set is provided. Please try running it locally.";
			theErrorFile.write(errMsg);
		}
		T.readDataset(inp.inpPath, inp.outPath, inp.nrv, inp.nqoi, inp.inpFileType, inp.outFileType, inp.nmc);
	}

	//T.readDataset("C:/Users/SimCenter/Dropbox/SimCenterPC/GSAPCA/X.txt", "C:/Users/SimCenter/Dropbox/SimCenterPC/GSAPCA/Y.txt", inp.nrv, inp.nqoi, "csv", inp.nmc);
	//T.readDataset("C:/Users/SimCenter/Dropbox/SimCenterPC/GSAPCA/X.bin", "C:/Users/SimCenter/Dropbox/SimCenterPC/GSAPCA/Y.bin", inp.nrv, inp.nqoi, "binary", inp.nmc);
	//

	//std::cout<<"Just testing this location 1\n";
	else {
		std::string errMsg = "Error running SimCenterUQ: UQ method " + inp.UQmethod + " NOT unknown.";
		theErrorFile.write(errMsg);
	}

	//
	//	(5) Run UQ analysis
	//

	if (!inp.uqType.compare("Sensitivity Analysis")) {

		//
		//	(5-1) Global sensitivity analysis - (parallel)
		//


		if (inp.UQmethod.compare("Import Data Files") != 0) {
			std::cout << "Writing Tab.out" << std::endl;
			runForward ForwardResults(T.X, T.G, procno);
			ForwardResults.writeTabOutputs(inp, procno); 	//	Write dakotaTab.out
		}
		std::cout << "Preparing sensitivity analysis" << std::endl;
		runGSA GsaResults(T.X, T.G, inp.groups, inp.PCAvarRatioThres, inp.qoiVectRange, 10, procno, nprocs); // int Kos = 25;
		elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - elapseStart).count()/1.e3;
		GsaResults.writeOutputs(inp, elapsedTime, procno); //	Write dakota.out
	}
	else if (!inp.uqType.compare("Forward Propagation")) {

		std::cout << "Running Forward Propagation" << std::endl;

		//
		//	(5-2) Forward analysis
		//
		runForward ForwardResults(T.X, T.G, procno);
		ForwardResults.writeTabOutputs(inp, procno); 	//	Write dakotaTab.out
		elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - elapseStart).count() /1.e3;
		ForwardResults.writeOutputs(inp, procno);		//	Write dakota.out <- curretly not being used anywhere
	}
	theErrorFile.close();
	if (procno == 0) std::cout << "Elapsed TOTAL ANALYSIS time: " << elapsedTime << " s\n";

	#ifdef MPI_RUN
		MPI_Finalize();
		theErrorFile.print("MPI done");
	#else
		theErrorFile.print("OpenMP done");
	#endif
	return 0;
}

