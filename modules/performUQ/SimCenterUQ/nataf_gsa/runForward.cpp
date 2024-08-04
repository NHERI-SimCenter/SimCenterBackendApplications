
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
 *  Calculates the moments of QoIs and writes the results at dakotaTab.out
 */

#include "runForward.h"
#include "ERANataf.h"
#include <iterator>

runForward::runForward() {}
runForward::runForward(string workflowDriver,
	string osType,
	string runType,
	jsonInput inp,
	ERANataf T,
	int procno,
	int nproc)
{
	vector<vector<double>> uvals(inp.nmc, vector<double>(inp.nrv, 0.0));
	vector<vector<int>> resampIDvals(inp.nmc, vector<int>(inp.nreg, 0.0));
	vector<vector<string>> discreteStrSamps(inp.nmc, vector<string>(inp.nst, ""));
	vector<vector<double>> gvals(inp.nmc, std::vector<double>(inp.nqoi, 0));

	T.sample(inp.nmc, inp, procno, uvals, resampIDvals, discreteStrSamps);
	vector<vector<double>> xvals(inp.nmc, vector<double>(inp.nrv, 0.0));

	T.simulateAppBatch(workflowDriver, osType, runType, inp, uvals, resampIDvals, discreteStrSamps, 0, xvals, gvals, procno, nproc);

	this->xval = xvals;
	this->xstrval = discreteStrSamps;
	this->gval = gvals;
}

runForward::~runForward() {};

void 
runForward::computeStatistics(int procno) {

	if (procno == 0) {

		//this->xval = xval;
		//this->xstrval = xstrval;
		//this->gval = gmat;
		nmc = xval.size();
		nrv = xval[0].size();

		std::cout << "RV     Mean    StdDev  Skewness  Kurtosis\n";
		for (int nr = 0; nr < nrv; nr++) {
			vector<double> xvec;
			for (int ns = 0; ns < nmc; ns++) {
				xvec.push_back(xval[ns][nr]);
			}

			double mean_val = calMean(xvec);
			double stdDev_val = calStd(xvec, mean_val);
			double skewness_val = calSkewness(xvec, mean_val, stdDev_val);
			double kurtosis_val = calKurtosis(xvec, mean_val, stdDev_val);

			std::cout << "RV " << nr + 1 << ": ";
			std::cout << mean_val << " " << stdDev_val << " " << skewness_val << " " << kurtosis_val << '\n';

			mean.push_back(mean_val);
			stdDev.push_back(stdDev_val);
			skewness.push_back(skewness_val);
			kurtosis.push_back(kurtosis_val);
			if (nr > 100) {
				std::cout << "RVs from " << nr + 2 << " to " << nrv + 1 << " not displayed here for memory efficiency" << '\n';
				break;
			}
		}
	}

}

double runForward::calMean(vector<double> x) {
	double sum = std::accumulate(std::begin(x), std::end(x), 0.0);
	return sum / x.size();
}

double runForward::calStd(vector<double> x, double m) {
	double accum = 0.0;
	std::for_each(std::begin(x), std::end(x), [&](const double d) {
		accum += (d - m) * (d - m);
		});
	return std::sqrt(accum / (x.size()));
}

double runForward::calSkewness(vector<double> x, double m, double s) {
	double accum = 0.0;
	std::for_each(std::begin(x), std::end(x), [&](const double d) {
		accum += (d - m) * (d - m) * (d - m);
		});
	return (accum / (x.size()) / (s*s*s));
}

double runForward::calKurtosis(vector<double> x, double m, double s) {
	double accum = 0.0;
	std::for_each(std::begin(x), std::end(x), [&](const double d) {
		accum += (d - m) * (d - m) * (d - m) * (d - m);
		});
	return (accum / (x.size()) / (s * s * s * s));
	 
}
void runForward::writeOutputs(jsonInput inp, int procno)
{
	if (procno == 0) {

		// dakota.out
		string writingloc = inp.workDir + "/dakota.out";
		std::ofstream outfile(writingloc);


		if (!outfile.is_open()) {

			std::string errMsg = "Error running UQ engine: Unable to write dakota.out";
			theErrorFile.write(errMsg);
		}

		json outJson;

		outJson["rvNames"] = inp.rvNames;
		outJson["mean"] = mean;
		outJson["standardDeviation"] = stdDev;
		outJson["skewness"] = skewness;
		outJson["kurtosis"] = kurtosis;

		outfile << outJson.dump(4) << std::endl;
	}
}


void runForward::writeTabOutputs(jsonInput inp, int procno)
{
	if (procno==0) {
		auto dispInterv = 1.e7; 
		int dispCount = 1;
		auto readStart = std::chrono::high_resolution_clock::now();
		auto readEnd = 0;
		// dakotaTab.out
		std::string writingloc1 = inp.workDir + "/dakotaTab.out";
		std::stringstream Taboutfile;
		//std::ofstream Taboutfile(writingloc1);

		//if (!Taboutfile.is_open()) {

		//	std::string errMsg = "Error running UQ engine: Unable to write dakota.out";
		//	theErrorFile.write(errMsg);
		//}

		Taboutfile.setf(std::ios::fixed, std::ios::floatfield); // set fixed floating format
		Taboutfile.precision(7); // for fixed format

		Taboutfile << "idx\t";
		for (int j = 0; j < inp.nrv + inp.nco + inp.nre; j++) {
			Taboutfile << inp.rvNames[j] << "\t";
		}
		for (int j = inp.nrv + inp.nco + inp.nre; j < inp.nrv + inp.nco + inp.nre + inp.nst; j++) {
			Taboutfile << inp.rvNames[j] << "\t";
		}
		for (int j = 0; j < inp.nqoi; j++) {
			Taboutfile << inp.qoiNames[j] << "\t";
		}
		Taboutfile << '\n';

		std::string multiModel = "MultiModel";

		for (int ns = 0; ns < inp.nmc; ns++) {
			Taboutfile << std::to_string(ns + 1) << "\t";
			for (int nr = 0; nr < inp.nrv + inp.nco + inp.nre; nr++) {

				if ((inp.rvNames[nr].compare(0, multiModel.length(), multiModel) == 0) && isInteger(xval[ns][nr])) {
					// if rv name starts with "MultiModel", write as integer
					Taboutfile << std::to_string(int(xval[ns][nr])) << "\t";
				}
				else {
					Taboutfile << std::scientific << std::setprecision(7) << (xval[ns][nr]) << "\t";
				}
				//Taboutfile << std::to_string(xval[ns][nr]) << "\t";
				
			}
			for (int nr = 0; nr < inp.nst; nr++) {
				Taboutfile << xstrval[ns][nr] << "\t";
			}
			for (int nq = 0; nq < inp.nqoi; nq++) {
				Taboutfile << std::scientific << std::setprecision(7) << (gval[ns][nq]) << "\t";
				//Taboutfile << std::to_string(gval[ns][nq]) << "\t";
			}
			Taboutfile << '\n';

			if (ns*inp.nqoi > dispInterv*dispCount) {
				std::cout << "  - Writing Tab file in progress: " << (double)ns /(double)inp.nmc*100 << "% \n";
				dispCount += 1;

				if (dispCount == 1) {
					readEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - readStart).count() / 1.e3;
					double expWriteTime = readEnd * (double)inp.nmc / (double)10;
					if (expWriteTime > 5.0) {
						// Only if the file is large, print the expected reading time
						std::cout << "  - Expected writing time: " << expWriteTime << " s\n";
					}
				}
			}
		}
		//Taboutfile.close();
		std::ofstream Taboutfile1(writingloc1);

		if (!Taboutfile1.is_open()) {

			std::string errMsg = "Error running UQ engine: Unable to write dakota.out";
			theErrorFile.write(errMsg);
		}
		Taboutfile1 << Taboutfile.str();
		Taboutfile1.close();

		readEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - readStart).count() / 1.e3;
		std::cout << "Elapsed time to write Tab.json: " << readEnd << " s\n";
	}
}

bool runForward::isInteger(double a) {
	double b = round(a), epsilon = 1e-9; //some small range of error
	return (a <= b + epsilon && a >= b - epsilon);
}
