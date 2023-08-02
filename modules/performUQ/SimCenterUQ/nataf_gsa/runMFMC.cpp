
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
 *  @date    7/2023
 *  @section DESCRIPTION
 *  Run Multi-fidelity MC Simulation
 */

#include "runMFMC.h"
#include "ERANataf.h"
#include <iterator>
#include <chrono>

using boost::math::normal;

runMFMC::runMFMC() {}
runMFMC::runMFMC(string workflowDriver,
	string osType,
	string runType,
	jsonInput inp,
	ERANataf T,
	int procno,
	int nproc)
{

	int nPilot = 50;
	int CB = 5 * 60; //sec
	//
	// Get number of models
	//

	modelID=-1;
	for (int i = 0; i < inp.nrv; i++) {
		if (inp.rvNames[i].rfind("MultiModel-", 0) == 0) {
			modelID = i;
		}
	}
	if (modelID < 0) {
		std::string errMsg = "Error running UQ engine: MultiModel index variable not found from the RV json";
		theErrorFile.write(errMsg);
	}

	auto numModels = T.M[modelID].theDist->getParam().size()/2;

	//
	// Pilot samples
	//

	vector<vector<double>> uvals(nPilot, vector<double>(inp.nrv, 0.0));
	vector<vector<int>> resampIDvals(nPilot, vector<int>(inp.nreg, 0.0));
	vector<vector<string>> discreteStrSamps(nPilot, vector<string>(inp.nst, ""));

	T.sample(nPilot, inp, procno, uvals, resampIDvals, discreteStrSamps);


	//
	// Simulate pilot samples
	//

	vector<vector<vector<double>>> xvals_list;
	vector<vector<vector<double>>> gvals_list;
	vector<double> cost_list; 

	int numExistingDirs = 0;

	for (int nm = 0; nm < numModels; nm++) { // numModels is not large
		auto elapseStart = std::chrono::high_resolution_clock::now();

		vector<vector<double>> xvals(nPilot, vector<double>(inp.nrv, 0.0));
		vector<vector<double>> gvals(nPilot, std::vector<double>(inp.nqoi, 0));

		// Update Model ID in uvals
		this->updateModelIndex(nm + 1, T, uvals);

		// Simulate
		T.simulateAppBatch(workflowDriver, osType, runType, inp, uvals, resampIDvals, discreteStrSamps, numExistingDirs, xvals, gvals, procno, nproc);
		xvals_list.push_back(xvals);
		gvals_list.push_back(gvals);
		numExistingDirs += nPilot;

		cost_list.push_back( (double) std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - elapseStart).count() / 1.e3 / nPilot); // unit:seconds

	}


	vector<vector<double>> var_list;		
	vector<vector<double>> corr_mat_list; 
	vector<vector<double>> alpha_list;
	vector<vector<double>> ratio_list;
	vector<vector<double>> N_list;


	for (int ng = 0; ng < inp.nqoi; ng++) {

		vector<double> var_tmp;
		vector<double> corr_tmp;
		vector<double> alpha_tmp;
		// value of model 1
		vector<double> gvec0;
		double var0;
		for (int nm = 0; nm < numModels; nm++) { // numModels is not large

			//
			// Compute variance
			//
			
			vector<double> gvec;

			for (int ns = 0; ns < nPilot; ns++) {
				gvec.push_back(gvals_list[nm][ns][ng]);
			}

			double mean_val = calMean(gvec);
			double var_val = calVar(gvec, mean_val);


			//
			// Compute correlation
			//

			/***
			 for (int nm2 = 0; nm2 < numModels; nm2++) {
				if (nm2 > nm) {
					//top triangle
					corr_tmp2.push_back(0.0);
				}
				else if (nm2 == nm) {
					corr_tmp2.push_back(1.0);
				}
				else {
					vector<double> gvec2;
					for (int ns = 0; ns < nPilot; ns++) {
						gvec2.push_back(gvals_list[nm2][ns][ng]);
					}
					corr_tmp2.push_back(correlationCoef(gvec, gvec2));
				}
			}
			***/

			if (nm == 0) {
				corr_tmp.push_back(1.0);
				gvec0 = gvec;
				var0 = var_val;
			} else {
				corr_tmp.push_back(correlationCoef(gvec, gvec0));
			}

			// save values
			var_tmp.push_back(var_val);
			alpha_tmp.push_back(corr_tmp[nm]*std::sqrt(var0/var_val));
		}

		//
		// Optimal ratio
		//

		vector<double> ratio_tmp;
		double rho2, rho2_;
		ratio_tmp.push_back(1.0);
		double sumLowCosts = 0; // used to compute N
		double varFactor = 0; // used to compute Var[H0]
		for (int nm = 1; nm < numModels; nm++) { // numModels is not large
			rho2 = corr_tmp[nm] * corr_tmp[nm];
			if (nm + 1 == numModels) {
				rho2_ = 0;
			}
			else {
				rho2_ = corr_tmp[nm + 1] * corr_tmp[nm + 1];
			}
			double ratio = std::sqrt((cost_list[0] * (rho2 - rho2_)) / (cost_list[nm] * (1 - rho2)));
			ratio_tmp.push_back(ratio);

			sumLowCosts += ratio * cost_list[nm];
			varFactor += (1 / ratio_tmp[nm - 1] - 1 / ratio) * rho2;
		}

		double N = ((double)CB) / (cost_list[0] + sumLowCosts); // Optimal number for the first model
		double VarEst = var_tmp[0]/N*(1-varFactor); // Estimation Variance

		//
		// save quantities
		//

		var_list.push_back(var_tmp);
		corr_mat_list.push_back(corr_tmp);
		alpha_list.push_back(alpha_tmp);
		ratio_list.push_back(ratio_tmp);
	}



	//
	// Pilot samples
	//

	//this->xval = xvals;
	//this->xstrval = discreteStrSamps;
	//this->gval = gvals;
}

runMFMC::~runMFMC() {};

void 
runMFMC::computeStatistics(int procno) {

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

double runMFMC::correlationCoef(vector<double> X, vector<double> Y) {

	double n = X.size();
	double sum_X = 0, sum_Y = 0, sum_XY = 0;
	double squareSum_X = 0, squareSum_Y = 0;

	for (int i = 0; i < n; i++)
	{
		// sum of elements of array X.
		sum_X = sum_X + X[i];

		// sum of elements of array Y.
		sum_Y = sum_Y + Y[i];

		// sum of X[i] * Y[i].
		sum_XY = sum_XY + X[i] * Y[i];

		// sum of square of array elements.
		squareSum_X = squareSum_X + X[i] * X[i];
		squareSum_Y = squareSum_Y + Y[i] * Y[i];
	}

	// use formula for calculating correlation coefficient.
	double corr = (n * sum_XY - sum_X * sum_Y)
		/ std::sqrt((n * squareSum_X - sum_X * sum_X)
			* (n * squareSum_Y - sum_Y * sum_Y));

	return corr;
}


void runMFMC::updateModelIndex(int modelNo, ERANataf T, vector< vector<double>> &uvals) {

	int nsamp = uvals.size();
	//vector<vector<double>> xvals = T.U2X(nsamp, uvals);
	//double id_u = T.M[modelID].theDist->getQuantile(T.M[modelID].theDist->getCdf(modelNo));

	normal stdNorm(0., 1.);

	double id_u = quantile(stdNorm, T.M[modelID].theDist->getCdf(modelNo));
	for (int ns = 0; ns < nsamp; ns++) {
		//xvals[ns][modelID] = nm + 1;
		uvals[ns][modelID] = id_u;
	}
	//uvals = T.X2U(nsamp, xvals);
}

double runMFMC::calMean(vector<double> x) {
	double sum = std::accumulate(std::begin(x), std::end(x), 0.0);
	return sum / x.size();
}

double runMFMC::calStd(vector<double> x, double m) {
	return std::sqrt(calVar(x,m));
}

double runMFMC::calVar(vector<double> x, double m) {
	double accum = 0.0;
	std::for_each(std::begin(x), std::end(x), [&](const double d) {
		accum += (d - m) * (d - m);
	});
	return accum / (x.size());
}

double runMFMC::calSkewness(vector<double> x, double m, double s) {
	double accum = 0.0;
	std::for_each(std::begin(x), std::end(x), [&](const double d) {
		accum += (d - m) * (d - m) * (d - m);
		});
	return (accum / (x.size()) / (s*s*s));
}

double runMFMC::calKurtosis(vector<double> x, double m, double s) {
	double accum = 0.0;
	std::for_each(std::begin(x), std::end(x), [&](const double d) {
		accum += (d - m) * (d - m) * (d - m) * (d - m);
		});
	return (accum / (x.size()) / (s * s * s * s));
	 
}
void runMFMC::writeOutputs(jsonInput inp, int procno)
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


void runMFMC::writeTabOutputs(jsonInput inp, int procno)
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

bool runMFMC::isInteger(double a) {
	double b = round(a), epsilon = 1e-9; //some small range of error
	return (a <= b + epsilon && a >= b - epsilon);
}
