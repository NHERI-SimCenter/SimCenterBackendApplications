
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
#include "jsonInput.h"
#include <iterator>
#include <chrono>
#include <cmath>    // std::log(double)

using boost::math::normal;

runMFMC::runMFMC(string workflowDriver,
	string osType,
	string runType,
	jsonInput inp,
	ERANataf T,
	int procno,
	int nproc) : workflowDriver(workflowDriver), osType(osType), runType(runType), inp(inp), T(T), nproc(nproc), procno(procno)
{

	//
	// User defined variables
	//

	int nPilot = inp.nPilot;
	this->optMultipleQoI = "average"; // conservative/average/targetVar
	this->CB_init = inp.compBudget; //sec
	this->do_mean_var = true;
    this->do_log_transform = inp.doLogTransform;

	//
	// Save variables
	//

	//this->inp = inp;
	//this->workflowDriver = workflowDriver;
	//this->osType = osType;
	//this->runType = runType;
	this->globalElapseStart = std::chrono::high_resolution_clock::now();

	//
	// Get number of models
	//

	for (int i = 0; i < inp.nrv; i++) {
		if (inp.rvNames[i].rfind("MultiModel-", 0) == 0) {
			modelIDs.push_back(i);
		}
	}
	if (modelIDs.size()< 0) {
		std::string errMsg = "Error running UQ engine: MultiModel index variable not found from the RV json";
		theErrorFile.write(errMsg);
	}


	vector<int> numModels_list;
	for (int modelID : modelIDs) {
		numModels_list.push_back(T.M[modelID].theDist->getParam().size() / 2);
	}

   if (numModels_list.size()==0)
   {
       std::string errMsg = "Error running UQ engine: Multi-fidelity MC expects more than one model";
       theErrorFile.write(errMsg);
   }

	if (std::adjacent_find(numModels_list.begin(), numModels_list.end(), std::not_equal_to<>()) == numModels_list.end())
	{
		// all elements are equal to each other
		numModels = numModels_list[0];
	}
	else {
		std::string errMsg = "Error running UQ engine: MFMC requires to have the same number of 'multimodel' for each tap.";
		theErrorFile.write(errMsg);
	}

	

	//
	// setup Pilot simulations
	//

	vector<int> numSim_pilot;
	for (int nm = 0; nm < numModels; nm++) {
		numSim_pilot.push_back(nPilot);
	}


	//
	// simulate Pilot simulations
	//

	vector<vector<vector<double>>> xvals_pilot;
	vector<vector<vector<double>>> gvals_pilot;
	vector<double> cost_sum_list;
	int numExistingDirs = 0;
	
	this->simulateMFMC(numSim_pilot, numExistingDirs, xvals_pilot, gvals_pilot, cost_sum_list);

	//
	// what are QoI?
	//


	vector<vector<vector<double>>> hvals_pilot;
	hvals_pilot = this->g2h(gvals_pilot, do_mean_var, {});

	//
	// Get Optimal simulation numbers
	//

	vector<double> Var_pilot;
	vector<double> HF_est_pilot;
    vector<double> cost_list = cost_sum_list;
    std::transform(cost_list.begin(), cost_list.end(), numSim_pilot.begin(), cost_list.begin(), std::divides<>());
	vector<int> numSim_list_all;
    vector<double> speedUp_list;
	bool updateNumSim = true;
	this->getOptimalSimNums(xvals_pilot, hvals_pilot, cost_list, updateNumSim, HF_est_pilot, Var_pilot, numSim_list_all, speedUp_list);

	//
	// Simulate Additional simulations
	//

	vector<int> numSim_list_add;
	int sum_numSim = std::accumulate(std::begin(numSim_list_all), std::end(numSim_list_all), 0.0);
	if (sum_numSim == 0) {
		numSim_list_add.resize(numModels);
		std::fill(numSim_list_add.begin(), numSim_list_add.end(), 0); // initializing
		numSim_list_all = numSim_pilot;
	}
	else {
		numSim_list_add = numSim_list_all;
		std::transform(numSim_list_add.begin(), numSim_list_add.end(), numSim_pilot.begin(), numSim_list_add.begin(), std::minus<double>());
		if (*std::min_element(numSim_list_add.begin(), numSim_list_add.end()) < 0) {
			//
			// TODO: what to do when n is not as small????? = > let us uniformly assign the numbers to the remaining
			//

			double numer = 0;
			double denom = 0;

			for (int nm = 0; nm < numModels; nm++) {
				numer += cost_list[nm] * numSim_list_add[nm];
				if (numSim_list_add[nm] > 0) {
					denom += cost_list[nm] * numSim_list_add[nm];
				}
			}
			double k = numer / denom;

			for (int nm = 0; nm < numModels; nm++) {
				if (numSim_list_add[nm] > 0) {
					numSim_list_add[nm]  = std::floor(numSim_list_add[nm]*k);
				}
				else {
					numSim_list_add[nm] = 0;
				}
			}
			numSim_list_all = numSim_list_add;
			std::transform(numSim_list_all.begin(), numSim_list_all.end(), numSim_pilot.begin(), numSim_list_all.begin(), std::plus<double>());
		}
	}

	if (procno == 0) {
		std::cout << " - Adding more simulations:" << " \n";
		for (int nm = 0; nm < numModels; nm++) {
			std::cout << " - model " << nm +1 << " : " << numSim_list_add[nm] << " \n";
		}
	}

	vector<vector<vector<double>>> xvals_add;
	vector<vector<vector<double>>> gvals_add;
	vector<double> cost_sum_list2; // not used
	this->simulateMFMC(numSim_list_add, numExistingDirs, xvals_add, gvals_add, cost_sum_list2);

	//
	// Add two samples - do I need to add also 
	//

	vector<vector<vector<double>>> xvals_all, gvals_all;
	
	for (int nm = 0; nm < numModels; nm++) {
		int N = xvals_pilot[nm].size() + xvals_add[nm].size();
		assert(numSim_list_all[nm] == N);

		//
		// append x
		//

		vector<vector<double>> xvals_tmp = xvals_pilot[nm];
		xvals_tmp.insert(xvals_tmp.end(), xvals_add[nm].begin(), xvals_add[nm].end());
		xvals_all.push_back(xvals_tmp);

		//
		// append g
		//

		vector<vector<double>> gvals_tmp = gvals_pilot[nm];

		gvals_tmp.insert(gvals_tmp.end(), gvals_add[nm].begin(), gvals_add[nm].end());
		gvals_all.push_back(gvals_tmp);

	}

	//
	// Add two samples - do I need to add also 
	//

	vector<vector<vector<double>>> hvals_all;
	hvals_all = this->g2h(gvals_all, do_mean_var, {});

	vector<double> Var_est;
	vector<double> HF_est;
    vector<double> speedUp_list2;

    vector<double>  cost_list2  = cost_sum_list2;
    std::transform(cost_list2.begin(), cost_list2.end(), cost_sum_list.begin(), cost_list2.begin(), std::plus<double>());
    std::transform(cost_list2.begin(), cost_list2.end(), numSim_list_all.begin(), cost_list2.begin(), std::divides<>());

	updateNumSim = false;
	this->getOptimalSimNums(xvals_all, hvals_all, cost_list2, updateNumSim, HF_est, Var_est, numSim_list_all, speedUp_list2);


	//
	// Post process the data
	//
    vector<double> gMean, gStdDev, gVar;
    vector<double> gMean_var, gVar_var;
	if (do_mean_var) {
		
		for (int ng = 0; ng < inp.nqoi; ng++) {
			double myMean = HF_est[ng];
			double myVar = HF_est[ng + inp.nqoi] - myMean* myMean;
            gMean.push_back(myMean);
			gVar.push_back(myVar);
            gStdDev.push_back(std::sqrt(myVar));

			gMean_var.push_back(Var_est[ng]);
            gVar_var.push_back(Var_est[ng + inp.nqoi]); // TODO: this is not correct
		}
	}

	setUpRes_rvStatistics(xvals_all[numModels - 1]); // computes rvMean, rvStdDev etc.
    setUpRes_qoiStatistics(gMean,gVar,gStdDev,speedUp_list2);
	setUpRes_modelInfo(numSim_pilot, numSim_list_add, gvals_all);


	xvals = xvals_all;
	gvals = gvals_all;




}

runMFMC::~runMFMC() {};


void
runMFMC::setUpRes_modelInfo(vector<int> numSim_pilot,
                                 vector<int> numSim_list_add,
                                 vector<vector<vector<double>>> g_vec_all){

	json modelListJson = infoJson["models"];
	for (int nm = 0; nm < numModels; nm++) {
		
		//
		// compute each statistics
		//
		vector<double> mean_vec;
		vector<double> var_vec;
        vector<double> coefVar_vec;

		for (int ng = 0; ng < inp.nqoi; ng++) {
			vector<double> gvec;
			for (int ns = 0; ns < numSim_pilot[nm]+numSim_list_add[nm]; ns++) {
				gvec.push_back(g_vec_all[nm][ns][ng]);
			}

            if (do_log_transform) {
                std::transform(gvec.begin(), gvec.end(), gvec.begin(), [](double element) { return  std::log(element); }); // avg of value
            }

			double mean_val = calMean(gvec);
			double var_val = calVar(gvec, mean_val);

			mean_vec.push_back(mean_val);
			var_vec.push_back(var_val);
            coefVar_vec.push_back(std::sqrt(var_val)/mean_val);
		}

		json modelJson = modelListJson["model" + std::to_string(nm + 1)];

		modelJson["nPilot"] = numSim_pilot[nm];
		modelJson["nAdd"] = numSim_list_add[nm];
		modelJson["modelMean"] = mean_vec;
		modelJson["modelVar"] = var_vec;
        modelJson["modelCoefVar"] = coefVar_vec;
		modelListJson["model" + std::to_string(nm + 1)] = modelJson;
	}
	infoJson["models"] = modelListJson;

}

void
runMFMC::setUpRes_qoiStatistics(vector<double> gMean,
                                 vector<double> gVar,
                                 vector<double> gStdDev,
                                vector<double> speedUp_list){

    qoiJson["qoiNames"] = inp.qoiNames;
    qoiJson["mean"] = gMean;
    qoiJson["var"] = gVar;
    qoiJson["standardDeviation"] = gStdDev;
    qoiJson["speedUp"] = speedUp_list;
}


void
runMFMC::setUpResJson(vector<double> cost_list,
	vector<vector<double>> var_list,
	vector<vector<double>> mean_diff_list,
	vector<vector<double>> rho_list,
	vector<vector<double>> alpha_list,
	vector<vector<double>> ratio_list) {

	json modelListJson;
	for (int nm = 0; nm < numModels; nm++) {
		json modelJson;
		modelJson["cost_sec_per_sim"] = cost_list[nm];
		
		vector<double> rho_nm, alpha_nm, ratio_nm, mean_diff_nm, var_nm;

		for (int ng = 0; ng < var_list.size(); ng++) {
			var_nm.push_back(var_list[ng][nm]);
			rho_nm.push_back(rho_list[ng][nm]);
			alpha_nm.push_back(alpha_list[ng][nm]);
			ratio_nm.push_back(ratio_list[ng][nm]);
			mean_diff_nm.push_back(mean_diff_list[ng][nm]);
		}

		modelJson["corrCoef"] = rho_nm;
		modelJson["ratio"] = ratio_nm;
		modelJson["mean_diff"] = mean_diff_nm;
		modelJson["a"] = alpha_nm;

		modelListJson["model" + std::to_string(nm + 1)] = modelJson;
	}
	infoJson["models"] = modelListJson;
}


//void
//runMFMC::writeInfo() {
//	if (procno == 0) {		// dakota.out
//		string writingloc = inp.workDir + "/InfoMFMC.out";
//		std::ofstream outfile(writingloc);
//		//json infoJson;
//
//		if (!outfile.is_open()) {
//
//			std::string errMsg = "Error running UQ engine: Unable to write info.out";
//			theErrorFile.write(errMsg);
//		}
//
//
//		outfile << infoJson.dump(4) << std::endl;
//	}
//}

vector<vector<vector<double>>>
runMFMC::g2h(vector<vector<vector<double>>> gvals, bool do_mean_var, vector<double> perc_list) {
	
	vector<vector<vector<double>>> hvals; // #model x #samples x #gdim [nm][ns][ng]

	for (int nm = 0; nm < gvals.size(); nm++) {
		vector<vector<double>> hval_tmp;

		for (int ns = 0; ns < gvals[nm].size(); ns++) {
			vector<double> hval_tmp_ns;

			if (do_mean_var) {
				//vector<double> hval_tmp_mean = gvals[nm][ns];
                hval_tmp_ns = gvals[nm][ns];

                if (do_log_transform) {
                    std::transform(hval_tmp_ns.begin(), hval_tmp_ns.end(), hval_tmp_ns.begin(), [](double element) { return  std::log(element); }); // avg of value
                }

				vector<double> hval_tmp_meansq = hval_tmp_ns;
				std::transform(hval_tmp_meansq.begin(), hval_tmp_meansq.end(), hval_tmp_meansq.begin(), [](double element) { return element * element; }); // avg of value

				hval_tmp_ns.insert(hval_tmp_ns.end(), hval_tmp_meansq.begin(), hval_tmp_meansq.end());
				
			} else {
				string errMsg = "Error running MFMC: no MFMC objective was set";
				theErrorFile.write(errMsg);
			}

			hval_tmp.push_back(hval_tmp_ns);
		}
		hvals.push_back(hval_tmp);
	}

	return hvals;
}

void
runMFMC::simulateMFMC(vector<int>Nsims, 
						int &numExistingDirs, 
						vector<vector<vector<double>>> &xvals_list, 
						vector<vector<vector<double>>> &gvals_list, 
						vector<double> &cost_sum_list) {

	//
	//
	//

	int Nmax = *std::max_element(Nsims.begin(), Nsims.end());

	vector<vector<double>> uvals(Nmax, vector<double>(inp.nrv, 0.0));
	vector<vector<int>> resampIDvals(Nmax, vector<int>(inp.nreg, 0.0));
	vector<vector<string>> discreteStrSamps(Nmax, vector<string>(inp.nst, ""));

	T.sample(Nmax, inp, procno, uvals, resampIDvals, discreteStrSamps);

	//
	// Simulate pilot samples
	//


	for (int nm = 0; nm < numModels; nm++) { // numModels is not large

		int N = Nsims[nm];
		vector<vector<double>> uvals_nm = uvals;
		uvals_nm.resize(N);

		auto elapseStart = std::chrono::high_resolution_clock::now();

		vector<vector<double>> xvals(N, vector<double>(inp.nrv, 0.0));
		vector<vector<double>> gvals(N, std::vector<double>(inp.nqoi, 0));

		// Update Model ID in uvals
		this->updateModelIndex(nm + 1, T, uvals_nm);

		// Simulate
		T.simulateAppBatch(workflowDriver, osType, runType, inp, uvals_nm, resampIDvals, discreteStrSamps, numExistingDirs, xvals, gvals, procno, nproc);
		xvals_list.push_back(xvals);
		gvals_list.push_back(gvals);
		numExistingDirs += N;

        cost_sum_list.push_back((double)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - elapseStart).count() / 1.e3); // unit:seconds
	}

}

void runMFMC::getOptimalSimNums(vector<vector<vector<double>>>xvals_list, 
								vector<vector<vector<double>>>gvals_list, 
								vector<double>cost_list, 
								bool updateNumSim,
								vector<double>&HF_est_list,
                                vector<double>& Var_HF_list,
								vector<int>& numSim_list_int,
                                vector<double>& speedUp_list)
{



	vector<vector<double>> var_list;
	vector<vector<double>> mean_diff_list;
	vector<vector<double>> rho_list;
	vector<vector<double>> alpha_list;
	vector<vector<double>> ratio_list;

	vector<vector<double>> Nsim_list;
	vector<double> Var_HF_opt_list;
	vector<bool> check_validity_list;
	vector<string> validity_msg_list;

	int optimal_N = -1;
	int optimal_ng = -1;

	int nqoi = gvals_list[0][0].size();

	double time_passed = (double)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - globalElapseStart).count() / 1.e3; // seconds
/*
double nProcessors;
#ifdef MPI_RUN
		nProcessors = nproc;
#else
		nProcessors = omp_get_num_procs();
#endif
*/
double CB = (CB_init - time_passed); //seconds								   
	//
	// Loop QoIs
	//
	for (int ng = 0; ng < nqoi; ng++) {

		vector<double> var_tmp;
		vector<double> mean_diff_tmp;
		vector<double> corr_tmp;
		vector<double> alpha_tmp;

		vector<double> gvec0; // HF results
		double var0;		  // HF variance
		int N0;		  // HF variance

		//
		// Loop models
		//

		for (int nm = 0; nm < numModels; nm++) { // numModels is not large

			//
			// collect QoI samples of nm-th model, ng-th dimension 
			//

			vector<double> gvec;

			for (int ns = 0; ns < gvals_list[nm].size(); ns++) {
				gvec.push_back(gvals_list[nm][ns][ng]);
			}

			//
			// Compute mean and variance
			//
			double mean_val = calMean(gvec);
			double var_val = calVar(gvec, mean_val);
			double mean_val_red;
			if (nm == 0) {
				mean_val_red = 0;
			} else {
				vector<double> gvec_trun = gvec;
				gvec_trun.resize(gvals_list[nm - 1].size());
				mean_val_red = calMean(gvec_trun);
			}

			//
			// Compute correlation [nm,0]
			//

			if (nm == 0) {
				corr_tmp.push_back(1.0);
				gvec0 = gvec;
				var0 = var_val;
				N0 = gvec.size();
			}
			else {
				vector<double> gvec_trun = gvec;
				gvec_trun.resize(N0);
                double corr_val = correlationCoef(gvec_trun, gvec0);
                if (corr_val>0.999) {
                    corr_val= 0.999;
                }
				corr_tmp.push_back(corr_val);
			}

			//
			// save values at nm
			//

			var_tmp.push_back(var_val);
			mean_diff_tmp.push_back(mean_val - mean_val_red);
			alpha_tmp.push_back(corr_tmp[nm] * std::sqrt(var0 / var_val)); 
		}

		//
		// Optimal simulation ratio 
		//

		vector<double> ratio_tmp;
		double rho2_t1, rho2_t2; // term1 term2 
		double rho2_t3 = corr_tmp[1] * corr_tmp[1];; // term3
		ratio_tmp.push_back(1.0);
		double sumCosts = cost_list[0]; // used to compute N
		double varFactor = 0; // used to compute Var[H0]
		for (int nm = 1; nm < numModels; nm++) { // numModels is not large
			rho2_t1 = corr_tmp[nm] * corr_tmp[nm];
			if (nm + 1 == numModels) {
				rho2_t2 = 0;
			}
			else {
				rho2_t2 = corr_tmp[nm + 1] * corr_tmp[nm + 1];
			}

			double ratio = std::sqrt((cost_list[0] * (rho2_t1 - rho2_t2)) / (cost_list[nm] * (1 - rho2_t3)));

			if (std::isnan(ratio)) {
				ratio = 0.0;
			}

			ratio_tmp.push_back(ratio);

			sumCosts += ratio * cost_list[nm];
			varFactor += (1 / ratio_tmp[nm - 1] - 1 / ratio) * rho2_t1;
			
			// I guess the second fidelity model is the pivot model
			for (int nm2 = 1; nm2 < nm+1; nm2++) {
				if (ratio_tmp[nm2] != 0) {
					rho2_t3 = corr_tmp[nm] * corr_tmp[nm];
					break;
				}
			}		
		}

		//
		// save quantities
		//

		var_list.push_back(var_tmp);
		mean_diff_list.push_back(mean_diff_tmp);
		rho_list.push_back(corr_tmp);
		alpha_list.push_back(alpha_tmp);
		ratio_list.push_back(ratio_tmp);

        //
        // Compute speed up
        //


        double speedUpNum = cost_list[0];
        double speedUpDenom = 0;
        for (int nm = 0; nm < numModels; nm++) { // numModels is not large
            speedUpDenom += cost_list[nm] * ratio_tmp[nm];
        }
        speedUpDenom *= (1 - varFactor);
        speedUp_list.push_back(speedUpNum/speedUpDenom);

        //
		// Find the id of the most conservative QoI - that simulates HF largest amout
		//


		if (updateNumSim) {
			//
			// Optimal Nsim
			//

			//remaining time
			double VarEst, N;
			if (CB > 0) {
				N = (CB_init) / (sumCosts); // Optimal number for the first model
			}
			else {
				N = 0; // No more simulations
			}
			VarEst = var_tmp[0] / N * (1 - varFactor); // Estimation Variance

			vector<double> Nsim_tmp;
			for (int nm = 0; nm < numModels; nm++) { // numModels is not large
				Nsim_tmp.push_back(N * ratio_tmp[nm]);
			}

			if (optimal_N < N) {
				optimal_N = N;
				optimal_ng = ng;
			}

			Nsim_list.push_back(Nsim_tmp);
			Var_HF_opt_list.push_back(VarEst);

			string msg;
			bool good = checkValidity(cost_list, corr_tmp, msg);
			check_validity_list.push_back(good);
			validity_msg_list.push_back(msg);
		}
	}

	if (updateNumSim) {
		if (std::all_of(check_validity_list.begin(), check_validity_list.end(), [](bool v) { return !v; })) {
			string errMsg = "Error running MFMC: MFMC is not valid\n";
			errMsg += validity_msg_list[0] + "\n";
			theErrorFile.write(errMsg);
		}

		assert(optimal_N >= 0);
		assert(optimal_ng >= 0);

		vector<double> numSim_list;
		if (optMultipleQoI == "conservative") {
			//choose the one has highest HF simulations
			numSim_list = Nsim_list[optimal_ng];
		}
		else if (optMultipleQoI == "average") {

			numSim_list = Nsim_list[0];
			for (int ng = 1; ng < nqoi; ng++) {
				std::transform(numSim_list.begin(), numSim_list.end(), Nsim_list[ng].begin(),
					numSim_list.begin(), std::plus<double>());
				//numSim_list += Nsim_list[ng]
			}
			const double scale(1 / (double)nqoi);
			std::transform(numSim_list.begin(), numSim_list.end(), numSim_list.begin(), [scale](double element) { return element *= scale; }); // avg of value
		}
		else {
			//
			// TODO: additional options???
			//
		}

		// Save it into a integer vector
		numSim_list_int.resize(numModels);
		std::fill(numSim_list_int.begin(), numSim_list_int.end(), 0); // initializing
		std::transform(numSim_list.begin(), numSim_list.end(), numSim_list_int.begin(), [](double element) { return element = (int)std::floor(element); }); // floor values
	}

	//
	// Get final estimate of variance
	//

	for (int ng = 0; ng < nqoi; ng++) {

		double V_tmp = (double) var_list[ng][0] / numSim_list_int[0];

		for (int nm = 1; nm < numModels; nm++) {
			V_tmp += (1.0 / numSim_list_int[nm - 1] - 1.0 / numSim_list_int[nm]) * (alpha_list[ng][nm] * alpha_list[ng][nm] * var_list[ng][nm] - 2 * alpha_list[ng][nm] * rho_list[ng][nm] * std::sqrt(var_list[ng][0] * var_list[ng][nm]));
		}
		Var_HF_list.push_back(V_tmp);


		double E_tmp = (double) mean_diff_list[ng][0];

		for (int nm = 1; nm < numModels; nm++) {
			E_tmp += alpha_list[ng][nm] * mean_diff_list[ng][nm];
		}
		HF_est_list.push_back(E_tmp);
	}


	if (!updateNumSim) {
		setUpResJson(cost_list, var_list, mean_diff_list, rho_list, alpha_list, ratio_list);
	}


}

bool
runMFMC::checkValidity(vector<double> cost_list, vector<double> corr_tmp, string & errMsg) {


	for (int nm = 1; nm < numModels; nm++) {

		//
		// Check cost
		//

		if (cost_list[nm - 1] < cost_list[nm]) {
			// if lower-fidelity model takes longer time
			if (nm == 1) {
				errMsg = "The high fidelity model (Model 1) is evaluated faster than the low fidelity model (Model 2). To get the best estimates, the user should run MCS with only the high fidelity model.";
			}
			else {
				errMsg = "We assume the model with lower index value has the higher fidelity, meaning its evaluation is computationally costly but accurate. However, the mean evaluation time of model " + std::to_string(nm) + "(" + std::to_string(cost_list[nm-1]) + " sec) is smaller than that of model " + std::to_string(nm + 1) + "(" + std::to_string(cost_list[nm]) + ").";
			}
			//theErrorFile.write(errMsg);
			return false;
		}

		//
		// Check corr
		//

		if (abs(corr_tmp[nm-1]) < abs(corr_tmp[nm])) {
			// if lower-fidelity model has higher correlation to HF model
			errMsg = "We assume the model with lower index value has the higher fidelity, meaning its evaluation is computationally costly but accurate. However, the correlation of model " + std::to_string(nm) + "(" + std::to_string(corr_tmp[nm-1]) + ") is smaller than that of model " + std::to_string(nm + 1) + "(" + std::to_string(corr_tmp[nm]) + ").";
			//theErrorFile.write(errMsg);
			return false;
		}


		//
		// Check condition
		//

		double c_ratio = cost_list[nm - 1] / cost_list[nm];
		double rho_ratio;
		if (nm + 1 == numModels) {
			rho_ratio = (corr_tmp[nm - 1] * corr_tmp[nm - 1] - corr_tmp[nm] * corr_tmp[nm]) / (corr_tmp[nm] * corr_tmp[nm]);
		}
		else {
			rho_ratio = (corr_tmp[nm - 1] * corr_tmp[nm - 1] - corr_tmp[nm] * corr_tmp[nm]) / (corr_tmp[nm] * corr_tmp[nm] - corr_tmp[nm + 1] * corr_tmp[nm + 1]);
		}
		
		if (c_ratio < rho_ratio) {
			// if lower-fidelity model has higher correlation to HF model
			errMsg += "Based on cost-benefit analysis, LF model, Model " + std::to_string(nm + 1) +", is not worth running. (Corr=" + std::to_string(corr_tmp[nm]) + ", Cost=" + std::to_string(cost_list[nm]) + "sec.) See technical manual for more information.";
			//theErrorFile.write(errMsg);
			return false;
		}
	}

	return true;
}

void 
runMFMC::setUpRes_rvStatistics(vector<vector<double>> xval) {

	if (procno == 0) {

		//this->xval = xval;
		//this->xstrval = xstrval;
		//this->gval = gmat;
		nmc = xval.size();
		nrv = xval[0].size();
        vector<double> rvMean, rvStdDev, rvSkewness, rvKurtosis;

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

			rvMean.push_back(mean_val);
			rvStdDev.push_back(stdDev_val);
			rvSkewness.push_back(skewness_val);
			rvKurtosis.push_back(kurtosis_val);
			if (nr > 100) {
				std::cout << "RVs from " << nr + 2 << " to " << nrv + 1 << " not displayed here for memory efficiency" << '\n';
				break;
			}
		}


        rvJson["rvNames"] = inp.rvNames;
        rvJson["mean"] = rvMean;
        rvJson["standardDeviation"] = rvStdDev;
        rvJson["skewness"] = rvSkewness;
        rvJson["kurtosis"] = rvKurtosis;
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

	for (int modelID : modelIDs) {
		double id_u = quantile(stdNorm, T.M[modelID].theDist->getCdf(modelNo));
		for (int ns = 0; ns < nsamp; ns++) {
			//xvals[ns][modelID] = nm + 1;
			uvals[ns][modelID] = id_u;
		}
		//uvals = T.X2U(nsamp, xvals);
	}
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
void runMFMC::writeOutputs(double elapsedTime)
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
		outJson["RV"] = rvJson;
		outJson["QoI"] = qoiJson;
        outJson["Info"] = infoJson;
		outJson["AnalysisTime_sec"] = elapsedTime;
        outJson["Log_transform"] = do_log_transform;

		outfile << outJson.dump(4) << std::endl;
	}
}


void runMFMC::writeTabOutputs()
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
		std::string multiModel = "MultiModel"; 

		Taboutfile.setf(std::ios::fixed, std::ios::floatfield); // set fixed floating format
		Taboutfile.precision(7); // for fixed format

		Taboutfile << "idx\t";
		for (int j = 0; j < inp.nrv + inp.nco + inp.nre; j++) {
			if (inp.rvNames[j].compare(0, multiModel.length(), multiModel) == 0) {
				//pass
			}
			else {
				Taboutfile << inp.rvNames[j] << "\t";
			}
		}
		for (int j = inp.nrv + inp.nco + inp.nre; j < inp.nrv + inp.nco + inp.nre + inp.nst; j++) {
				Taboutfile << inp.rvNames[j] << "\t";
		}
		for (int nm = 0; nm < numModels; nm++) {
			for (int j = 0; j < inp.nqoi; j++) {
				Taboutfile << inp.qoiNames[j] << "-M" << nm + 1 << "\t";
			}
		}
		Taboutfile << '\n';


		int nsamp = xvals[numModels - 1].size();
		for (int ns = 0; ns < nsamp; ns++) {
			Taboutfile << std::to_string(ns + 1) << "\t";
			for (int nr = 0; nr < inp.nrv + inp.nco + inp.nre; nr++) {
				if (inp.rvNames[nr].compare(0, multiModel.length(), multiModel) == 0) {
					//pass
				}
				else {
					Taboutfile << std::scientific << std::setprecision(7) << (xvals[numModels - 1][ns][nr]) << "\t";
				}
			}

			for (int nm = 0; nm < numModels; nm++) {
					for (int nq = 0; nq < inp.nqoi; nq++) {
						if (ns < gvals[nm].size()) {
							Taboutfile << std::scientific << std::setprecision(7) << (gvals[nm][ns][nq]) << "\t";
						} else {
							Taboutfile << std::scientific << std::setprecision(7) << "-" << "\t"; // assigning not a number
						}
					}
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
