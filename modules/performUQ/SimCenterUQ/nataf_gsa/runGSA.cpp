
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
 *  Runs global sensitivity analysis. See: Hu, Z. and Mahadevan, S. (2019). Probability models for data-driven global sensitivity analysis. Reliability Engineering & System Safety, 187, 40-57.
 */

#include "runGSA.h"
#include "ERANataf.h"
#include <iterator>
using namespace arma::newarp;

runGSA::runGSA() {}

runGSA::runGSA(string workflowDriver,
	string osType,
	string runType,
	jsonInput inp,
	ERANataf T,
	int procno,
	int nproc)
{

	//
	// Sample MCS
	//

	vector<vector<double>> xvals(inp.nmc, vector<double>(inp.nrv, 0.0));
	vector<vector<double>> gvals(inp.nmc, std::vector<double>(inp.nqoi, 0));	
	vector<vector<string>> discreteStrSamps(inp.nmc, vector<string>(inp.nst, ""));
	if (inp.uqMethod.compare("Monte Carlo") == 0) {

	vector<vector<double>> uvals(inp.nmc, vector<double>(inp.nrv, 0.0));
	vector<vector<int>> resampIDvals(inp.nmc, vector<int>(inp.nreg, 0.0));

	T.sample(inp.nmc, inp, procno, uvals, resampIDvals, discreteStrSamps);

	//
	// Simulate model
	//
	T.simulateAppBatch(workflowDriver, osType, runType, inp, uvals, resampIDvals, discreteStrSamps, 0, xvals, gvals, procno, nproc);

	}
	else if (inp.uqMethod.compare("Import Data Files") == 0) {

		if (runType.compare("runningLocal") != 0) {
			std::string errMsg = "Error running SimCenterUQ: No need to run remotely when the data set is provided. Please try running it locally.";
			theErrorFile.write(errMsg);
		}
		T.readDataset(inp.inpPath, inp.outPath, inp.nrv, inp.nqoi, inp.inpFileType, inp.outFileType, inp.nmc, xvals, gvals);

	} else {
		std::string errMsg = "Error running SimCenterUQ: UQ method " + inp.uqMethod + " unknown.";
		theErrorFile.write(errMsg);
	}

	this->xval = xvals;
	this->xstrval = discreteStrSamps;
	this->gmat = gvals;

	//
	// Read GSA Parameters
	//

	this->combs_tmp = inp.groups;
	this->PCAvarRatioThres = inp.PCAvarRatioThres;
	vector<vector<int>> qoiVectRange = inp.qoiVectRange;
	int Kos = 10;

	if (PCAvarRatioThres ==0.0) {
		this->performPCA = false;
	} else {
		this->performPCA = true;
	}    
	
	nmc = xval.size();
	nrv = xval[0].size();
	nqoi = gmat[0].size();
	ncombs = combs_tmp.size();

	vector<vector<double>> gmat_eff = gmat;
    vector<vector<double>> gmat_red;

	//if (nrv == 1) {
	//	vector<double> vect(nqoi, 1.0);
	//	Simat.push_back(vect);
	//	Stmat.push_back(vect);
	//	return;
	//}

	//
	// Preprocess gmat find a constant column
	//

	preprocess_gmat(gmat, gmat_eff); // Make it zero mean, get constants idx

	//
	// PCA process
	//

	//mat princ_dir_red;
    if (performPCA) {
		std::cout << "Running PCA ..." << std::endl;
		auto readStart = std::chrono::high_resolution_clock::now();
        runPCA(gmat_eff, gmat_red, princ_dir_red);
		auto readEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - readStart).count() / 1.e3;
		std::cout << " - Done PCA... Elapsed total time: " << readEnd  << "s\n";
    } else {
        //copy
        //for (int i = 0; i < gmat_eff.size(); i++) {
        //    gmat_red.push_back(gmat_eff[i]);
        //}
		std::cout << "Processing without PCA ..." << std::endl;

		for (int nq = constantQoiIdx.size() - 1; nq >= 0; nq--) {
			for (auto& row : gmat_eff) row.erase(next(row.begin(), constantQoiIdx[nq]));
		}
		gmat_red = gmat_eff;
		princ_dir_red.eye(nqoi, nqoi);
    }

    runMultipleGSA(gmat_red, Kos);

	//
	// Post processing
	//

	if (qoiVectRange.size()!=0){
		std::cout << "Calculating the aggregated GSA outputs for field QoIs ...\n";
		for (int nc = 0; nc < ncombs; nc++) {
			vector<double> Siagg_tmp, Stagg_tmp;

			for (int nu = 0; nu < qoiVectRange.size(); nu++) {
				double numerSi = 0, numerSt = 0, denom = 0;
				for (int nq = qoiVectRange[nu][0]; nq < qoiVectRange[nu][1]; nq++)
				{
					//if (std::find(constantQoiIdx.begin(), constantQoiIdx.end(), nq) != constantQoiIdx.end())
					//	continue;
					numerSi += varQoI[nq] * Simat[nc][nq];
					numerSt += varQoI[nq] * Stmat[nc][nq];
					denom += varQoI[nq];
				}
				if (denom == 0) denom = 1.0; // just for numerical stability. Si is anyways zero regardless.
				//if (numerSi > numerSt) numerSt = numerSi; // enforcing Si < St
				Siagg_tmp.push_back(numerSi / denom);
				Stagg_tmp.push_back(numerSt / denom);
			}
			Simatagg.push_back(Siagg_tmp);
			Stmatagg.push_back(Stagg_tmp);
		}
		std::cout << " - Done calculating the aggregated GSA ...\n";

	}
}

void runGSA::preprocess_gmat(vector<vector<double>> gmat, vector<vector<double>>& gmat_eff)
{
	// Make the matrix centered and have unit variance...

	std::cout << "Preprocessing.." << std::endl;

	std::vector<double> avg(nqoi, 0.0);
	std::vector<double> var(nqoi, 0.0);
	std::vector<double> normVar(nqoi, 0.0);
	for (std::vector<double>& row : gmat_eff)
	{
		std::transform(avg.begin(), avg.end(), row.begin(), avg.begin(), std::plus<double>()); // sum
		std::transform(var.begin(), var.end(), row.begin(), var.begin(), [](double a, double b) {return a + b*b; }); // square sum
	}
	const double scale(1/(double)nmc);
	std::transform(avg.begin(), avg.end(), avg.begin() ,[scale](double element) { return element *= scale; }); // avg of value
	std::transform(var.begin(), var.end(), var.begin(), [scale](double element) { return element *= scale; }); // avg of squared value

	// Final Variance
	std::transform(var.begin(), var.end(), avg.begin(), var.begin(), [](double a, double b) {return abs(a - b * b); });
	varQoI = var;

	// Normalized effective matrix
	for (auto& row : gmat_eff)
	{
		// zero mean
		std::transform(row.begin(), row.end(), avg.begin(), row.begin(), std::minus<double>()); 
		std::transform(row.begin(), row.end(), var.begin(), row.begin(), [](double a, double b) { return a / sqrt(b); });  // do not need to worry about constants yet
	}

	std::transform(var.begin(), var.end(), avg.begin(), normVar.begin(), [](double a, double b) {
		if ((b * b) == 0) {
			return a; // don't normalize..
		}
		else {
			return a/b/b; // normalize..
		}});// avg of square - square of avg
	

	std::cout << " - QoI now has zero mean  " << std::endl;
	std::cout << " - Checking if there are constant QoIs  " << std::endl;

	/*
	while ((it = std::find_if(it, var.end(), [](double x) {return abs(x)<1.e-15; })) != var.end())
	{
		constantQoiIdx.push_back(std::distance(var.begin(), it));
		std::cout << std::distance(var.begin(), it) << "\n";
		it++;
	}

	while ((it = std::find_if(it, var.end(), [](double x) {return abs(x) >= 1.e-15; })) != var.end())
	{
		nonConstantQoiIdx.push_back(std::distance(var.begin(), it));
		std::cout << std::distance(var.begin(), it) << "\n";
		it++;
	}
	*/

	int i = 0;
	for (auto it = normVar.begin(); it != normVar.end(); it++) {
		if (abs(*it) < 1.e-13) { // double precision 1.e-15
			constantQoiIdx.push_back(i);
		}
		else {
			nonConstantQoiIdx.push_back(i);
		}
		i++;
	} 
	std::cout << "  - Number of constant QoIs:  " << constantQoiIdx.size() << std::endl;
	std::cout << "  - Number of nonconstant QoIs:  " << nonConstantQoiIdx.size() << std::endl;

/*
	int count = 0;
	for (int nq = nqoi-1; nq >= 0; nq--) {
		bool isConstant = true;
		
		
		std::cout << gmat_eff[0][nq] << " " << gmat_eff][nq] << std::endl;
		int testcount = 0;
		for (int i = 0; i < nmc-1; i++)
		{ 
			testcount += 1;
			if (gmat_eff[i][nq] != gmat_eff[i + (int)1][nq]) {
				isConstant = false;
				break;
			}
			

		}
		
		//std::cout << isConstant << std::endl;

		if (isConstant) {
			std::cout << "IS CONSTANT" << std::endl;
			constantQoiIdx.push_back(nq);
			for (auto& row : gmat_eff) row.erase(next(row.begin(), nq));
			count++;
		}
*/
		/*
		if (gmat[1][nq] == gmat[0][nq]) {

			for (int i = 0; i < nmc; i++)
				mean += gmat[i][nq];

			mean = mean / double(nmc);
			for (int i = 0; i < nmc; i++)
				sqDiff += (gmat[i][nq] - mean) * (gmat[i][nq] - mean);

			//double var = sqDiff / nmc;
			if (sqDiff < 1.e-10) {
				constantQoiIdx.push_back(nq);
				// remove the column 
				for (auto& row : gmat_eff) row.erase(next(row.begin(), nq));
			}
			else {
				for (auto& row : gmat_eff) row[nq] = row[nq]- mean;
				count++;
			}

		}
		
		if (nq % 10000 == 0) {
			std::cout << "  - Current row is " << nq << " among " << nqoi << std::endl;
		}
	}
	*/
	nqoi_eff = nqoi- constantQoiIdx.size();

}

void runGSA::runMultipleGSA(vector<vector<double>> gmat_red, int Kos)
{
    int nqoi_red = gmat_red[0].size();

    int Kos_base_main = std::min(Kos, int(ceil(nmc / 20.0)));
    int Kos_base_total = std::min(Kos, int(ceil(nmc / 20.0)));

    //std::cout<<"Just testing this location 2\n";

	/*
    #ifdef MPI_RUN
        std::cout<<"sensitivity running MPI " << std::endl;

            //
            // MPI
            //

            int chunkSize = std::ceil(double(nqoi) / double(nprocs));
            //int lastChunk = inp.nmc - chunkSize * (nproc-1);
            double* SmAll = (double*)malloc(ncombs * chunkSize * nprocs * sizeof(double));
            double* SmTmp = (double*)malloc(ncombs * chunkSize * sizeof(double));
            double* StAll = (double*)malloc(ncombs * chunkSize * nprocs * sizeof(double));
            double* StTmp = (double*)malloc(ncombs * chunkSize * sizeof(double));
            // for each QoI
            //std::cout<<"Just testing this location 3 \n";
            for (int nq = 0; nq < chunkSize ; nq++) {
                int id = chunkSize * procno + nq;
                if (id >= nqoi) { // dummy
                    for (int i = 0; i < ncombs; i++) {
                        StTmp[nq * ncombs + i] = 0.;
                        SmTmp[nq * ncombs + i] = 0.;
                    }
                    continue;
                }

                vector<double> gvec;
                double sqDiff = 0;
                gvec.reserve(nmc);
                for (int i = 0; i < nmc; i++) {
                    gvec.push_back(gmat_red[i][id]);
                }

                // check if the variance is zero
                double mean = 0;
                for (int i = 0; i < nmc; i++)
                    mean += gvec[i];

                mean = mean / double(nmc);
                for (int i = 0; i < nmc; i++)
                    sqDiff += (gmat_red[i][id] - mean) * (gmat_red[i][id] - mean);

                //double var = sqDiff / nmc;
                if (sqDiff < 1.e-10) {
                    //theErrorFile << "Error running FEM: the variance of output is zero. Output value is " << mean;
                    //theErrorFile.close();
                    //exit(1);
                    //vector<double> zeros(ncombs, 0.0);
                    //Simat.push_back(zeros);
                    //Stmat.push_back(zeros);
                    //continue;
                    for (int i = 0; i < ncombs; i++) {
                        StTmp[nq * ncombs + i] = 0.;
                        SmTmp[nq * ncombs + i] = 0.;
                    }
                    continue;
                };

				vector<double> Sij, Stj;
				vector<vector<double>> Eij, Etj;

                runSingleGSA(gvec, Kos_base_main, 'M', Sij, Eij);
                runSingleGSA(gvec, Kos_base_total, 'T', Stj, Etj);

                if (Stj < Sij) {
                    Stj = Sij;
                }

                for (int i = 0; i < ncombs; i++) {
                    SmTmp[nq * ncombs + i] = Sij[i];
                    StTmp[nq * ncombs + i] = Stj[i];
                }
                //Simat.push_back(Stj);
                //Stmat.push_back(Sij);
            }
            MPI_Allgather(StTmp, ncombs * chunkSize, MPI_DOUBLE, StAll, ncombs * chunkSize, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Allgather(SmTmp, ncombs * chunkSize, MPI_DOUBLE, SmAll, ncombs * chunkSize, MPI_DOUBLE, MPI_COMM_WORLD);
            for (int i = 0; i < nqoi; i++) {
                vector<double> StVectmp(ncombs,0), SmVectmp(ncombs, 0);
                for (int j = 0; j < ncombs; j++) {
                    StVectmp[j] = StAll[i * ncombs + j];
                    SmVectmp[j] = SmAll[i * ncombs + j];
                }
                Stmat.push_back(StVectmp);
                Simat.push_back(SmVectmp);
            }

    #else
	*/
	/*
        std::cout<<"sensitivity running open MP " << std::endl;
        for (int j = 0; j < nqoi; j++) {

            vector<double> gvec;
            double sqDiff = 0;
            gvec.reserve(nmc);
            for (int i = 0; i < nmc; i++) {
                gvec.push_back(gmat_red[i][j]);
            }

            // check if the variance is zero
            double mean = 0;
            for (int i = 0; i < nmc; i++)
                mean += gvec[i];

            mean = mean / double(nmc);
            for (int i = 0; i < nmc; i++)
                sqDiff += (gmat_red[i][j] - mean) * (gmat_red[i][j] - mean);

            //double var = sqDiff / nmc;
            if (sqDiff < 1.e-10) {
                vector<double> zeros(ncombs, 0.0);
                Simat.push_back(zeros);
                Stmat.push_back(zeros);
                continue;
            };

            vector<double> Sij, Stj;
			vector<vector<double>> Eij, Etj;

            runSingleGSA(gvec, Kos, 'M', Sij, Eij);
            runSingleGSA(gvec, Kos, 'T', Stj, Etj);

            vector<double> Si_temp, Kos, St_temp;

            for (int nc = 0; nc < ncombs; nc++) {
                if (Stj[nc] < Sij[nc]) {
                    Stj[nc] = Sij[nc];
                }
            }
            Simat.push_back(Sij);
            Stmat.push_back(Stj);
        }
	*/
	//std::cout << "sensitivity running open MP " << std::endl;

	vector<vector<int>> combsM;
	combsM = combs_tmp;
	auto readStart = std::chrono::high_resolution_clock::now();
	double readEnd;

	for (int nc = 0; nc < combsM.size(); nc++) {

		// check if the variance is zero
		std::cout << "RV (combinations) " + std::to_string(nc + 1) + " among " + std::to_string(combsM.size()) << std::endl;
		vector<double> Sij, Stj;
		std::cout << " >> Main :" << std::endl;
		readStart = std::chrono::high_resolution_clock::now();
		runSingleCombGSA(gmat_red, Kos, combsM[nc], Sij, 'M');
		readEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - readStart).count() / 1.e3;
		std::cout << "  - Main took " << readEnd << " s\n";


		std::cout << " >> Total :" << std::endl;
		readStart = std::chrono::high_resolution_clock::now();
		runSingleCombGSA(gmat_red, Kos, combsM[nc], Stj, 'T');
		readEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - readStart).count() / 1.e3;
		std::cout << "  - Total took " << readEnd << " s\n";

		vector<double> Si_temp, Kos, St_temp;

		for (int nq = 0; nq < nqoi; nq++) {
			if (Stj[nq] < Sij[nq]) {
				Stj[nq] = Sij[nq];
			}
		}
		Simat.push_back(Sij);
		Stmat.push_back(Stj);
	}

	//vector<vector<double> > Simat_T(Simat[0].size(), vector<double>());
	//vector<vector<double> > Stmat_T(Simat[0].size(), vector<double>());

	//
	// Postprocess
	//
	//for (int nq = 0; nq < nqoi; nq++)
	//{
	//	if (std::find(constantQoiIdx.begin(), constantQoiIdx.end(), nq) != constantQoiIdx.end()) {
	//		for (int nc = 0; nc < Simat.size(); nc++)
	//		{
	//			Simat[nc].insert(Simat[nc].begin() + nq, 0.0);
	//			Stmat[nc].insert(Stmat[nc].begin() + nq, 0.0);
	//		}
	//	}
	//}

	/*
	for (int nc = 0; nc < Simat.size(); nc++)
	{
		for (int nq = 0; nq < Simat[nc].size(); nq++)
		{
			Simat_T[nq].push_back(Simat[nc][nq]);
			Stmat_T[nq].push_back(Stmat[nc][nq]);
		}
	}
	Simat = Simat_T;    // <--- reassign here
	Stmat = Stmat_T;    // <--- reassign here
	*/
	//#endif

}

void runGSA::runSingleCombGSA(vector<vector<double>> gmat, int Ko, vector<int> comb, vector<double>& Si, char Opt)
{
	//
	// we will ignore NaN in gvec
	//

	if (Opt == 'T') {
		vector<int> allSet(nrv);
		std::iota(allSet.begin(), allSet.end(), 0);
		vector<int> comb_new;
		std::set_difference(allSet.begin(), allSet.end(), comb.begin(), comb.end(), std::inserter(comb_new, comb_new.begin()));
		comb = comb_new;

		//
		// if we have discrete string variable, we don't do total..
		//

		if (xstrval.size()>0) {
			if (xstrval[0].size()>0) {
				Si.assign(nqoi, { sqrt(-1) }); // enforcing NaN
				return;
			}
		}		
	}

	int nqoi_red = gmat[0].size();

	//vector<double> Si;
	vector<vector<double>> Ei;
	vector<double> Si_tmp;
	vector<double> Var_tmp;
	Si_tmp.reserve(nqoi); // with zeros
	//Var_tmp.reserve(nqoi); // with zeros
	
	int	total_qoi_count = 0;
	int nconst = 0;

	for (int nq = 0; nq < nqoi_red; nq++) {
		vector<double> gvec;
		gvec.reserve(nmc);
		for (int i = 0; i < nmc; i++) {
			gvec.push_back(gmat[i][nq]);
		}


		int nmc_new = 0;
		for (int ns = 0; ns < nmc; ns++)
		{
			// Only if g is not NaN
			if (!std::isnan(gvec[ns])) {
				nmc_new++;
			}
		}

		double V = calVar(gvec);
		int Kos = Ko;

		const int endm = comb.size(); // (nx+ng)-1
		const int endx = endm - 1;			// (nx)-1
		if (endm == 0)
		{
			if (Opt == 'T')
			{
				Si.push_back(1.); // total
			}
			else
			{
				Si.push_back(0.);   // main
			}
			if (!performPCA){
			  printf("    GSA nq=%i, Si=%.2f, K=%i \n", nq + 1, Si[nq], Kos);
			} else { 
			  printf("    GSA PCA %i, Si=%.2f, K=%i \n", nq + 1, Si[nq], Kos);
			}
			continue;
		}
		/*
		else if (endm == nrv)
		{
			if (Opt == 'T')
			{
				Si.push_back(0.); // total
			}
			else
			{
				Si.push_back(1.);   // main
			}
			if (!performPCA) {
				printf("    GSA nq=%i, Si=%.2f, K=%i \n", nq + 1, Si[nq], Kos);
			}
			else {
				printf("    GSA PCA %i, Si=%.2f, K=%i \n", nq + 1, Si[nq], Kos);
			}
			continue;
		}
		*/
		mat data(endm + 1, nmc_new);

		int count_valid = 0;
		for (int ns = 0; ns < nmc; ns++)
		{
			// Only if g is not NaN
			if (!std::isnan(gvec[ns])) {
				data(endm, count_valid) = gvec[ns];
				count_valid++;
			}
		}

		for (int ne = 0; ne < endm; ne++)
		{
			int idx = comb[ne];

  			if (idx > nrv - 1) {
				std::string errMsg = "Error running UQ engine: combination index exceeds the bound";
				theErrorFile.write(errMsg);
			}
			count_valid = 0;
			for (int ns = 0; ns < nmc; ns++)
			{
				// Only if g is not NaN
				if (!std::isnan(gvec[ns])) {
					data(ne, count_valid) = xval[ns][idx];
					count_valid++;
				}
			}
		}

		gmm_full model;
		//bool status = model.learn(data, Kos, maha_dist, static_subset, 30, 100, V *1.e-3, false);
		double oldLogL = -INFINITY, logL;
		bool status;

		int Kthres;
		if (Opt == 'T')
		{
			Kthres = nmc_new / 100; // total
		}
		else
		{
			Kthres = nmc_new / 10;   // main
		}

		while (1) {
			status = model.learn(data, Kos, maha_dist, static_subset, 1000, 1000, V * 1.e-12, false);// max kmeans iter = 100, max EM iter = 200, convergence variance = V*1.e-15
			logL = model.sum_log_p(data);
			if ((logL < oldLogL) || (Kos >= Kthres)) {
				break;
			}
			else {
				oldLogL = logL;
				Kos = Kos + 1;
				//printf("increasing Ko to %i, ll=%.f3\n", Kos, logL);
			}
		}

		if (!performPCA) {
			printf("    GSA nq=%i, K=%i \n", nq + 1, Kos);
		}
		else {
			printf("    GSA PCA %i, K=%i \n", nq + 1, Kos);
		}

		if (status == false)
		{
			std::string errMsg = "Error running UQ engine: GSA learning failed";
			theErrorFile.write(errMsg);
		}

		if (Kos == 0)
		{
			std::string errMsg = "Error running UQ engine: GSA learning failed. Try with more number of samples.";
			theErrorFile.write(errMsg);
		}

		mat mu = model.means;   //nrv x Ko
		cube cov = model.fcovs; //nrv x nrv x Ko
		rowvec pi = model.hefts;   //1 x Ko 
		rowvec mug = mu.row(endm);    //1 x Ko

		vector<double> mui;
		mui.reserve(nmc_new);
		// Used to calculate conditional mean and covariance
		cube SiginvSig(1, endm, Kos);
		mat muk(endm, Kos);
		for (int k = 0; k < Kos; k++)
		{
			mat Sig12 = cov.subcube(0, endm, k, endx, endm, k);
			mat Sig11 = cov.subcube(0, 0, k, endx, endx, k);
			muk.col(k) = mu.submat(0, k, endx, k);
			SiginvSig.slice(k) = solve(Sig11, Sig12).t();
		}

		//model.means.print("means:");
		//model.fcovs.print("fcovs:");

		for (int i = 0; i < nmc_new; i++)
		{
			rowvec pik_tmp(Kos, fill::zeros);
			colvec muki(Kos);
			mat xi = data.submat(0, i, endx, i);

			for (int k = 0; k < Kos; k++)
			{
				mat tmp = SiginvSig.slice(k);
				mat muval = muk.col(k);
				muki.subvec(k, k) = mug(k) + SiginvSig.slice(k) * (xi - muval);
				pik_tmp(k) = pi(k) * mvnPdf(xi, muval, cov.subcube(0, 0, k, endx, endx, k));

			}

			rowvec piki = pik_tmp / sum(pik_tmp);
			mat tmp = piki * muki;
			mui.push_back(tmp(0, 0));
		}

		double var1 = 0, var2 = 0;
		for (int k = 0; k < Kos; k++)
		{
			mat Sig22 = cov.subcube(endm, endm, k, endm, endm, k);
			var1 = var1 + pi(k) * Sig22(0, 0) + pi(k) * mug(k) * mug(k);
			var2 = var2 + pi(k) * mug(k);
		}
		double V_approx = var1 - var2 * var2;

		if (!performPCA) {
			//while (std::find(constantQoiIdx.begin(), constantQoiIdx.end(), total_qoi_count) != constantQoiIdx.end())
			//{
			//	if (Opt == 'T') {
			//		Si_tmp.push_back(1.0);
			//	}
			//	else if (Opt == 'M') {
			//		Si_tmp.push_back(0.0);
			//	}
			//	Var_tmp.push_back(0.0);
			//	total_qoi_count++;
			//}
			if (nqoi != nqoi_eff) {
				while (constantQoiIdx[nconst] == total_qoi_count)
				{
					if (Opt == 'T') {
						Si_tmp.push_back(1.0);
					}
					else if (Opt == 'M') {
						Si_tmp.push_back(0.0);
					}
					Var_tmp.push_back(0.0);
					total_qoi_count++;
					nconst++;
				}
			}
			//In this case, (nqoi_eff == nqoi_red)
			Si_tmp.push_back(calVar(mui)/ V_approx);

			Var_tmp.push_back(V_approx);
			total_qoi_count++;
		}
		else {
			Ei.push_back(mui);
		}
				
	}

	if (performPCA) {
		std::cout << "  - Converting PCA sobol incides (" << nqoi_red << ") to original domain values (" << nqoi << ")...\n";

		auto readStart = std::chrono::high_resolution_clock::now();


		total_qoi_count=0;

		mat Sigmaij(nqoi_red, nqoi_red);
		for (int nq1 = 0; nq1 < nqoi_red; nq1++) {
			for (int nq2 = 0; nq2 < nqoi_red; nq2++) {
				if (nq1 >= nq2) {
					Sigmaij(nq1, nq2) = calCov(Ei[nq1], Ei[nq2]);
					Sigmaij(nq2, nq1) = Sigmaij(nq1, nq2);

				}
			}
		}

		nconst = 0;
		for (int nq = 0; nq < nqoi_eff; nq++) {

			//while (std::find(constantQoiIdx.begin(), constantQoiIdx.end(), total_qoi_count) != constantQoiIdx.end())
			//{
			//	if (Opt == 'T') {
			//		Si_tmp.push_back(1.0);
			//	} else if (Opt == 'M') {
			//		Si_tmp.push_back(0.0);
			//	}
			//	Var_tmp.push_back(0.0);
			//	total_qoi_count++;
			//}

			if (nqoi!=nqoi_eff) {
				while (constantQoiIdx[nconst]==total_qoi_count)
				{
				if (Opt == 'T') {
					Si_tmp.push_back(1.0);
				} else if (Opt == 'M') {
					Si_tmp.push_back(0.0);
				}
				Var_tmp.push_back(0.0);
				total_qoi_count++;
				nconst++;
				}
			}
			rowvec aa = princ_dir_red.row(nq);
			double Vi = (aa * Sigmaij * trans(aa)).eval()(0, 0);
			//double Vi = 0.1;
			double V = sum(aa % trans(lambs_red) % aa);
			//double V = 0.1;
			Si_tmp.push_back(Vi / V);
            //std::cout<<"flag==========================="<<std::endl;
            //std::cout<<Vi<<std::endl;
            //std::cout<<V<<std::endl;
			Var_tmp.push_back(V);
			total_qoi_count++;
		}

		auto readEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - readStart).count() / 1.e3;
		std::cout << "  - Done conversion elapsed time " << readEnd << " s\n";
		std::cout << "  - QoI count after reconstruction: " << total_qoi_count << " \n";
	}

	if (total_qoi_count < nqoi) {
		while (nconst < constantQoiIdx.size())
		{
			if (Opt == 'T') {
				Si_tmp.push_back(1.0);
			}
			else if (Opt == 'M') {
				Si_tmp.push_back(0.0);
			}
			Var_tmp.push_back(0.0);
			total_qoi_count++;
			nconst++;
		}
	}

	//Si.push_back(Vi / V);
	if (Opt == 'T')
	{
		std::for_each(Si_tmp.begin(), Si_tmp.end(), [](double& S) { S = 1.0 - S; });
	}
	Si= Si_tmp;   
	//varQoI = Var_tmp;

	for (int nq = 0; nq < nqoi_eff; nq++) {
		//printf("GSA nq=%i, Si=%.2f, %c \n", nq + 1, Si[nq], Opt);
		if (isinf(Si[nq]) || isnan(Si[nq]))
		{
			Si[nq] = -100;
		}
	}

}



void runGSA::runSingleGSA(vector<double> gvec,int Ko,char Opt, vector<double>& Si, vector<vector<double>>& Ei)
{
    //
    // we will ignore NaN in gvec
    //

    int nmc_new = 0;
    for (int ns = 0; ns < nmc; ns++)
    {
        // Only if g is not NaN
        if (!std::isnan(gvec[ns])) {
            nmc_new++;
        }
    }

	vector<vector<int>> combs;

	if (Opt == 'T')
	{
		vector<int> allSet(nrv);
		std::iota(allSet.begin(), allSet.end(), 0);

		for (auto comb : combs_tmp)
		{
			vector<int> cnc;
			//std::set_difference(allSet.begin(), allSet.end(), comb.begin(), comb.end(), cnc.begin());
			std::set_difference(allSet.begin(), allSet.end(), comb.begin(), comb.end(), std::inserter(cnc, cnc.begin()));
			combs.push_back(cnc);
		}
	}
	else
	{
		combs = combs_tmp;
	}

	double V = calVar(gvec);
	double Vi;
	//vector<double> Si;
	Si.reserve(ncombs);

	for (int nc = 0; nc < ncombs; nc++)
	{
		int Kos = Ko;

		const int endm = combs[nc].size(); // (nx+ng)-1
		const int endx = endm - 1;			// (nx)-1
		if (endm == 0)
		{
			if (Opt == 'T')
			{
				Si.push_back(1.); // total
			}
			else
			{
				Si.push_back(0.);   // main
			}
			printf("GSA i=%i, Si=%.2f, K=%i \n", nc + 1, Si[nc], Kos);
			continue;
		}
		else if (endm == nrv)
		{
			if (Opt == 'T')
			{
				Si.push_back(0.); // total
			}
			else
			{
				Si.push_back(1.);   // main
			}
			printf("GSA i=%i, Si=%.2f, K=%i \n", nc + 1, Si[nc], Kos);
			continue;
		}

		mat data(endm + 1, nmc_new);


        int count_valid = 0;
        for (int ns = 0; ns < nmc; ns++)
        {
            // Only if g is not NaN
            if (!std::isnan(gvec[ns])) {
                data(endm, count_valid) = gvec[ns];
                count_valid++;
            }
        }

		for (int ne = 0; ne < endm; ne++)
		{
			int idx = combs[nc][ne];

			if (idx > nrv - 1) {
				std::string errMsg = "Error running UQ engine: combination index exceeds the bound";
				theErrorFile.write(errMsg);
			}
            count_valid = 0;
			for (int ns = 0; ns < nmc; ns++)
			{
                // Only if g is not NaN
                if (!std::isnan(gvec[ns])) {
                    data(ne, count_valid) = xval[ns][idx];
                    count_valid++;
                }
			}
        }

		gmm_full model;
		//bool status = model.learn(data, Kos, maha_dist, static_subset, 30, 100, V *1.e-3, false);
		double oldLogL = -INFINITY, logL;
		bool status;


		int Kthres;
		if (Opt == 'T')
		{
			Kthres = nmc_new / 100; // total
		}
		else
		{
			Kthres = nmc_new / 10;   // main
		}

		while (1) {
			status = model.learn(data, Kos, maha_dist, static_subset, 500, 500, V * 1.e-15, false);// max kmeans iter = 100, max EM iter = 200, convergence variance = V*1.e-15
			logL = model.sum_log_p(data);
			if ((logL < oldLogL) || (Kos >= Kthres)) {
				break;
			}
			else {
				oldLogL = logL;
				Kos = Kos + 1;
				//printf("increasing Ko to %i, ll=%.f3\n", Kos, logL);
			}
		}
		printf("FINAL Ko = %i \n", Kos);


		if (status == false)
		{
			std::string errMsg = "Error running UQ engine: GSA learning failed";
			theErrorFile.write(errMsg);
		}

		if (Kos == 0)
		{
			std::string errMsg ="Error running UQ engine: GSA learning failed. Try with more number of samples.";
			theErrorFile.write(errMsg);
		}

		mat mu = model.means;   //nrv x Ko
		cube cov = model.fcovs; //nrv x nrv x Ko
		rowvec pi = model.hefts;   //1 x Ko 
		rowvec mug = mu.row(endm);    //1 x Ko

		vector<double> mui;
		mui.reserve(nmc_new);
		// Used to calculate conditional mean and covariance
		cube SiginvSig(1, endm, Kos);
		mat muk(endm, Kos);
		for (int k = 0; k < Kos; k++)
		{
			mat Sig12 = cov.subcube(0, endm, k, endx, endm, k);
			mat Sig11 = cov.subcube(0, 0, k, endx, endx, k);
			muk.col(k) = mu.submat(0, k, endx, k);
			SiginvSig.slice(k) = solve(Sig11, Sig12).t();
		}

		//model.means.print("means:");
		//model.fcovs.print("fcovs:");

		for (int i = 0; i < nmc_new; i++)
		{
			rowvec pik_tmp(Kos, fill::zeros);
			colvec muki(Kos);
			mat xi = data.submat(0, i, endx, i);

			for (int k = 0; k < Kos; k++)
			{
				mat tmp = SiginvSig.slice(k);
				mat muval = muk.col(k);
				muki.subvec(k, k) = mug(k) + SiginvSig.slice(k) * (xi - muval);
				pik_tmp(k) = pi(k) * mvnPdf(xi, muval, cov.subcube(0, 0, k, endx, endx, k));

			}

			rowvec piki = pik_tmp / sum(pik_tmp);
			mat tmp = piki * muki;
			mui.push_back(tmp(0, 0));
		}

		double var1 = 0, var2 = 0;
		for (int k = 0; k < Kos; k++)
		{
			mat Sig22 = cov.subcube(endm, endm, k, endm, endm, k);
			var1 = var1 + pi(k) * Sig22(0, 0) + pi(k) * mug(k) * mug(k);
			var2 = var2 + pi(k) * mug(k);
		}
		double V_approx = var1 - var2 * var2;

		Vi = calVar(mui);
		//Si.push_back(Vi / V);

		if (Opt == 'T')
		{
			Si.push_back(1 - Vi / V_approx); // total
		}
		else
		{
			Si.push_back(Vi / V_approx);   // main
		}

		printf("GSA i=%i, Si=%.2f, K=%i, %c \n", nc + 1, Si[nc], Kos, Opt);

		if (isinf(Si[nc]) || isnan(Si[nc]))
		{
			Si[nc] = -100;
		}

		Ei.push_back(mui);
	}
}

void runGSA::runPCA(vector<vector<double>> gmat, vector<vector<double>>& gmat_red, mat& princ_dir_red) {

    mat U_matrix;
    vec svec;
    mat V_matrix;

    int n = gmat.size();
	int p = nqoi_eff;

    mat gmat_matrix(n, p);

	arma::uvec idx(nonConstantQoiIdx.size());
	for (int nqe = 0; nqe < nonConstantQoiIdx.size(); nqe++) {
		idx(nqe) = nonConstantQoiIdx[nqe];
	}

    for (int nr = 0; nr < n; nr++)
    {
		arma::vec r(gmat[nr]);
		gmat_matrix.row(nr) = r.elem(idx).t();
    }

	//
	// run SVD
	//
	auto readStart = std::chrono::high_resolution_clock::now();
	auto readEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - readStart).count() / 1.e3;
	
	if (n>p) {
		//
		// run SVD - took 386 s
		//
		std::cout << " - Running SVD\n";
		readStart = std::chrono::high_resolution_clock::now();
		svd_econ(U_matrix, svec, V_matrix, gmat_matrix); // gmat_matrix = U*L*V^T
		readEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - readStart).count() / 1.e3;
		std::cout << "  - SVD took " << readEnd << " s\n";
		//std::cout << "lambdas are " << std::endl << pow(svec, 2) << std::endl;
	} else {
		//
		// run PCA -- gives memory error
		//
		//mat coeff, score;
		//vec latent, tsquared;
		//readStart = std::chrono::high_resolution_clock::now();
		//princomp(coeff, score, latent, tsquared, gmat_matrix);
		//readEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - readStart).count() / 1.e3;
		//std::cout << "  - PCA took " << readEnd << " s\n";

		//
		// Create gram - took 17 s
		//
		std::cout << " - Creating gram matrix\n";
		readStart = std::chrono::high_resolution_clock::now();
		mat gramMat = gmat_matrix * trans(gmat_matrix);
		readEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - readStart).count() / 1.e3;
		std::cout << "  - Creating gram matrix - done. It took " << readEnd << " s\n";

		//
		// FULL EIGEN - took 0.1 s
		//

		std::cout << " - Full eigen using Lapack2\n";
		readStart = std::chrono::high_resolution_clock::now();
		vec Lvece;
		eig_sym(Lvece, U_matrix, gramMat, "dc"); // gramMat = U*L*U^T
		//std::cout << "lambda is " << Lvece << std::endl;
		svec = sqrt(reverse(Lvece)); // because eigenvalues are ascending order
		U_matrix = fliplr((U_matrix)); // because eigenvalues are ascending order
		svec.replace(datum::nan, min(svec));

		mat invSmat = arma::diagmat(1 / svec);
		//V_matrix2 = (invSmat * U_matrix2.t() * gmat_matrix).t();
		V_matrix = (gmat_matrix.t() * U_matrix * invSmat); // faster
		readEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - readStart).count() / 1.e3;
		std::cout << "  - Full eigen using Lapack2 - done. It took " << readEnd << " s\n";


		/*
		std::cout << "U " << std::endl;
		std::cout << U_matrix.n_cols << std::endl;// 500
		std::cout << U_matrix.n_rows << std::endl;// 500

		std::cout << "V " << std::endl;
		std::cout << V_matrix.n_cols << std::endl;// 500
		std::cout << V_matrix.n_rows << std::endl;// 1558663

		
		mat tmp = U_matrix * arma::diagmat(svec) * V_matrix.t();

		std::cout << "G " << std::endl;

		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 4; j++) {
				std::cout << tmp(i, j) << " " << gmat_matrix(i, j) << std::endl;
			}
		}
		*/
	}

	/*
	std::cout << "sqrt lambda "<< std::endl;

	for (int i = 0; i < 10; i++) {
			std::cout << svec(i) << " " << svec2(i) << std::endl;
	}	
	
	std::cout << "U " << std::endl;

	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			std::cout << U_matrix(i,j) << " " << U_matrix2(i, j) << std::endl;
		}
	}

	std::cout << "V " << std::endl;

	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			std::cout << V_matrix(i,j) << " " << V_matrix2(i, j) << std::endl;
		}
	}
	*/

	//
	// Partial EIGEN - no decrease in time
	//
	/*
	std::cout << " - Partial eigen using Arpack\n";
	readStart = std::chrono::high_resolution_clock::now();
	// Construct matrix operation object using the wrapper class DenseGenMatProd
	DenseGenMatProd<double> op(gramMat);

	// Construct eigen solver object, requesting the largest three eigenvalues
	SymEigsSolver< double, EigsSelect::LARGEST_ALGE, DenseGenMatProd<double> > eigs(op, 3, 6);

	// Initialize and compute
	eigs.init();
	int nconv = eigs.compute();

	// Retrieve results
	arma::vec evalues;
	if (nconv > 0)
		evalues = eigs.eigenvalues();

	evalues.print("Eigenvalues found:");

	std::cout << "   - Partial eigen using Arpach - done. It took " << readEnd << " s\n";
	*/

	//
	// Eigen SVD function took 562 sec
	//
	/*
	Eigen::MatrixXd gmat_matrix2 = Eigen::MatrixXd(n, p);

	for (int nr = 0; nr < n; nr++)
	{
		
		for (int nc = 0; nc < p; nc++)
		{
			gmat_matrix2(nr, nc) = gmat_matrix(nr,nc);
		}
	}

	auto readStart = std::chrono::high_resolution_clock::now();
	std::cout << " - Running SVD\n";
	Eigen::BDCSVD<Eigen::MatrixXd> SVD(gmat_matrix2, Eigen::ComputeThinU | Eigen::ComputeThinV);
	auto readEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - readStart).count() / 1.e3;
	std::cout << "  - SVD took " << readEnd << " s\n";
	*/


	mat princ_dir, princ_comp;
	int neigen = std::min(n,p);
	princ_dir = V_matrix;       // projection matrix
//princ_comp = U_matrix.cols(0, p - 1) * arma::diagmat(svec); // reduced variables
	princ_comp = U_matrix * arma::diagmat(svec);;

	double sum_var = 0;
	//double totVar = sum(trace(C));
	vec lamb = pow(svec, 2);
	double totVar = sum(lamb);

	for (int i = 0; i < neigen; i++) {
		sum_var = sum_var + lamb[i] / totVar;
		if (sum_var > PCAvarRatioThres) {
			npc = i+1;
			PCAvarRatio = sum_var;
			break;
		}
	}

	std::cout << " - Number of the final PC components are " << npc << " to capture " << PCAvarRatioThres*100 << "% of variance" << std::endl;

	princ_dir_red = princ_dir.cols(0, npc - 1); // projection matrix
	mat princ_comp_red = princ_comp.cols(0, npc - 1);           // reduced variables

	for (size_t i = 0; i < princ_comp_red.n_rows; ++i) {
		gmat_red.push_back(arma::conv_to< vector<double> >::from(princ_comp_red.row(i)));
	};
	lambs_red = pow(svec.rows(0, npc - 1), 2) / nmc;
	/*
	//std::cout << "print this first" << std::endl;
	//std::cout << princ_comp_red* trans(princ_dir_red) << std::endl;
	//std::cout << "print this second" << std::endl;
	//std::cout << gmat_matrix << std::endl;


	mat V_matrixe;
	vec Lvece;
	mat C = gmat_matrix * trans(gmat_matrix);
	eig_sym(Lvece, V_matrixe, C, "dc");

	std::cout << ">>eigenvalue analysis" << std::endl;
	std::cout << "total variance is " << trace(C) << std::endl;
	std::cout << "normalized lambdas are " << std::endl  << pow(svec, 2) / sum(trace(C)) << std::endl;
	std::cout << "sum of lambdas " << sum(pow(svec, 2) / sum(trace(C))) << std::endl;


	//std::cout << "singularvalue decomposition" << std::endl;
	//std::cout << "lambda is " << Lvece << std::endl;
	//std::cout << "normalized lambda is " << Lvece / sum(trace(C)) << std::endl;
	
	*/
}

double runGSA::mvnPdf(mat x, mat mu, mat cov) 
{
	
	double n = size(x)(1);
	double sqrt2pi = std::sqrt(2 * PI);
	mat xmu = x - mu;
	mat quadform = xmu.t() * inv(cov) * xmu;
	double norm = std::pow(sqrt2pi, -n)*std::pow(abs(det(cov)), -0.5);
	//std::cout << norm << std::endl;

	return norm * std::exp(-0.5 * quadform(0,0));
}

double runGSA::calMean(vector<double> x) {
	double sum = std::accumulate(std::begin(x), std::end(x), 0.0);
	return sum / x.size();

}

runGSA::~runGSA() {};

double runGSA::calVar(vector<double> x) {
	double m = calMean(x);
	double accum = 0.0;
    int count = 0;
	std::for_each(std::begin(x), std::end(x), [&](const double d) {
            if (!std::isnan(d)) {
		        accum += (d - m) * (d - m);
                count++;
            }
		});
	//std::cout << (accum / (x.size())) << std::endl;
	return (accum / count);
}

double runGSA::calCov(vector<double> x1, vector<double> x2) {
	double m1 = calMean(x1);
	double m2 = calMean(x2);
	int N = x1.size();
	double accum = 0.0;
	int count = 0;
	for (int i = 0; i < N; i++) {
		if (!std::isnan(x1[i]) && !std::isnan(x2[i]))  {
			accum += (x1[i] - m1) * (x2[i] - m2);
			count++;
		}
	}

	//std::cout << (accum / (x.size())) << std::endl;
	return (accum / count);
}

/*
void runGSA::writeTabOutputs(jsonInput inp, int procno)
{
	if (procno==0) {
		// dakotaTab.out
		std::string writingloc1 = inp.workDir + "/dakotaTab.out";
		std::ofstream Taboutfile(writingloc1);

		if (!Taboutfile.is_open()) {

			std::string errMsg = "Error running UQ engine: Unable to write dakotaTab.out";
			theErrorFile.write(errMsg);
		}

		Taboutfile.setf(std::ios::fixed, std::ios::floatfield); // set fixed floating format
		Taboutfile.precision(10); // for fixed format

		Taboutfile << "idx         ";
		for (int j = 0; j < inp.nrv + inp.nco + inp.nre; j++) {
			Taboutfile << inp.rvNames[j] << "           ";
		}
		for (int j = 0; j < inp.nqoi; j++) {
			Taboutfile << inp.qoiNames[j] << "            ";
		}
		Taboutfile << std::endl;


		for (int i = 0; i < inp.nmc; i++) {
			Taboutfile << std::to_string(i + 1) << "    ";
			for (int j = 0; j < inp.nrv + inp.nco + inp.nre; j++) {
				Taboutfile << std::to_string(xval[i][j]) << "    ";
			}
			for (int j = 0; j < inp.nqoi; j++) {
				Taboutfile << std::to_string(gmat[i][j]) << "    ";
			}
			Taboutfile << std::endl;
		} 
	}
}
*/

void runGSA::writeOutputs(jsonInput inp, double dur, int procno)
{
	if (procno == 0) {
		std::cout << "Writing global sensitivity analysis outputs ...\n";
		auto readStart = std::chrono::high_resolution_clock::now();
		double readEnd;

		// dakota.out
		string writingloc = inp.workDir + "/dakota.out";
		std::ofstream outfile(writingloc);

		if (!outfile.is_open()) {

			std::string errMsg = "Error running UQ engine: Unable to write dakota.out";
			theErrorFile.write(errMsg);

		}

		outfile.setf(std::ios::fixed, std::ios::floatfield); // set fixed floating format
		outfile.precision(8); // for fixed format

		outfile << "* data generation" << std::endl;
		outfile << inp.uqMethod << std::endl;
		if (inp.uqMethod.compare("Import Data Files")==0) {
			outfile << inp.inpPath << std::endl;
			outfile << inp.outPath << std::endl;
		}

		outfile << "* number of input combinations" << std::endl;
		outfile << inp.ngr << std::endl;

		outfile << "* input names" << std::endl;
		for (int i = 0; i < inp.ngr; i++) {
			for (int j = 0; j < inp.groups[i].size() - 1; j++) {
				outfile << inp.rvNames[inp.groups[i][j]] << ",";
			}
			outfile << inp.rvNames[inp.groups[i][inp.groups[i].size() - 1]] << std::endl;
		}


		outfile << "* number of aggregated outputs" << std::endl;
		outfile << inp.nqoiVects << std::endl;

		outfile << "* number of outputs" << std::endl;
		outfile << inp.nqoi << std::endl;

		outfile << "* output names" << std::endl;
		for (int nu = 0; nu < inp.nqoiVects; nu++) {
			outfile << "[aggregated]" << inp.qoiVectNames[nu] << std::endl;
		}
		for (int i = 0; i < inp.nqoi; i++) {
			outfile << inp.qoiNames[i] << std::endl;
		}

		outfile << "* ";
		for (int j = 0; j < inp.ngr; j++) {
			//outfile << "Sm(" << std::to_string(j + 1) << ")  ";
			outfile << "Sm(";
			for (int k = 0; k < inp.groups[j].size() - 1; k++) {
				outfile << inp.rvNames[inp.groups[j][k]] << ",";
			}
			outfile << inp.rvNames[inp.groups[j][inp.groups[j].size() - 1]] << ") ";
		}
		for (int j = 0; j < inp.ngr; j++) {
			//outfile << "St(" << std::to_string(j + 1) << ")  ";
			outfile << "St(";
			for (int k = 0; k < inp.groups[j].size() - 1; k++) {
				outfile << inp.rvNames[inp.groups[j][k]] << ",";
			}
			outfile << inp.rvNames[inp.groups[j][inp.groups[j].size() - 1]] << ") ";
		}
		outfile << std::endl;
		
		for (int i = 0; i < inp.nqoiVects; i++) {
			for (int j = 0; j < inp.ngr; j++) {
				outfile << Simatagg[j][i] << " ";
			}
			for (int j = 0; j < inp.ngr; j++) {
				outfile << Stmatagg[j][i] << " ";
			}
			outfile << std::endl;
		}




		// let us display writing status if the file size is big
		auto dispInterv = 1.e7;
		int dispCount = 1;
		int thres = dispInterv * dispCount / inp.nrv * 2;

		for (int i = 0; i < inp.nqoi; i++) {

			for (int j = 0; j < inp.ngr; j++) {
				outfile << Simat[j][i] << " ";
			}
			for (int j = 0; j < inp.ngr; j++) {
				outfile << Stmat[j][i] << " ";
			}
			outfile << std::endl;

			if (i > thres) {
				std::cout << " - Writing output file in progress: " << (double)i / (double)inp.nqoi * 100 << "% \n";
				dispCount += 1;
				if (dispCount == 1) {
					readEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - readStart).count() / 1.e3; // before Si is trivial
					double expWriteTime = readEnd * (double)inp.nqoi / (double)1;
					if (expWriteTime > 5.0) {
						if (expWriteTime > 60.0) {
							expWriteTime = expWriteTime / 60.0;
						}
						std::cout << "  - Expected writing time: " << expWriteTime << " min\n";

					}
				}
				thres = dispInterv * dispCount / inp.nrv * 2;
			}
		}

		outfile << "* number of samples" << std::endl;
		outfile << inp.nmc << std::endl;

		outfile << "* elapsed time:" << std::endl;
		outfile << dur << " s" << std::endl;

		if (performPCA) {
			outfile << "* PCA" << std::endl;
			outfile << "yes" << std::endl;

			outfile << "* number of PCA components" << std::endl;
			outfile << npc << std::endl;

			outfile << "* proportion of variance explained by PCA" << std::endl;
			outfile << PCAvarRatio << std::endl;

		}
		else {
			outfile << "* PCA" << std::endl;
			outfile << "no" << std::endl;
		}

		outfile.close();

		readEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - readStart).count() / 1.e3; // before Si is trivial
		std::cout << " - Done writing output files: " << readEnd << " sec\n";

	}

}


void runGSA::writeTabOutputs(jsonInput inp, int procno)
{
	if (procno == 0) {
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
				Taboutfile << std::scientific << std::setprecision(7) << (gmat[ns][nq]) << "\t";
				//Taboutfile << std::to_string(gval[ns][nq]) << "\t";
			}
			Taboutfile << '\n';

			if (ns * inp.nqoi > dispInterv* dispCount) {
				std::cout << "  - Writing Tab file in progress: " << (double)ns / (double)inp.nmc * 100 << "% \n";
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


bool runGSA::isInteger(double a) {
	double b = round(a), epsilon = 1e-9; //some small range of error
	return (a <= b + epsilon && a >= b - epsilon);
}
