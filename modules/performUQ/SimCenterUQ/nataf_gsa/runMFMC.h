#ifndef RUN_MFMC_H
#define RUN_MFMC_H

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

#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <set>
#include <algorithm>
#include <iostream>
#include <vector>
#include <numeric> // std::iota
#include <armadillo>
#include "jsonInput.h"
#include "ERANataf.h"
//extern std::ofstream theErrorFile;

#include "writeErrors.h"
extern writeErrors theErrorFile; // Error log

using std::vector;
using std::string;

class runMFMC
{
public:
	runMFMC(); 
	runMFMC(string workflowDriver,
		string osType,
		string runType,
		jsonInput inp,
		ERANataf T,
		int procno,
		int nproc);
	~runMFMC();
	void computeRvStatistics(vector<vector<double>> xval);
	void writeOutputs();
	void writeTabOutputs(jsonInput inp);

	//vector<double> Si;

	vector<vector<double>> xval;
	vector<vector<string>> xstrval;
	vector<vector<double>> gval;

private:
	void simulateMFMC(vector<int>Nsims, 
						int& numExistingDirs, 
						vector<vector<vector<double>>>& xvals_list, 
						vector<vector<vector<double>>>& gvals_list, 
						vector<double>& cost_list);
	void getOptimalSimNums(vector<vector<vector<double>>>xvals_list, 
						vector<vector<vector<double>>>gvals_list, 
						vector<double>cost_list, 
						bool updateNumSim,
						vector<double>& HF_est, 
						vector<int>& numSim_list, 
						vector<double>& Var_list);
	void updateModelIndex(int nm, ERANataf T, vector< vector<double>>& uvals);
	double calMean(vector<double> x);
	double calStd(vector<double> x, double m);
	double calVar(vector<double> x, double m);
	double calSkewness(vector<double> x, double m, double s);
	double calKurtosis(vector<double> x, double m, double s);
	double correlationCoef(vector<double> X, vector<double> Y);
	bool isInteger(double a);
	bool checkValidity(vector<double> cost_list, vector<double> corr_tmp, string &msg);
	vector<vector<vector<double>>> g2h(vector<vector<vector<double>>> hvals_pilot, bool do_mean_var, vector<double> perc_list);

	vector<double> rvMean, rvStdDev, rvSkewness, rvKurtosis;
	vector<double> gMean, gStdDev;
	vector<double> gMean_var, gStdDev_var;

	int modelID;
	int nproc;
	int procno;
	const double PI = atan(1) * 4;
	int nrv;
	int nmc;
	int numModels;
	jsonInput inp;
	ERANataf T;
	string workflowDriver;
	string osType;
	string runType;
	string optMultipleQoI;

	bool do_mean_var;
	//
	// Computational budget
	//

	double CB_init; // computational budget
	std::chrono::time_point<std::chrono::steady_clock> globalElapseStart;
};

#endif// RUN_MFMC_H
