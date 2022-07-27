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
 *  Exponential distribution class
 */


#include "exponentialDist.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "nlopt.hpp"

using std::vector;

//extern std::ofstream theErrorFile; // Error log
double nnlExponential(unsigned n, const double* x, double* grad, void* my_func_data);

exponentialDist::exponentialDist(string opt, vector<double> val, vector<double> add)
{
	name = "exponential";
	if (opt.compare("PAR") == 0)
	{
		if (val.size() != 1)
		{
			std::string errMSG = "Error running UQ engine: The " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMSG);
		}
		else if (val[0] <= 0)
		{
			std::string errMSG = "Error running UQ engine: The " + name + " distribution must be greater than 0";
			theErrorFile.write(errMSG);
		}
		else
		{
			lambda = val[0];
		}
	}
	else if (opt.compare("MOM") == 0)
	{

		if (val.size() != 1)
		{
			std::string errMSG = "Error running UQ engine: The " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMSG);
		}
		else if (val[0] <= 0)
		{
			std::string errMSG = "Error running UQ engine: mean of " + name + " distribution must be greater than 0";
			theErrorFile.write(errMSG);
		}
		else
		{
			lambda = 1/val[0];
		}

	}
	else if (opt.compare("DAT") == 0)
	{
		double minSmp = *std::min_element(std::begin(val), std::end(val));
		if (minSmp < 0)
		{
			std::string errMSG = "Error running UQ engine: samples of " + name + " distribution exceeds the range [0,inf]";
			theErrorFile.write(errMSG);
		}

		const int np = 1;
		//double lb[np] = { -HUGE_VAL, 0 };
		//const double  *lb = { 0 };

		// initial lambda
		double mu = 0.0, var = 0.0, ns=0, std;
		for(double value : val)
		{
			mu += value;
			//var += value*value;
			ns += 1;
		}
		mu = mu / ns;
		lambda = 1 / mu;
		vector<double> xs = val;

		// MLE optimization
		double lb[1] = { 1.e-10 };
		//double x[1] = { lambda };  //`*`some` `initial` `guess`*` 
		nlopt_opt optim;
		//optim = nlopt_create(NLOPT_LD_MMA, np); // gradient-based algorithm
		optim = nlopt_create(NLOPT_LN_COBYLA, np); // derivative-free algorithm
		nlopt_set_lower_bounds(optim, lb);
		//vector<double> xs = { 10.0, 20.0, 30.0 };
		nlopt_set_min_objective(optim, nnlExponential, &xs);
		nlopt_set_xtol_rel(optim, 1e-6);
		double minf; // `*`the` `minimum` `objective` `value,` `upon` `return`*` 

		if (nlopt_optimize(optim, &lambda, &minf) < 0) {
			printf("nlopt failed!\n");
			std::string errMSG = "Error running UQ engine: MLE optimization filed";
			theErrorFile.write(errMSG);
		}
		else {
			//lambda = x[0];
			printf("found minimum at f(%g) = %0.10g\n", lambda, minf);
		}

	}

	exponential expDist1(lambda);
	expDist = expDist1;
	checkParams();
}

exponentialDist::~exponentialDist() {}
//==


void exponentialDist::checkParams()
{
	double std = getStd();
	if (isnan(std) || isinf(std) || std <= 0)
	{
		std::string errMSG = "Error running UQ engine: stdandard deviation of " + name + " should be greater than 0 ";
		theErrorFile.write(errMSG);
	}
	double mean = getMean();
	if (mean <= 0)
	{
		std::string errMSG = "Error running UQ engine: mean of " + name + " distribution must be greater than 0 ";
		theErrorFile.write(errMSG);
	}
	vector<double> par = getParam();
	if (par[0] <= 0)
	{
		std::string errMSG = "Error running UQ engine: parameter of " + name + " distribution must be greater than 0 ";
		theErrorFile.write(errMSG);
	}
}


double exponentialDist::getPdf(double x)
{
	//double result;
	//if (0.0 < x) {
	//	result = lambda * exp(-lambda * x);
	//}
	//else {
	//	result = 0.0;
	//}
	//return result;
	return pdf(expDist,x);

}

double exponentialDist::getCdf(double x)
{
	/*
	double result;
	if (0.0 < x) {
		result = 1.0-exp(-(lambda) * x);
	}
	else {
		result = 0.0;
	}
	return result;
	*/
	return cdf(expDist, x);

}


double exponentialDist::getMean(void)
{
	//return moms[0];
	//double a =mean(expDist);
	return mean(expDist);
}

double exponentialDist::getStd(void)
{
	//return moms[1];
	return standard_deviation(expDist);
}

double exponentialDist::getQuantile(double p)
{

	return quantile(expDist, p);

}

vector<double> exponentialDist::getParam(void)
{
	return { lambda };
}


string exponentialDist::getName(void)
{
	return name;
}


double nnlExponential(unsigned n, const double* x, double* grad, void* my_func_data)
{
	double lamb = x[0];
	double nll=0;
	my_samples* samp = (my_samples*)my_func_data;
	vector<double> xs = samp->xs;

	for(int i=0;i< xs.size();i++)
	{
		nll += - log(lamb) - (-lamb * xs[i]);
	}
	if (grad) {
		grad[0] = 0;
	}
	return nll;
}



