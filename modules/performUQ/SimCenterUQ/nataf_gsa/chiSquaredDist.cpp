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
 *  Chi-squared distribution class
 */

#include "chiSquaredDist.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "nlopt.hpp"
#include "boost/math/distributions/normal.hpp" // for normal_distribution

using std::vector;

double nnlChiSq(unsigned n, const double* x, double* grad, void* my_func_data);

chiSquaredDist::chiSquaredDist(string opt, vector<double> val, vector<double> add): chiSqDist(1.0)
{
	name = "chisquared";
	int npa = 1;
	if (opt.compare("PAR") == 0)
	{
		if (val.size() != npa)
		{
			std::string errMSG = "Error running UQ engine: The " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMSG);
		}
		else if ((val[0] <= 0) || (val[0]-floor(val[0])>1.e-10)) // not integer
		{
			std::string errMSG = "Error running UQ engine: parameter of  " + name + " distribution must a positive integer ";
			theErrorFile.write(errMSG);
		}
		else
		{
			k = val[0];
		}
	}
	else if (opt.compare("MOM") == 0)
	{

		if (val.size() != npa)
		{
			std::string errMSG = "Error running UQ engine: The " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMSG);
		}
		else if ((val[0] <= 0) || (val[0] - floor(val[0]) > 1.e-10)) // not integer
		{
			std::string errMSG = "Error running UQ engine: mean of " + name + " distribution must a positive integer ";
			theErrorFile.write(errMSG);
		}
		else
		{
			k = val[0];
		}

	}
	else if (opt.compare("DAT") == 0)
	{
		std::string errMSG = "The Chisquare distribution is not supported in DATA input type";
		theErrorFile.write(errMSG);
		/*
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
		k = mu;

		vector<double> xs = val;

		// MLE optimization
		double lb[1] = { 1.e-10 };
		nlopt_opt optim;
		optim = nlopt_create(NLOPT_LN_COBYLA, np); // derivative-free algorithm
		nlopt_set_lower_bounds(optim, lb);
		nlopt_set_min_objective(optim, nnlChiSq, &xs);
		nlopt_set_xtol_rel(optim, 1e-6);
		double minf; 

		if (nlopt_optimize(optim, &k, &minf) < 0) {
			printf("nlopt failed!\n");
			theErrorFile << "Error running UQ engine: MLE optimization filed" << std::endl;
			theErrorFile.close();
			exit(1);
		}
		else {
			//lambda = x[0];
			printf("found minimum at f(%g) = %0.10g\n", k, minf);
		}
		*/
	}

	chi_squared chiSqDist1(k);
	chiSqDist = chiSqDist1;
	checkParams();

}

chiSquaredDist::~chiSquaredDist() {}
//==

void chiSquaredDist::checkParams()
{
	double std = getStd();
	double mean = getMean();
	vector<double> par = getParam();

	if (isnan(std) || isinf(std) || std <= 0)
	{
		std::string errMSG = "Error running UQ engine: stdandard deviation of " + name + " must be greater than 0 ";
		theErrorFile.write(errMSG);
	}

	if ((mean <= 0) || (mean - floor(mean) > 1.e-10)) // not integer
	{
		std::string errMSG = "Error running UQ engine: mean of "  + name + " must a positive integer ";
		theErrorFile.write(errMSG);
	}

	if ((par[0] <= 0) || (par[0] - floor(par[0]) > 1.e-10)) // not integer
	{
		std::string errMSG = "Error running UQ engine: mean of " + name + " must a positive integer ";
		theErrorFile.write(errMSG);
	}
}


double chiSquaredDist::getPdf(double x)
{
	//double result;
	//if (0.0 < x) {
	//	result = lambda * chiSq(-lambda * x);
	//}
	//else {
	//	result = 0.0;
	//}
	//return result;
	return pdf(chiSqDist,x);

}

double chiSquaredDist::getCdf(double x)
{
	/*
	double result;
	if (0.0 < x) {
		result = 1.0-chiSq(-(lambda) * x);
	}
	else {
		result = 0.0;
	}
	return result;
	*/
	return cdf(chiSqDist, x);

}


double chiSquaredDist::getMean(void)
{
	//return moms[0];
	double a =mean(chiSqDist);
	return mean(chiSqDist);
}

double chiSquaredDist::getStd(void)
{
	//return moms[1];
	return standard_deviation(chiSqDist);
}

double chiSquaredDist::getQuantile(double p)
{

	return quantile(chiSqDist, p);

}

vector<double> chiSquaredDist::getParam(void)
{
	return { k };
}


string chiSquaredDist::getName(void)
{
	return name;
}

double nnlChiSq(unsigned n, const double* x, double* grad, void* my_func_data)
{
	boost::math::chi_squared_distribution<> chiSqDist1(x[0]);

	double nll = 0;
	my_samples* samp = (my_samples*)my_func_data;
	vector<double> xs = samp->xs;

	for (int i = 0; i < xs.size(); i++)
	{
		nll += -log(pdf(chiSqDist1, xs[i]));
	}
	if (grad) {
		grad[0] = 0;
	}
	return nll;

}