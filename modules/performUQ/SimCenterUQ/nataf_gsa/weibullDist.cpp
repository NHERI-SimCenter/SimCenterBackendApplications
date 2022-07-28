
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
 *  Weibull distribution class
 */

#include "weibullDist.h"
//#include "boost/math/distributions/normal.hpp" // for normal_distribution
#include <algorithm>
#include <iostream>
#include <fstream>
#include "nlopt.hpp"
#include <iomanip>
// bet is lamb, an, - follow ERA notation
// alp is k
// boost recieves in order (k,an)

using std::vector;

double nnlWeib(unsigned n, const double* x, double* grad, void* my_func_data);
double paramWeibObj(unsigned n, const double* x, double* grad, void* my_func_data);

weibullDist::weibullDist(string opt, vector<double> val, vector<double> add): weibDist(1.,1.)
{
	name = "weibull";
	int npa = 2;
	if (opt.compare("PAR") == 0)
	{
		if (val.size() != npa)
		{
			std::string errMsg =  "Error running UQ engine: The " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMsg);
		}
		else if (!((val[0] > 0) && (val[1] > 0)))
		{
			std::string errMsg = "Error running UQ engine: parameters of " + name + " distribution must be greater than 0";
			theErrorFile.write(errMsg);
		}
		else
		{
			an = val[0];
			k = val[1];
		}
	}
	else if (opt.compare("MOM") == 0)
	{

		if (val.size() != npa)
		{
			std::string errMsg = "Error running UQ engine: The " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMsg);
		}
		else if (!((val[0] > 0) && (val[1] > 0))) 
		{
			std::string errMsg = "Error running UQ engine: parameters of " + name + " distribution must be greater than 0";
			theErrorFile.write(errMsg);
		}
		else
		{
			// need to solve optimization

			// Param optimization
			k = { 0.5 };
			double lb[1] = { 1.e-10 };
			nlopt_opt optimP;
			optimP = nlopt_create(NLOPT_LN_COBYLA, 1); // derivative-free algorithm
			nlopt_set_lower_bounds(optimP, lb);
			nlopt_set_min_objective(optimP, paramWeibObj, &val);
			nlopt_set_xtol_rel(optimP, 1e-6);
			double minf;

			if (nlopt_optimize(optimP, &k, &minf) < 0) {
				printf("nlopt failed!\n");
				std::string errMsg = "Error running UQ engine: parameter optimization filed";
				theErrorFile.write(errMsg);
			}
			else {
				//lambda = x[0];
				printf("found minimum at f(%g) = %0.10g\n", k, minf);
			}

			an = val[0] / std::tgamma(1 + 1 / k);
		}

	}
	else if (opt.compare("DAT") == 0)
	{

		double minSmp = *std::min_element(std::begin(val), std::end(val));
		if (minSmp < 0)
		{
			std::string errMsg = "Error running UQ engine: samples of " + name + " distribution exceeds the range [0,inf]";
			theErrorFile.write(errMsg);
		}

		const int np = 2;
		//double lb[np] = { -HUGE_VAL, 0 };
		//const double  *lb = { 0 };

		// initial lambda
		double mu = 0.0, var = 0.0, ns=0, std;
		for(double value : val)
		{
			mu += value;
			ns ++;
		}
		an = mu/ns/2;
		k = 0.5;

		vector<double> xs = val;

		// MLE optimization
		double lb[2] = { 1.e-10,1.e-10 };
		nlopt_opt optim;
		optim = nlopt_create(NLOPT_LN_COBYLA, np); // derivative-free algorithm
		nlopt_set_lower_bounds(optim, lb);
		nlopt_set_min_objective(optim, nnlWeib, &xs);
		nlopt_set_xtol_rel(optim, 1e-6);
		double minf;
		double params[2] = { an,k };

		if (nlopt_optimize(optim, params, &minf) < 0) {
			printf("nlopt failed!\n");
			std::string errMsg = "Error running UQ engine: MLE optimization filed";
			theErrorFile.write(errMsg);
		}
		else {
			//lambda = x[0];
			printf("found minimum at f(%g) = %0.10g\n", k, minf);
		}

	}

	weibull weibDist1(k,an);
	weibDist = weibDist1;
	checkParams();

}

weibullDist::~weibullDist() {}
//==


void weibullDist::checkParams()
{
	double std = getStd();
	double mean = getMean();
	vector<double> par = getParam();

	if (isnan(std) || isinf(std) || std <= 0)
	{
		std::string errMsg = "Error running UQ engine: stdandard deviation of " + name + " must be greater than 0 ";
		theErrorFile.write(errMsg);
	}

	if (mean<=0)
	{
		std::string errMsg = "Error running UQ engine: mean of " + name + " distribution must be greater than 0 ";
		theErrorFile.write(errMsg);
	}

	if (!(par[0] > 0 && (par[1] > 0)))
	{
		std::string errMsg = "Error running UQ engine: parameters of " + name + " distribution must be greater than 0 ";
		theErrorFile.write(errMsg);
	}
}


double weibullDist::getPdf(double x)
{
	return pdf(weibDist,x);

}

double weibullDist::getCdf(double x)
{
	return cdf(weibDist, x);

}


double weibullDist::getMean(void)
{
	double a =mean(weibDist);
	return mean(weibDist);
}

double weibullDist::getStd(void)
{
	return standard_deviation(weibDist);
}

double weibullDist::getQuantile(double p)
{

	return quantile(weibDist, p);

}

vector<double> weibullDist::getParam(void)
{
	return { an,k };
}


string weibullDist::getName(void)
{
	return name;
}


double nnlWeib(unsigned n, const double* x, double* grad, void* my_func_data)
{
	weibull weibDist(x[1], x[0]);

	double nll = 0;
	my_samples* samp = (my_samples*)my_func_data;
	vector<double> xs = samp->xs;

	for (int i = 0; i < xs.size(); i++)
	{
		nll += -log(pdf(weibDist, xs[i]));
	}
	if (grad) {
		grad[0] = 0;
	}
	return nll;
}

double paramWeibObj(unsigned n, const double* x, double* grad, void* my_func_data)
{
	my_samples* values = (my_samples*)my_func_data;
	vector<double> val = values->xs;

	if (grad) {
		grad[0] = 0;
	}
	return abs(std::sqrt(tgamma(1 + 2 / x[0]) - (tgamma(1 + 1 / x[0]))* tgamma(1 + 1 / x[0])) / tgamma(1 + 1 / x[0]) - val[1] / val[0]);

}