

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
 *  Gamma distribution class
 */

#include "gammaDist.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include "nlopt.hpp"
#include "boost/math/distributions/normal.hpp" // for normal_distribution


#include <iomanip>

using std::vector;

double nnlGamma(unsigned n, const double* x, double* grad, void* my_func_data);

gammaDist::gammaDist(string opt, vector<double> val, vector<double> add) : gamDist(1.0,1.0)
{
	name = "gamma";

	if (opt.compare("PAR") == 0)
	{

		if (val.size()!=2)
		{
			std::string errMSG = "Error running UQ engine: The " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMSG);

		}
		else if ((val[0] <= 0) || (val[1] <= 0))
		{
			std::string errMSG = "Error running UQ engine: parameters of " + name + " distribution is always greater than 0";
			theErrorFile.write(errMSG);
		}
		else
		{
			k = val[0];
			lambda = val[1];
		}


	}
	else if (opt.compare("MOM") == 0)
	{

		if (val.size() != 2)
		{
			std::string errMSG = "Error running UQ engine: " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMSG);
		}
		else if (val[0] <= 0)
		{
			std::string errMSG = "Error running UQ engine: mean of " + name + " distribution must be greater than 0";
			theErrorFile.write(errMSG);
		}
		else if (val[1] <= 0)
		{
			std::string errMSG = "Error running UQ engine: standard deviation of " + name + " distribution must be greater than 0";
			theErrorFile.write(errMSG);
		}
		else
		{
			lambda = val[0] / val[1] / val[1];
			k = val[0] * lambda;
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

		const int np = 2;

		double mu = 0.0;
		double var = 0.0, ns = 0;
		for (double value : val)
		{
			mu += value;
			var += value * value;
			ns += 1;
		}
		mu = mu / ns;
		double sigma = sqrt(var / ns - mu * mu);
		lambda = mu / var;
		k = mu * lambda;


		vector<double> xs = val;
		double params[2] = { k,lambda };


		// MLE optimization
		double lb[2] = { 1.e-10 ,  1.e-10 };
		nlopt_opt optim = nlopt_create(NLOPT_LN_COBYLA, np); // derivative-free algorithm
		nlopt_set_lower_bounds(optim, lb);
		
		nlopt_set_min_objective(optim, nnlGamma, &xs);
		nlopt_set_xtol_rel(optim, 1e-6);
		double minf; // `*`the` `minimum` `objective` `value,` `upon` `return`*` 

		//double params;
		if (nlopt_optimize(optim, params, &minf) < 0) {
			printf("nlopt failed!\n");
			std::string errMSG = "Error running UQ engine: MLE optimization filed ";
			theErrorFile.write(errMSG);
		}
		else {
			//lambda = x[0];
			//printf("found minimum at f(%g) = %0.10g\n", params[0], minf);
		}

		k = params[0];
		lambda = params[1];

	}
	boost::math::gamma_distribution<> gamDist1(k, 1/lambda); // shape k, scale theta=1/lambda (or 1/beta)
	gamDist = gamDist1;

}

gammaDist::~gammaDist() {}

//
void gammaDist::checkParams()
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
	if (!(par[0] > 0) && (par[1] > 0))
	{
		std::string errMSG = "Error running UQ engine: parameters of "  + name + " distribution must be greater than 0 ";
		theErrorFile.write(errMSG);
	}
}

double gammaDist::getPdf(double x)
{
	return pdf(gamDist, x);
}

double gammaDist::getCdf(double x)
{
	return cdf(gamDist, x);
}

double gammaDist::getMean(void)
{
	return mean(gamDist);
}

double gammaDist::getStd(void)
{
	return standard_deviation(gamDist);
}

double gammaDist::getQuantile(double p)
{
	// slow
	return quantile(gamDist, p);
}

vector<double> gammaDist::getParam(void)
{
	return { k, lambda };
}

string gammaDist::getName(void)
{
	return name;
}

double nnlGamma(unsigned n, const double* x, double* grad, void* my_func_data)
{
	//k:x[0]
	//lambda:x[1]
	//theta: 1/x[1]
	//gam(k,theta)
	boost::math::gamma_distribution<> gamDist1(x[0], 1.0/x[1]);
	
	double nll = 0;
	my_samples* samp = (my_samples*)my_func_data;
	vector<double> xs = samp->xs;

	for (int i = 0; i < xs.size(); i++)
	{
		nll += -log(pdf(gamDist1,xs[i]));
	}
	if (grad) {
		grad[0] = 0;
	}
	return nll;
}


