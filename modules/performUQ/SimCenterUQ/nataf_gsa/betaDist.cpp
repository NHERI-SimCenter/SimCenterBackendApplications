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
 *  Beta distribution class
 */

#include "betaDist.h"
#include "nlopt.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>

using std::vector;

double nnlBeta(unsigned n, const double* x, double* grad, void* my_func_data);

betaDist::betaDist(string opt, vector<double> val, vector<double> add) : betDist(1.0,1.0)
{
	name = "beta";
	int npa = 4;
	if (opt.compare("PAR") == 0)
	{
		if (val.size()!= npa)
		{
			std::string errMSG = "Error running UQ engine: The " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMSG);
		}
		else if ((val[0] <= 0) || (val[1] <= 0) || (val[2] >= val[3]))
		{
			std::string errMSG = "Error running UQ engine: The " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMSG);
		}
		else
		{
			alp = val[0];
			bet = val[1];
			a = val[2];
			b = val[3];
		}

	}
	else if (opt.compare("MOM") == 0)
	{

		if (val.size() != npa)
		{
			std::string errMSG = "Error running UQ engine: The " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMSG);
		}
		else if (val[1] <= 0)
		{
			std::string errMSG = "Error running UQ engine: The " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMSG);
		}
		else if ((val[0] <= val[2]) || (val[0] >= val[3]))
		{
			std::string errMSG = "Error running UQ engine: mean of  " + name + " distribution must be in a valid range";
			theErrorFile.write(errMSG);
		}
		else
		{
			double mu = val[0];
			double sig = val[1];
			a = val[2];
			b = val[3];

			alp = ((b - mu) * (mu - a) / (sig*sig) - 1) * (mu - a) / (b - a);
			bet = alp * (b - mu) / (mu - a);
		}
		if (!((alp > 0) && (bet > 0)))
		{
			std::string errMSG = "Error running UQ engine: parameters of "  + name + " distribution must be greater than 0 ";
			theErrorFile.write(errMSG);
		}
	}
	else if (opt.compare("DAT") == 0)
	{

		if (add.size() == 0)
		{
			std::string errMSG = "Error running UQ engine: Provide a valid rangef for "  + name + " distribution.";
			theErrorFile.write(errMSG);
		}

		a = add[0];
		b = add[1];

		double maxSmp = *std::max_element(std::begin(val), std::end(val));
		double minSmp = *std::min_element(std::begin(val), std::end(val));
		if ((maxSmp > b) || (minSmp < a))
		{
			std::string errMSG = "Error running UQ engine: samples of "  + name + " distribution exceeds the range [min,max]";
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
		double sig = sqrt(var / ns - mu * mu);
		//modify
		alp = ((b - mu) * (mu - a) / (sig * sig) - 1) * (mu - a) / (b - a);
		bet = mu * (b - mu) / (mu - a);

		vector<double> xs;
		for (int i = 0; i < xs.size(); i++)
		{
			xs.push_back((xs[i] - a) / (b - a));
		}

		
		double params[2] = { alp, bet };

		// MLE optimization
		double lb[2] = { 1.e-10 ,  1.e-10 };
		nlopt_opt optim = nlopt_create(NLOPT_LN_COBYLA, np); // derivative-free algorithm
		nlopt_set_lower_bounds(optim, lb);
		
		nlopt_set_min_objective(optim, nnlBeta, &xs);
		nlopt_set_xtol_rel(optim, 1e-6);
		double minf; // `*`the` `minimum` `objective` `value,` `upon` `return`*` 

		//double params;
		if (nlopt_optimize(optim, params, &minf) < 0) {
			printf("nlopt failed!\n");
			std::string errMSG = "Error running UQ engine: MLE optimization filed";
			theErrorFile.write(errMSG);
		}
		else {
			//lambda = x[0];
			printf("found minimum at f(%g) = %0.10g\n", params[0], minf);
		}

		alp = params[0];
		bet = params[1];

	}
	boost::math::beta_distribution<> betDist1(alp, bet); // shape k, scale theta=1/lambda (or 1/beta)
	betDist = betDist1;
	checkParams();
}


betaDist::~betaDist() {}

void betaDist::checkParams()
{
	double std = getStd();
	double mean = getMean();
	vector<double> par = getParam();
	
	if (isnan(std) || isinf(std) || std <= 0)
	{
		std::string errMSG = "Error running UQ engine: stdandard deviation of " + name + " must be greater than 0 ";
		theErrorFile.write(errMSG);
	}
	
	if ((mean <= par[2]) || (mean >=par[3]))
	{
		std::string errMSG = "Error running UQ engine: mean of " + name + " distribution must be in a valid range ";
		theErrorFile.write(errMSG);
	}
	
	if (par[2] > par[3])
	{
		std::string errMSG = "Error running UQ engine: range of " + name + " distribution is not valid ";
		theErrorFile.write(errMSG);
	}

	if (!(par[0]>0 && (par[1]>0)))
	{
		std::string errMSG = "Error running UQ engine: parameters of " + name + " distribution must be greater than 0 ";
		theErrorFile.write(errMSG);
	}
}


//==

double betaDist::getPdf(double x)
{
	return pdf(betDist, (x - a) / (b - a)) / (b - a);
}

double betaDist::getCdf(double x)
{
	return cdf(betDist, (x - a) / (b - a));
}

double betaDist::getMean(void)
{

	//std::cout << (bet * a + alp * b) / (alp + bet) << std::endl;
	//std::cout << mean(betDist)*(b-a)+a << std::endl;
	return mean(betDist) * (b - a) + a;
}

double betaDist::getStd(void)
{
	return standard_deviation(betDist) * (b - a);
}

double betaDist::getQuantile(double p)
{
	return quantile(betDist, p) * (b - a) + a;
}

vector<double> betaDist::getParam(void)
{
	return { alp,bet,a,b };
}

string betaDist::getName(void)
{
	return name;
}

double nnlBeta(unsigned n, const double* x, double* grad, void* my_func_data)
{
	//k:x[0]
	//lambda:x[1]
	//theta: 1/x[1]
	//Bet(k,theta)
	boost::math::beta_distribution<> BetDist1(x[0], x[1]);
	
	double nll = 0;
	my_samples* samp = (my_samples*)my_func_data;
	vector<double> xs = samp->xs;

	for (int i = 0; i < xs.size(); i++)
	{
		nll += -log(pdf(BetDist1,(xs[i])));
	}
	if (grad) {
		grad[0] = 0;
	}
	return nll;
}


