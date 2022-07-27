
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
 *  Lognoraml distribution class
 */

#include "lognormalDist.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include "nlopt.hpp"

#include <iomanip>
using std::vector;

//double nnlLogn(unsignxed n, const double* x, double* grad, void* my_func_data);

lognormalDist::lognormalDist(string opt, vector<double> val, vector<double> add) 
{
	name = "lognormal";
	int npa = 2;
	if (opt.compare("PAR") == 0)
	{

		if (val.size() != npa)
		{
			std::string errMsg = "Error running UQ engine: The " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMsg);
		}
		else if ((val[1] <= 0))
		{
			std::string errMsg = "Error running UQ engine: The " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMsg);
		}
		else
		{
			lambda = val[0]; // log mean
			zeta = val[1]; // log std
		}

	}
	else if (opt.compare("MOM") == 0)
	{

		if (val.size() != npa)
		{
			std::string errMsg = "Error running UQ engine: The " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMsg);

		}
		else if ((val[0] <= 0) || (val[1] <= 0))
		{
			std::string errMsg = "Error running UQ engine: The " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMsg);
		}
		else
		{
			double mean = val[0];
			double std = val[1];

			lambda = log(mean) - log(sqrt(1 + (std / mean)* (std / mean))); // mean normal
			zeta = sqrt(log(1 + (std / mean)* (std / mean))); // sigma normal

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
		{
			const int np = 1;

			lambda = 0.0;
			double var = 0.0, ns = 0;
			for (double value : val)
			{
				double lnVal = log(value);
				lambda += lnVal;
				var += lnVal* lnVal;
				ns += 1;
			}
			lambda = lambda / ns;
			zeta = sqrt(var / ns - lambda * lambda);
		}

	}
	boost::math::lognormal_distribution<> lognDist1(lambda, zeta); // shape k, scale theta=1/lambda (or 1/lognormal)
	lognDist = lognDist1;
	checkParams();
}

lognormalDist::~lognormalDist() {}


void lognormalDist::checkParams()
{
	double std = getStd();
	if (isnan(std) || isinf(std) || std <= 0)
	{
		std::string errMsg = "Error running UQ engine: stdandard deviation of " + name + " should be greater than 0 ";
		theErrorFile.write(errMsg);
	}
	double mean = getMean();
	if (mean <= 0)
	{
		std::string errMsg = "Error running UQ engine: mean of " + name + " should be greater than 0 ";
		theErrorFile.write(errMsg);
	}
	vector<double> par = getParam();
	if (par[1] <= 0)
	{
		std::string errMsg = "Error running UQ engine: zeta of " + name + " should be greater than 0 ";
		theErrorFile.write(errMsg);
	}
}
//==

double lognormalDist::getPdf(double x)
{
	return pdf(lognDist,x);
}

double lognormalDist::getCdf(double x)
{
	return cdf(lognDist, x);
}

double lognormalDist::getMean(void)
{

	//std::cout << (logn * a + alp * b) / (alp + logn) << std::endl;
	//std::cout << mean(lognDist)*(b-a)+a << std::endl;
	return mean(lognDist);
}

double lognormalDist::getStd(void)
{
	return standard_deviation(lognDist) ;
}

double lognormalDist::getQuantile(double p)
{
	return quantile(lognDist, p) ;
}

string lognormalDist::getName(void)
{
	return name;
}


vector<double> lognormalDist::getParam(void)
{
	return { lambda, zeta };
}

/*
double nnlLogn(unsigned n, const double* x, double* grad, void* my_func_data)
{

	boost::math::lognormal_distribution<> lognDist1(x[0], x[1]);
	
	double nll = 0;
	my_samples* samp = (my_samples*)my_func_data;
	vector<double> xs = samp->xs;

	for (int i = 0; i < xs.size(); i++)
	{
		nll += -log(pdf(lognDist1,(xs[i])));
	}
	if (grad) {
		grad[0] = 0;
	}
	return nll;
}
*/

