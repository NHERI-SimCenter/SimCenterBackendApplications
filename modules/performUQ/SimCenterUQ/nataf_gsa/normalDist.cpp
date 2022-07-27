
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
 *  Noraml distribution class
 */

#include "normalDist.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "nlopt.hpp"


using std::vector;

//double nnlNormal(unsigned n, const double* x, double* grad, void* my_func_data);

normalDist::normalDist(string opt, vector<double> val, vector<double> add )
{
	name = "normal";
	if ((opt.compare("PAR") == 0) || (opt.compare("MOM") == 0))
	{

		if (val.size() != 2)
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
			mu = val[0];
			sigma = val[1];
		}
	}
	else if (opt.compare("DAT") == 0)
	{
		const int np = 1;

		mu = 0.0;
		double var = 0.0, ns = 0;
		for (double value : val)
		{
			mu += value;
			var += value * value;
			ns += 1;
		}
		mu = mu / ns;
		sigma = sqrt(var / ns - mu * mu);
	}

	normal normDist1(mu,sigma);
	normDist = normDist1;
}

normalDist::~normalDist() {}

//==

double normalDist::getPdf(double x)
{

	//return (1.0 / (sigma * sqrt(2.0 * PI))) * exp(-0.5 * (x - mu)*(x - mu) / (sigma * sigma));
	return pdf(normDist, x);
}

double normalDist::getCdf(double x)
{
	//return erfc(-x / sqrt(2.0)) / 2.0;
	return cdf(normDist, x);
}

double normalDist::getMean(void)
{
	return mean(normDist);
}

double normalDist::getStd(void)
{
	return standard_deviation(normDist);
}

double normalDist::getQuantile(double p)
{
	return quantile(normDist, p);
}

string normalDist::getName(void)
{
	return name;
}

vector<double> normalDist::getParam(void)
{
	return { mu, sigma };
}


/*
double nnlNormal(unsigned n, const double* x, double* grad, void* my_func_data)
{
	double lamb = x[0];
	double nll = 0;
	my_samples* samp = (my_samples*)my_func_data;
	vector<double> xs = samp->xs;

	for (int i = 0; i < xs.size(); i++)
	{
		nll += -log(lamb) - (-lamb * xs[i]);
	}
	if (grad) {
		grad[0] = 0;
	}
	return nll;
}
*/


