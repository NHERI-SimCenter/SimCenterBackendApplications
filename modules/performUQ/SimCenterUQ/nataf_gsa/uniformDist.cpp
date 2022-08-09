
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
 *  Uniform distribution class
 */

#include "uniformDist.h"
#include <algorithm>
#include <iostream>
#include <fstream>

#include <iomanip>

using std::vector;

uniformDist::uniformDist(string opt, vector<double> val, vector<double> add) 
{
	name = "uniform";
	int npa = 2;
	if (opt.compare("PAR") == 0)
	{

		if (val.size()!= npa)
		{
			std::string errMsg = "Error running UQ engine: " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMsg);
		}
		else if ((val[0] >= val[1]))
		{
			std::string errMsg = "Error running UQ engine: " + name + " distribution is not valid";
			theErrorFile.write(errMsg);
		}
		else
		{
			a = val[0]; 
			b = val[1]; 
		}

	}
	else if (opt.compare("MOM") == 0)
	{

		if (val.size() != npa)
		{
			std::string errMsg = "Error running UQ engine: The " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMsg);
		}
		else if ((val[1] <= 0))
		{
			std::string errMsg = "Error running UQ engine: standard deviation of " + name + " distribution must be greater than 0";
			theErrorFile.write(errMsg);
		}
		else
		{
			double mean = val[0];
			double std = val[1];


			a = mean - sqrt(12) * std / 2;
			b = mean + sqrt(12) * std / 2;
		}

	}
	else if (opt.compare("DAT") == 0)
	{
		int ns = val.size();
		auto min_value = *std::min_element(val.begin(), val.end());
		auto max_value = *std::max_element(val.begin(), val.end());

		// Estimating the range
		a = max_value + ((double) ns + 1.0) * (min_value - max_value) / ns;
		b = min_value + ((double) ns + 1.0) * (max_value - min_value) / ns;

	}
	boost::math::uniform_distribution<> unifDist1(a, b); // shape k, scale theta=1/lambda (or 1/uniform)
	unifDist = unifDist1;

}

uniformDist::~uniformDist() {}

//==

double uniformDist::getPdf(double x)
{
	return pdf(unifDist,x);
}

double uniformDist::getCdf(double x)
{
	return cdf(unifDist, x);
}

double uniformDist::getMean(void)
{

	//std::cout << (unif * a + alp * b) / (alp + unif) << std::endl;
	//std::cout << mean(unifDist)*(b-a)+a << std::endl;
	return mean(unifDist);
}

double uniformDist::getStd(void)
{
	return standard_deviation(unifDist) ;
}

double uniformDist::getQuantile(double p)
{
	return quantile(unifDist, p) ;
}

string uniformDist::getName(void)
{
	return name;
}


vector<double> uniformDist::getParam(void)
{
	return { a, b };
}

/*
double nnlUnif(unsigned n, const double* x, double* grad, void* my_func_data)
{

	boost::math::uniform_distribution<> unifDist1(x[0], x[1]);
	
	double nll = 0;
	my_samples* samp = (my_samples*)my_func_data;
	vector<double> xs = samp->xs;

	for (int i = 0; i < xs.size(); i++)
	{
		nll += -log(pdf(unifDist1,(xs[i])));
	}
	if (grad) {
		grad[0] = 0;
	}
	return nll;
}

*/