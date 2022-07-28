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
 *  Discrete distribution class
 */

#include "discreteDist.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "nlopt.hpp"

#include "writeErrors.h"
extern writeErrors theErrorFile; // Error log

using std::vector;

//extern std::ofstream theErrorFile; // Error log
//double nnldiscrete(unsigned n, const double* x, double* grad, void* my_func_data);

discreteDist::discreteDist(string opt, vector<double> val, vector<double> add )
{
	name = "discrete";
	int np = val.size();
	if ((opt.compare("PAR") == 0))
	{
		if (np%2==1)
		{
			std::string errMSG = "Error running UQ engine: The " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMSG);
		}
		else
		{
			nv = np / 2;
			value.reserve(nv);
			weight.reserve(nv);
			double sumWei = 0;
			for (int i = 0 ; i < nv; i++)
			{
				double w = val[i * 2 + 1];
				if (w <0)
				{
					std::string errMSG = "Error running UQ engine: The " + name + " distribution is not defined for your parameters";
					theErrorFile.write(errMSG);
				}
				value.push_back(val[i * 2]);
				weight.push_back(w);
				sumWei += w;
			}
			if (sumWei/=1)
			{
				for (int i = 0; i < np / 2; i++)
				{
					weight[i]= weight[i] / sumWei;
				}
			}
		}
	}
	else if (opt.compare("MOM") == 0)
	{
		std::string errMSG = "Error running UQ engine: Moment input is not supported for " + name + " distribution";
		theErrorFile.write(errMSG);
	}
	else if (opt.compare("DAT") == 0)
	{
		value = val;
		value.erase(unique(value.begin(), value.end()), value.end());
		nv = value.size();
		int nx = val.size();
		weight.reserve(nv);
		for (int i = 0; i < nv; i++)
		{
			int count = 0;
			for (double v : val)
			{
				if (v == value[i])
				{
 					count += 1;
				}
			}
			weight.push_back((double) count / nx);
		}
	}

	vector<int> y(value.size());
	std::size_t n(0);
	std::generate(std::begin(y), std::end(y), [&] { return n++; });

	std::sort(std::begin(y),
			  std::end(y),
			  [&](int i1, int i2) { return value[i1] < value[i2]; });

	sortIdx = y;

}

discreteDist::~discreteDist() {}

//==

double discreteDist::getPdf(double x)
{

	//return (1.0 / (sigma * sqrt(2.0 * PI))) * exp(-0.5 * (x - mu)*(x - mu) / (sigma * sigma));
	double pdfval = 0;
	for (int i=0;i<nv;i++)
	{
		if (value[i] == x)
		{
			pdfval = weight[i];
		}
	}
	return pdfval;
}

double discreteDist::getCdf(double x)
{
	double cdfval = 0;
	//return erfc(-x / sqrt(2.0)) / 2.0;
	for (int i = 0; i < nv; i++)
	{
		if (value[i] < x)
		{
			cdfval += weight[i];
		}
	}
	return cdfval;
}

double discreteDist::getMean(void)
{
	double meanval = 0;
	for (int i = 0; i < nv; i++)
	{
		meanval += weight[i] * value[i];
	}
	return meanval;
}

double discreteDist::getStd(void)
{
	double varval = 0;
	for (int i = 0; i < nv; i++)
	{
		varval += weight[i] * value[i] * value[i];
	}
	double meanval = getMean();
	return std::sqrt(varval - meanval * meanval);
}

double discreteDist::getQuantile(double p)
{
	double icdfval= HUGE_VAL;
	double cumwei=0;
	for (int i : this->sortIdx)
	{
		cumwei += weight[i];
		if (cumwei>p)
		{ 
			icdfval = value[i];
			break;
		}
	}
	return icdfval;
}

string discreteDist::getName(void)
{
	return name;
}

vector<double> discreteDist::getParam(void)
{
	vector<double> param;
	param.reserve(nv * 2);
	for (int i=0; i < nv; i++)
	{
		param.push_back(value[i]);
		param.push_back(weight[i]);
	}
	return param;
}


/*
double nnldiscrete(unsigned n, const double* x, double* grad, void* my_func_data)
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


