
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
 *  Truncated exponential distribution class
 */

#include "truncExponentialDist.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include "nlopt.hpp"
#include <iomanip>

using std::vector;


double nnlTruncExponential(unsigned n, const double* x, double* grad, void* my_func_data);
double paramTruncExpbObj(unsigned n, const double* x, double* grad, void* my_func_data);

truncExponentialDist::truncExponentialDist(string opt, vector<double> val, vector<double> add)
{
	name = "TruncatedExponential";
	int np=3;
	if (opt.compare("PAR") == 0)
	{
		if (val.size() != np)
		{
			std::string errMsg = "Error running UQ engine: " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMsg);

		}
		else if (val[0] <= 0) 
		{
			std::string errMsg = "Error running UQ engine: " + name + " distribution must be greater than 0";
			theErrorFile.write(errMsg);
		}
		else if ((val[1] <= 0) || (val[1] > val[2]))
		{
			std::string errMsg = "Error running UQ engine: range of " + name + " distribution is not valid";
			theErrorFile.write(errMsg);
		}
		else
		{
			lambda = val[0];
			a = val[1];
			b = val[2];
		}
	}
	else if (opt.compare("MOM") == 0)
	{

		if (val.size() != np)
		{
			std::string errMsg = "Error running UQ engine: " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMsg);
		}
		else if ((val[1] <= 0) || (val[1] > val[2]))
		{
			std::string errMsg = "Error running UQ engine: range of " + name + " distribution is not valid";
			theErrorFile.write(errMsg);

		}
		else if ( (val[0] <= 0) || (2*val[0]>=(val[1]+val[2])) )
		{
			std::string errMsg =  "Error running UQ engine: parameter of " + name + " distribution cannot be defined for your input";
			theErrorFile.write(errMsg);
		}
		else
		{
			a = val[1];
			b = val[2];

			lambda = { 1/val[0] };
			double lb[1] = { 1.e-10 };
			nlopt_opt optimP;
			optimP = nlopt_create(NLOPT_LN_COBYLA, 1); // derivative-free algorithm
			nlopt_set_lower_bounds(optimP, lb);
			nlopt_set_min_objective(optimP, paramTruncExpbObj, &val);
			nlopt_set_xtol_rel(optimP, 1e-6);
			double minf;

			if (nlopt_optimize(optimP, &lambda, &minf) < 0) {
				printf("nlopt failed!\n");
				std::string errMsg = "Error running UQ engine: parameter optimization filed ";
				theErrorFile.write(errMsg);
			}
			else {
				//lambda = x[0];
				printf("found minimum at f(%g) = %0.10g\n", lambda, minf);
			}


		}

	}
	else if (opt.compare("DAT") == 0)
	{

		if (add.size() == 0)
		{
			std::string errMsg = "Error running UQ engine: Provide a valid range";
			theErrorFile.write(errMsg);
		}

		a = add[0];
		b = add[1];

		double maxSmp = *std::max_element(std::begin(val), std::end(val));
		double minSmp = *std::min_element(std::begin(val), std::end(val));
		if ((maxSmp > b) || (minSmp < a))
		{
			std::string errMsg = "Error running UQ engine: samples of " + name + " distribution exceeds the range [min,max]=[" + std::to_string(a) + "," + std::to_string(b) + "]";
			theErrorFile.write(errMsg);
		}

		const int np = 1;

		// data
		my_samples info;
		info.xs = val;
		info.add = add;
		// initial value
		double mu = 0.0, ns = 0;
		for (double value : val)
		{
			mu += value;
			ns += 1;
		}
		mu = mu / ns;
		lambda = { 1 / mu };
		double lb[1] = { 1.e-10 };
		// MLE optimization
		nlopt_opt optim;
		optim = nlopt_create(NLOPT_LN_COBYLA, np); // derivative-free algorithm
		nlopt_set_lower_bounds(optim, lb);
		nlopt_set_min_objective(optim, nnlTruncExponential, &info);
		nlopt_set_xtol_rel(optim, 1e-6);
		double minf; // `*`the` `minimum` `objective` `value,` `upon` `return`*` 

		if (nlopt_optimize(optim, &lambda, &minf) < 0) {
			printf("nlopt failed!\n");
			std::string errMsg = "Error running UQ engine: samples of MLE optimization filed";
			theErrorFile.write(errMsg);
		}
		else {
			//lambda = x[0];
			printf("found minimum at f(%g) = %0.10g\n", lambda, minf);
		}

	}

	exponential expDist1(lambda);
	expDist = expDist1;
	normConst=cdf(expDist,b)-cdf(expDist,a);
	checkParams();
}

truncExponentialDist::~truncExponentialDist() {}


void truncExponentialDist::checkParams()
{
	double std = getStd();
	double mean = getMean();
	vector<double> par = getParam();

	if (isnan(std) || isinf(std) || std <= 0)
	{
		std::string errMsg = "Error running UQ engine: stdandard deviation of " + name + " distribution must be greater than 0 ";
		theErrorFile.write(errMsg);
	}

	if ((mean <= par[1]) || (mean >= par[2]))
	{
		std::string errMsg = "Error running UQ engine: parameter of " + name + " distribution cannot be defined for the input";
		theErrorFile.write(errMsg);
	}

	if (!(par[0] > 0))
	{
		std::string errMsg =  "Error running UQ engine: parameter of " + name + " distribution must be greater than 0 ";
		theErrorFile.write(errMsg);
	}
}

double truncExponentialDist::getPdf(double x)
{
	
	double pdfval;
	if ((x<a) || (x>b))
	{
		pdfval = 0;
	} else {
		pdfval=pdf(expDist,x)/normConst;
	}
	return pdfval;

}

double truncExponentialDist::getCdf(double x)
{
	double cdfval;
	if (x>b){
		cdfval=1; 
	}
	else if (x>a)
	{
		cdfval=(cdf(expDist,x)-cdf(expDist,a))/normConst;
	} else {
		cdfval = 0;
	}
	return cdfval;

}


double truncExponentialDist::getMean(void)
{
	double up = std::min(b,getQuantile(0.999));
	int ncount=1.e5;
  	double step = (up - a) / ncount;
  	double integral = 0.5*(a*getPdf(a)+ up* getPdf(up));

  	for (int i=1; i<ncount+1;i++) {
  		double xval = a+step*i;
  		integral += xval* getPdf(xval);
  	}

  	integral *= step;
	return integral;
}

double truncExponentialDist::getStd(void)
{
	double up = std::min(b, getQuantile(0.999));
	int ncount = 1.e5;
	double step = (up - a) / ncount;
	double mean = getMean();
	double integral = 0.5 * ((a* a) * getPdf(a) + (up*up) * getPdf(up));

	for (int i = 1; i < ncount + 1; i++) {
		double xval = a + step * i;
		integral += (xval*xval)* getPdf(xval);
	}

	integral *= step;
	return std::sqrt(integral-mean* mean);
}

double truncExponentialDist::getQuantile(double p)
{

	double minpf = cdf(expDist, a);
	return quantile(expDist, minpf + p*normConst);

}

vector<double> truncExponentialDist::getParam(void)
{
	return { lambda, a, b };
}


string truncExponentialDist::getName(void)
{
	return name;
}


double nnlTruncExponential(unsigned n, const double* x, double* grad, void* my_func_data)
{
	double nll=0;

	exponential expDist1(x[0]);

	my_samples* samp = (my_samples*)my_func_data;
	vector<double> xs = samp->xs;
	vector<double> add = samp->add;

	double lb = add[0];
	double ub = add[1];

	double normConst = cdf(expDist1, ub) - cdf(expDist1, lb);
	for (int i = 0; i < xs.size(); i++)
	{
		nll += -log(pdf(expDist1, xs[i]))+log(normConst);
	}

	if (grad) {
		grad[0] = 0;
	}

	return nll;
}



double paramTruncExpbObj(unsigned n, const double* x, double* grad, void* my_func_data)
{
	my_samples* values = (my_samples*)my_func_data;
	vector<double> val = values->xs;
	double mean = val[0];
	double lb = val[1];
	//double ub = val[2];
	exponential expDist(x[0]);

	double ub = std::min(val[2], quantile(expDist,0.999));

	int ncount = 1.e5;
	double step = (ub - lb) / ncount;
	double integral = 0.5 * (lb * pdf(expDist,lb) + ub * pdf(expDist, lb));

	for (int i = 1; i < ncount + 1; i++) {
		double xval = lb + step * i;
		integral += xval * pdf(expDist, xval);
	}

	integral *= step;
	
	// let us use trapzoidal integration to find the mean


	if (grad) {
		grad[0] = 0;
	}
	return abs(integral / (cdf(expDist, ub) - cdf(expDist, lb))  - mean);

}

