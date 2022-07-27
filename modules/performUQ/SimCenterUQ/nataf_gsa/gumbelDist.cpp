
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
 *  Gumbel distribution class
 */

#include "gumbelDist.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include "nlopt.hpp"
//#include <gsl/gsl>

#include <iomanip>

using std::vector;

double nnlGumb(unsigned n, const double* x, double* grad, void* my_func_data);
//double nnlGumb2(const gsl_vector *v, void *params, void* my_func_data);

gumbelDist::gumbelDist(string opt, vector<double> val, vector<double> add) 
{
	name = "gumbel";
	int npa = 2;
	if (opt.compare("PAR") == 0)
	{

		if (val.size()!= npa)
		{
			std::string errMsg =  "Error running UQ engine: The " + name + " distribution is not defined for your parameters"; 
			theErrorFile.write(errMsg);
		}
		else if ((val[0] <= 0))
		{
			std::string errMsg = "Error running UQ engine: scale parameter of " + name + " distribution must be greater than 0 " ;
			theErrorFile.write(errMsg);
		}
		else
		{
			alp = val[0]; // 1/an
			bet = val[1];
		}


	}
	else if (opt.compare("MOM") == 0)
	{

		if (val.size() != npa)
		{
			std::string errMsg = "Error running UQ engine: The " + name + " distribution is not defined for your parameters";
			theErrorFile.write(errMsg);

		}
		else if (val[1] <= 0)
		{
			std::string errMsg = "Error running UQ engine: stdandard deviation of " + name + " distribution must be greater than 0 ";
			theErrorFile.write(errMsg);
		}
		else
		{
			double an = val[1] * sqrt(6) / PI;// ; // scale parameter
			bet = val[0] - EC * an; // location parameter, EC: euler constant
			alp = 1 / an;
		}

	}
	else if (opt.compare("DAT") == 0)
	{
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

		double an = sigma * sqrt(6) / PI;// ; // scale parameter
		bet = mu - EC * an; // location parameter, EC: euler constant
		alp = 1 / an;


		size_t iter = 0;
		int status;

//		const gsl_multimin_fminimizer_type *T;
//		gsl_multimin_fminimizer *s;
//
//		/* Position of the minimum (1,2), scale factors
//           10,20, height 30. */
//
//		gsl_vector *x;
//		gsl_multimin_function_f my_func;
//
//		my_func.n = 2;
//		my_func.f = nnlGumb;
//		my_func.params = par;
//
//		/* Starting point, x = (5,7) */
//		x = gsl_vector_alloc (2);
//		gsl_vector_set (x, 0, alp);
//		gsl_vector_set (x, 1, bet);
//
//		T = gsl_multimin_fminimizer_conjugate_fr;
//		s = gsl_multimin_fminimizer_alloc (T, 2);
//
//		gsl_multimin_fminimizer_set (s, &my_func, x, 0.01, 1e-4);
//
//		do
//		{
//		    iter++;
//		    status = gsl_multimin_fminimizer_iterate (s);
//
//		    if (status)
//		        break;
//
//		    printf ("%5d %.5f %.5f %10.5f\n", iter,
//                    gsl_vector_get (s->x, 0),
//                    gsl_vector_get (s->x, 1),
//                    s->f);
//		}
//		while (status == GSL_CONTINUE && iter < 100);
//
//		gsl_multimin_fminimizer_free (s);
//		gsl_vector_free (x);


		/////////////

		vector<double> xs = val;
		double params[2] = { alp,bet };

		// MLE optimization
		double lb[2] = { 1.e-10 , -INFINITY };
		nlopt_opt optim = nlopt_create(NLOPT_LN_COBYLA, np); // derivative-free algorithm
		nlopt_set_lower_bounds(optim, lb);
		
		nlopt_set_min_objective(optim, nnlGumb, &xs);
		nlopt_set_xtol_rel(optim, 1e-6);
		double minf; // `*`the` `minimum` `objective` `value,` `upon` `return`*` 

		//double params;
		if (nlopt_optimize(optim, params, &minf) < 0) {
			printf("nlopt failed!\n");
			std::string errMsg = "Error running UQ engine: MLE optimization filed";
			theErrorFile.write(errMsg);
		}
		else {
			//lambda = x[0];
			printf("found minimum at f(%g) = %0.10g\n", params[0], minf);
		}

		alp = params[0]; // 1/scale
		bet = params[1]; // loca


	}

	boost::math::extreme_value_distribution<> gumbDist1(bet, 1 / alp);
	gumbDist = gumbDist1;

	// (location, scale) parameters
	// =(a, b) in boost-doc
	// =(bn, an) in ERADist
	// =(bet, 1/alp) Dakota manual
	// User input of quoFEM & Dakota : alp, bet
	checkParams();
}

gumbelDist::~gumbelDist() {}

//==

void gumbelDist::checkParams()
{
	double std = getStd();
	double mean = getMean();
	vector<double> par = getParam();

	if (isnan(std) || isinf(std) || std <= 0)
	{
		std::string errMsg = "Error running UQ engine: stdandard deviation of " + name + " distribution must be greater than 0 ";
		theErrorFile.write(errMsg);
	}

	if (par[0] <= 0)
	{
		std::string errMsg = "Error running UQ engine: scale parameter of " + name + " distribution must be greater than 0 ";
		theErrorFile.write(errMsg);
	}
}


double gumbelDist::getPdf(double x)
{
	return pdf(gumbDist, x);
}

double gumbelDist::getCdf(double x)
{
	return cdf(gumbDist, x);
}

double gumbelDist::getMean(void)
{
	//std::cout << bet + EC / alp << std::endl;
	//std::cout << mean(gumbDist) << std::endl;
	return mean(gumbDist);
}

double gumbelDist::getStd(void)
{
	//std::cout << PI / alp / sqrt(6) << std::endl;
	//std::cout << standard_deviation(gumbDist) << std::endl;
	return standard_deviation(gumbDist);
}

double gumbelDist::getQuantile(double p)
{

	//printf("BOOST: %.5f\n", bet - 1 / alp*log(-log(p)));
	//printf("BOOST: %.5f\n", quantile(gumbDist, p));

	return quantile(gumbDist, p); // slow
	//return  bet - 1 / alp * log(-log(p));
}

vector<double> gumbelDist::getParam(void)
{
	return { alp, bet};
}

string gumbelDist::getName(void)
{
	return name;
}


double nnlGumb(unsigned n, const double* x, double* grad, void* my_func_data)
{
	boost::math::extreme_value_distribution<> gumbDist1(x[1], 1 / x[0]);
	
	double nll = 0;
	my_samples* samp = (my_samples*)my_func_data;
	vector<double> xs = samp->xs;

	double loc = x[1];
	double sca = 1/x[0];

	for (int i = 0; i < xs.size(); i++)
	{
		//nll += -log(pdf(gumbDist1,xs[i]));
		nll += log(sca)+(xs[i]- loc)/sca + exp(-(xs[i] - loc) / sca);
	}
	if (grad) {
		grad[0] = 0;
	}
	return nll;

}

//
//double nnlGumb2(const gsl_vector *v, void* my_func_data)
//{
//
//    double *p = (double *)params;
//
//    double sca = 1/gsl_vector_get(v, 0);
//    double loc = gsl_vector_get(v, 1);
//
//    boost::math::extreme_value_distribution<> gumbDist1(loc, sca);
//
//    double nll = 0;
//    my_samples* samp = (my_samples*)my_func_data;
//    vector<double> xs = samp->xs;
//
//    for (int i = 0; i < xs.size(); i++)
//    {
//        //nll += -log(pdf(gumbDist1,xs[i]));
//        nll += log(sca)+(xs[i]- loc)/sca + exp(-(xs[i] - loc) / sca);
//    }
//
//    return nll;
//
//}
