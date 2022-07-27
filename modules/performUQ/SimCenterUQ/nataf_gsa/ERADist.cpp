
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
 *  ERADist class - translated to c++ from the work of Engineering Risk Analysis Group at Technical University of Munich 
 *  https://www.bgu.tum.de/en/era/software/eradist/
 */

#include "ERADist.h"
#include <algorithm>
#include <iostream>
#include <fstream>

ERADist::ERADist() {}

ERADist::ERADist(string getName, string getOpt, vector<double> getVal, vector<double> getAdd, int procno)
{
	// NAME
	transform(getName.begin(), getName.end(), getName.begin(), ::tolower);
	this->name = getName;
	// OPT
	transform(getOpt.begin(), getOpt.end(), getOpt.begin(), ::toupper);
	this->opt = getOpt;

	if (getName.compare("exponential") == 0)
	{
		this->theDist = new exponentialDist(getOpt, getVal);
	} 
	else if (getName.compare("normal") == 0)
	{
		this->theDist = new normalDist(getOpt, getVal);
	}
	else if (getName.compare("gamma") == 0)
	{
		this->theDist = new gammaDist(getOpt, getVal);
	}
	else if (getName.compare("beta") == 0)
	{
		this->theDist = new betaDist(getOpt, getVal, getAdd);
	}
	else if (getName.compare("lognormal") == 0)
	{
		this->theDist = new lognormalDist(getOpt, getVal);
	}
	else if (getName.compare("uniform") == 0)
	{
		this->theDist = new uniformDist(getOpt, getVal);
	}
	else if (getName.compare("chisquared") == 0)
	{
		this->theDist = new chiSquaredDist(getOpt, getVal);
	}
	else if (getName.compare("chisquare") == 0)
	{
		this->theDist = new chiSquaredDist(getOpt, getVal);
	}
	else if (getName.compare("gumbel") == 0)
	{
		this->theDist = new gumbelDist(getOpt, getVal);
	}
	else if (getName.compare("weibull") == 0)
	{
		this->theDist = new weibullDist(getOpt, getVal);
	}
	else if (getName.compare("truncatedexponential") == 0)
	{
		this->theDist = new truncExponentialDist(getOpt, getVal, getAdd);
	} 
	else if (getName.compare("discrete") == 0)
	{
		this->theDist = new discreteDist(getOpt, getVal);
	}
	else
	{
		std::string errMSG = "Error running UQ engine: " + name + " distribution is not supported";
		theErrorFile.write(errMSG);
	}
	
	if (procno==0) {
		std::cout << getName << "[" << getOpt << "] =================================" << std::endl;

		printf("Mean: %3.2f, Std: %3.2f \n", theDist->getMean(), theDist->getStd());
		printf("Params: ");
		int countPars = 0;
		for (auto par : theDist->getParam())
		{
			printf("%3.2f, ", par);
			countPars++;
			if (countPars > 10) {
				printf(" .... ");
				break;		
			}
		}

		printf("\n");
		printf("PDF at mean : %3.2f\n\n", theDist->getPdf(theDist->getMean()));
	}

}

ERADist::~ERADist(void) {}
