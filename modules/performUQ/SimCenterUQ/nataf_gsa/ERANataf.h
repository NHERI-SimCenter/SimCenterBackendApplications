#ifndef ERA_NATAF_H
#define ERA_NATAF_H

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
 *  ERAnataf class - translated to c++ from the work of Engineering Risk Analysis Group at Technical University of Munich
 *  https://www.bgu.tum.de/en/era/software/eradist/
 */

#include "ERADist.h"
#include "jsonInput.h"
#include "Eigen/Dense"
#include "writeErrors.h"
//#define MPI

#ifdef MPI
	#include <mpi.h>
#else
	#include <omp.h>
#endif
 //extern std::ofstream theErrorFile; // Error log

extern writeErrors theErrorFile; // Error log

using std::string;
using std::vector;

class ERANataf
{

public:
	ERANataf(void);
	ERANataf(jsonInput inp, int procno);
	~ERANataf();

	int nrv;
	vector<vector<double>> Rhox;
	vector<vector<double>> Rhoz;
	vector<vector<double>> U;
	vector<vector<int>> resampID;
	vector<vector<double>> X;
	vector<vector<double>> G;
	Eigen::MatrixXd RhozMat, RhozInv;

	vector<vector<double>> X2U(int nmc, vector<vector<double>> x);
	vector<vector<double>> U2X(int nmc, vector<vector<double>> u);
	//ERADist **M_;
	vector<ERADist> M;
	void simulateAppBatch(string workflowDriver,
						 string osType, 
						 string runType, 
						 jsonInput inp, 
						 int procno,
						 int nproc);
	void simulateAppSequential(string workflowDriver, 
						string osType,
						string runType,
						jsonInput inp,
						vector<vector<double>> u,
						vector<vector<double>>& xvals,
						vector<vector<double>>& gvals,
						int idx);
	vector<double> simulateAppOnce(int i,
						string workingDirs,
						string copyDir,
						int nrvcore,
						int qoi,
						vector<string> rvNames,
						vector<double> xs,
						string workflowDriver,
						string osType,
						string runType);
	void sample(jsonInput inp, int procno);
	void readCSV(string filename, int ndim, vector<vector<double>>& mat, int& nsamp);
	void readBin(string filename, int ndim, vector<vector<double>>& mat, int& nsamp);
	void readDataset(string inpFilePath, string outFilePath, int xdim, int ydim, string option, int &nmcs);


private:
	const double PI = 4 * atan(1);
	void quadGL(int N, double a, double b, vector<double>& x, vector<double>& w);
	Eigen::MatrixXd A;
	normal stdNorm;
	double mvnpdfR(Eigen::VectorXd u);
	double mvncdfR(Eigen::VectorXd u);
	double getJointPdf(vector<double> x);
	double getJointCdf(vector<double> x);
	double normCdf(double x);


};

// For MLE optimization
typedef struct NatafStr {
	vector<double> points = {};
	vector<double> fxii = {};
	vector<double> fxij = {};
	double Rhoxij=0;
	int ngrid=0;
} my_NatafInfo;

#endif // ERA_NATAF_H

