#ifndef JSON_INPUT_H
#define JSON_INPUT_H

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
 *  To to read in dakota.json file
 */

#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <regex>
#include "writeErrors.h"
#include <filesystem>

extern writeErrors theErrorFile; // Error log

using json = nlohmann::json;
using std::string;
using std::vector;

class jsonInput
{
public:
	jsonInput(string workDir, string inpFile, int procno);
	virtual ~jsonInput(void);

	string workDir;
	string uqType;

	int nmc;
	int nrv;
	int nco;
	int nre;
	int nreg;
	int nqoi;
	int rseed;
	int ngr;
	int nqoiVects;
	string UQmethod;
	vector<string> distNames;
	vector<vector<double>> vals;
	vector<vector<double>> resampleCandidates;
	vector<double> constants;
	vector<string> opts;
	vector<string> rvNames;
	vector<string> qoiNames;
	vector<string> qoiVectNames;
	vector<vector<int>> qoiVectRange;
	vector<vector<double>> corr;
	vector<vector<double>> adds;
	vector<vector<int>> groups;
	vector<vector<int>> resamplingGroups;
	vector<int> resamplingSize;
	bool performPCA;
	double PCAvarRatioThres;

private:
	void getPnames(string distname, string optname, vector<std::string>& par_char);
	void fromTextToStr(string groupTxt, vector<vector<string>>& groupStringVector, vector<string>& flattenStringVect);
	void fromTextToId(string groupTxt, vector<string>& groupPool, vector<vector<int>>& groupIdVect);

};


#endif //  JSON_INPUT_H