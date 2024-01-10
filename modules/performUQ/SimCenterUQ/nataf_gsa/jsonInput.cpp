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

#include "jsonInput.h"
#include <regex>


jsonInput::jsonInput(string workDir, string inpFile, int procno)
{

	// presetting - change this only for testing

	if (procno == 0)  std::cout << "Reading input json scripts.." << " \n";

	this->workDir = workDir;
	
	//
	// finding json file
	//

    // sy & frank - designsafe shell will always pass me scInput.json as input file name which is not good for non-quofem apps
    // check if the sc_Input.json exists and if so overwirte the input file name

    std::string sc_jsonPath  = workDir + "/templatedir/sc_scInput.json";
    bool sc_exists = std::filesystem::exists(sc_jsonPath);
    if (sc_exists)
    {
        inpFile = sc_jsonPath;
    }

	std::filesystem::path jsonPath(inpFile);

	if (jsonPath.is_relative()){
		jsonPath  = workDir + "/templatedir/" + inpFile;
	}

	std::ifstream myfile(jsonPath.make_preferred());
	if (!myfile.is_open()) {

		std::string errMsg = "Error running UQ engine: Unable to open JSON " + jsonPath.u8string();
		theErrorFile.write(errMsg);
	}
  
	json UQjson = json::parse(myfile, nullptr, false);
	if (UQjson.is_discarded())
	{
		std::string errMsg = "Error reading json: JSON syntax is broken";
		theErrorFile.write(errMsg);
	}

	if (procno == 0)  std::cout << " - Checking the problem definition" << " \n";

	//
	// Check if I am the correct engine.
	//

	uqType = UQjson["UQ"]["uqType"];

	if (((uqType.compare("Forward Propagation") == 0) || (uqType.compare("Sensitivity Analysis") == 0)) ) {
		// pass
	} else
	{
		//*ERROR*
		std::string methodType = UQjson["UQ"]["uqType"];
		std::string errMsg = "Error reading json: 'Forward Analysis', 'Sensitivity Analysis' or 'Multi-fidelity Monte Carlo' backend is called, but the user requested " + methodType;
		theErrorFile.write(errMsg);
	}


	//
	// Are we gonna use surrogate FEM? If so multiple columns will be created in params.in
	//


	if (procno == 0)  std::cout << " - The FEM name? " ;

	// If "Applications" exists
	femAppName="unknown";
	if (UQjson.find("Applications") != UQjson.end()) {
;		// If FEM exists
		if (UQjson["Applications"].find("FEM") != UQjson["Applications"].end()) {
			femAppName = UQjson["Applications"]["FEM"]["Application"];
            std::cout << femAppName << " \n";
		}
	}

	//
	// get EDP names
	//
	if (procno == 0)  std::cout << " - Getting EDP names" << " \n";

	int count_qoi = 0;
	for (auto& elem : UQjson["EDP"]) {
		// *name of distribution
		int lenVal;
		string varName;
		if (elem.find("length") != elem.end()) {
			lenVal = int(elem["length"]);
		} else {
			lenVal = 1;
		}

		if (elem.find("name") != elem.end()) {
			varName = elem["name"];
		}
		else {
			varName = "standard"; 
			lenVal = -1;
		}

		if (lenVal == 1) {
			qoiNames.push_back(varName);
			count_qoi++;
		}
		else if (lenVal > 1) {
			qoiVectNames.push_back(varName); // to combine Sobol indices later
			qoiVectRange.push_back({ count_qoi, count_qoi + lenVal });
			for (int j = 0; j < lenVal; j++) {
				qoiNames.push_back(varName + "_" + std::to_string(j + 1));
				count_qoi++;
			}
		}
	}


	nqoi = count_qoi;
	nqoiVects = qoiVectNames.size();
	

	//
	// Perform PCA?
	//

	if (procno == 0)  std::cout << "   - Perform PCA for EDPs?";
	// default
	if (nqoi > 15) {
		performPCA = true;
	}
	else {
		performPCA = false;
	}

	if (UQjson["UQ"].find("performPCA") != UQjson["UQ"].end()) {

		std::string PCAoption = UQjson["UQ"]["performPCA"];
		if ((PCAoption.compare("Yes") == 0)) {
			performPCA = true;
		}
		else if ((PCAoption.compare("No") == 0)) {
			performPCA = false;
		}
	}

	if (performPCA && (uqType.compare("Sensitivity Analysis") == 0)) {
		if (UQjson["UQ"].find("PCAvarianceRatio") != UQjson["UQ"].end()) {
			PCAvarRatioThres = UQjson["UQ"]["PCAvarianceRatio"];
			if (PCAvarRatioThres <= 0) {
				std::string errMsg = "Error reading input: PCA variance ratio should be greater than zero.";
				theErrorFile.write(errMsg);
			}
			else if (PCAvarRatioThres > 1.0) {
				std::string errMsg = "Error reading input: PCA variance ratio should not be greater than one.";
				theErrorFile.write(errMsg);
			}
		}
		else {
			PCAvarRatioThres = 0.0;
		}
	}
	else {
		PCAvarRatioThres = 0.0;
	}
	if (performPCA) {
		if (procno == 0)  std::cout << " Yes\n";
	}
	else {
		if (procno == 0)  std::cout << " Nope\n";
	}


	//
	// Basic Info
	//

	uqMethod = UQjson["UQ"]["samplingMethodData"]["method"];

	//
	// Else if we read samples...
	//
	if (procno == 0)  std::cout << " - Import files?";

	if (uqMethod.compare("Import Data Files") == 0) {

		nrv = 0;
		for (auto& elem : UQjson["randomVariables"])
		{
			nrv++;
			rvNames.push_back(elem["name"]);
		}
		inpPath = UQjson["UQ"]["samplingMethodData"]["inpFile"];
		outPath = UQjson["UQ"]["samplingMethodData"]["outFile"];

		getGroupIdx(UQjson);		

		inpFileType = UQjson["UQ"]["samplingMethodData"]["inpFiletype"];		
		outFileType = UQjson["UQ"]["samplingMethodData"]["outFiletype"];

		nmc = 0;
		rseed = 0;
        nst = 0;
		if (procno == 0)  std::cout << " Yes, no sampling\n";

		return;
	} else if (uqMethod.compare("Multi-fidelity Monte Carlo") == 0) {
        inpPath = "";
        outPath = "";
        //nmc = UQjson["UQ"]["samplingMethodData"]["samples"];
        nmc = 0;
        nPilot = UQjson["UQ"]["samplingMethodData"]["numPilot"];
        compBudget = UQjson["UQ"]["samplingMethodData"]["maxTime"];
        compBudget *= 60.0; // sec
        rseed = UQjson["UQ"]["samplingMethodData"]["seed"];
        doLogTransform = UQjson["UQ"]["samplingMethodData"]["logTransform"];

    }
	else {
		inpPath = "";
		outPath = "";
		nmc = UQjson["UQ"]["samplingMethodData"]["samples"];
		rseed = UQjson["UQ"]["samplingMethodData"]["seed"];

		if (procno == 0)  std::cout << " Nope we do sampling\n";

	}


	//
	// Specify parameters of each distribution.
	//
	if (procno == 0)  std::cout << " - Reading RV Data groups\n";

	//std::vector<int> corrIdx;
	std::vector<int> randIdx, constIdx, resampIdx, discreteStrIdx; // random variables, constant variables, resampling variables
	int count = 0;
	nrv = 0;
	nco = 0;
	nre = 0;
	nst = 0;

	std::string resampGroupTxt;
	if (UQjson["UQ"]["samplingMethodData"].find("RVdataGroup") != UQjson["UQ"]["samplingMethodData"].end()) {
		// if the key "sensitivityGroups" exists
		resampGroupTxt = UQjson["UQ"]["samplingMethodData"]["RVdataGroup"];
		resampGroupTxt.erase(remove(resampGroupTxt.begin(), resampGroupTxt.end(), ' '), resampGroupTxt.end());
		std::cout << resampGroupTxt << std::endl;
	}
	else if (UQjson["UQ"].find("RVdataGroup") != UQjson["UQ"].end()) {
		// FOR VERSION COMPETIBILITY - TO BE REMOVED SOON.... sy 08/12/2022
		resampGroupTxt = UQjson["UQ"]["RVdataGroup"];
		resampGroupTxt.erase(remove(resampGroupTxt.begin(), resampGroupTxt.end(), ' '), resampGroupTxt.end());
		std::cout << resampGroupTxt << std::endl;
	}
	else {
		resampGroupTxt = "";
	}
	
	vector<vector<string>> resamplingGroupsString;
	vector<string> flattenResamplingGroups;

	fromTextToStr(resampGroupTxt, resamplingGroupsString, flattenResamplingGroups);
	nreg = resamplingGroupsString.size();

	auto it = std::unique(flattenResamplingGroups.begin(), flattenResamplingGroups.end());
	bool isUnique = (it == flattenResamplingGroups.end());
	if (!isUnique) {
		//*ERROR*
		std::string errMsg = "Error reading input: groups of random variables should be mutually exclusive";
		theErrorFile.write(errMsg);

	}


	//
	// Specify parameters of each distribution.
	//
	if (procno == 0)  std::cout << " - Reading RV parameters\n";

	for (auto& elem : UQjson["randomVariables"])
	{
		std::string distName = elem["distribution"];
		// name of distribution
		std::transform(distName.begin(), distName.end(), distName.begin(), ::tolower); // make it lower case
		distName.erase(remove_if(distName.begin(), distName.end(), isspace), distName.end()); // remove space

		std::string inpType;
		if ((elem.find("inputType") == elem.end()))
		{
			if (distName=="discrete_design_set_string") {
				inpType = "Parameters";
			} else if (distName=="normal") {
				inpType = "Parameters";
			} else {
				std::string errMsg = "Error reading json: input file does not have the key 'inputType'";
				theErrorFile.write(errMsg);
			}
		}
		else {
			inpType = elem["inputType"];
		}
		// if key "correlationMatrix" exists
		

		// type of input (PAR, MOM, or DAT)
		std::string inpTypeSub = inpType.substr(0, 3);
		for (int i = 0; i < 3; i++) {
			inpTypeSub[i] = toupper(inpTypeSub[i]);
		}

		// get parameter names for each dist
		std::vector<std::string> pnames;
		getPnames(distName, inpTypeSub, pnames);

		// If resampling

		if (std::find(flattenResamplingGroups.begin(), flattenResamplingGroups.end(), elem["name"]) != flattenResamplingGroups.end()) {
			//is_resamplingGroup
			if (!(distName.compare("discrete") == 0) || !(inpTypeSub.compare("DAT")) == 0) {
				//*ERROR*
				string InputType;
				if (!inpTypeSub.compare("DAT")) {
					InputType = "Dataset";
				}
				else if (!inpTypeSub.compare("PAR")) {
					InputType = "Parameters";
				}
				else {
					InputType = "Moments";
				}
				std::string errMsg = "Error reading input: RVs specified in UQ tab should have the option Dataset-discrete. Your input is " + InputType + "-" + distName + "";
				theErrorFile.write(errMsg);
			}
			resampIdx.push_back(count);
			nre++;
			count++;
			continue;
		}


		// If constant
		if (distName.compare("constant") == 0) {
			constIdx.push_back(count);
			nco++;
			count++;
			continue;
		}
		if ((distName.compare("discrete") == 0) && (inpTypeSub.compare("PAR")) == 0) {
			if (elem[pnames[0]].size() == 1) {
				// discrete distribution with only one quantity = constant
				constIdx.push_back(count);
				nco++;
				count++;
				continue;
			}
		}
		// If discrete_string
		if (distName.compare("discrete_design_set_string") == 0) {
			discreteStrIdx.push_back(count);
			nst++;
			count++;
			continue;
		}


		// save name of random variable etc
		rvNames.push_back(elem["name"]);
		distNames.push_back(distName);
		opts.push_back(inpTypeSub);

		// if "DAT" 
		if (opts[nrv].compare("DAT") == 0) {

			// Sample set inside vals
			// std::string directory = elem["dataDir"];

			
			
			std::string tmpName = elem["name"];
			std::filesystem::path dir = workDir;
			std::filesystem::path relPath = "templatedir//" + tmpName + ".in";
			relPath = relPath.make_preferred();
			std::filesystem::path directory = dir / relPath;

			std::ifstream data_table(directory);
			if (!data_table.is_open()) {
				//*ERROR*
				std::string errMsg = "Error reading json: cannot open data file: " + directory.u8string();
				theErrorFile.write(errMsg);
			}

			std::vector<double> vals_tmp;
			double samps = 0.0;
			while (data_table >> samps)
			{ 
				vals_tmp.push_back(samps);
				if (data_table.peek() == ',')
					data_table.ignore();
			}
			data_table.close();
			resampleCandidates.push_back(vals_tmp);
			vals.push_back(vals_tmp);

			if (vals_tmp.size() < 1) { //*ERROR*
				auto a = vals_tmp.size();

				std::string errMsg = "Error reading json: data file of " + rvNames[nrv] + " has less then one sample.";
				theErrorFile.write(errMsg);
			}

			// Save boundary informaions

			if (distNames[nrv].compare("binomial") == 0) {
				adds.push_back({ elem["n"],0.0 }); // not used
				addStrs.push_back({}); // not used
			}
			else if (distNames[nrv].compare("beta") == 0) {
				adds.push_back({ elem["lowerbound"],elem["upperbound"] });
				addStrs.push_back({});
			}
			else if (distNames[nrv].compare("truncatedexponential") == 0) {
				adds.push_back({ elem["a"],elem["b"] });
				addStrs.push_back({});
			}
			else
			{
				adds.push_back({}); // default
				addStrs.push_back({});
			}
		}
		else // if "PAR" or "MOM" 
		{
			// Parameter (moment) values inside vals
			if (distNames[nrv].compare("discrete") == 0) {
				std::vector<double> vals_tmp;
				auto numdisc = elem[pnames[0]].size();
				for (int i = 0; i < numdisc; i++)
				{
					vals_tmp.push_back(elem[pnames[0]][i]);
					vals_tmp.push_back(elem[pnames[1]][i]);
				}
				vals.push_back(vals_tmp);
			}else if (distNames[nrv].compare("discrete_design_set_string") == 0) {
				auto nelem = elem["elements"].size();
				std::vector<double> vals_tmp(nelem);
				std::iota(vals_tmp.begin(), vals_tmp.end(), 1);
				vals.push_back(vals_tmp);
			}
			else
			{
				std::vector<double> vals_temp;
				for (auto& pn : pnames)
				{
					if (elem.find(pn) != elem.end())
					{ 
						vals_temp.push_back(elem[pn]); // get parameter values
					}
					else
					{
						std::string errMsg = "Error reading json: cannot find " + pn + " in " + distName + " from input json.";
						theErrorFile.write(errMsg);
					}

				}
				vals.push_back(vals_temp);
			}


			if (distNames[nrv].compare("discrete_design_set_string") == 0) {
				adds.push_back({}); // default
				addStrs.push_back(elem["elements"]);
			}
			else {
				adds.push_back({});
				addStrs.push_back({});
			}
		}
		randIdx.push_back(count);
		nrv++;
		count++;
	}

	//
	// get resamples
	//

	if (procno == 0)  std::cout << " - Getting resampling indices.. if any\n";

	for (int i : resampIdx)
	{

		auto elem = UQjson["randomVariables"][i];

		// Sample set inside vals
		// std::string directory = elem["dataDir"];
		std::string tmpName = elem["name"];
		std::filesystem::path dir = workDir;
		std::filesystem::path relPath = "templatedir//" + tmpName + ".in";
		relPath = relPath.make_preferred();
		std::filesystem::path directory = dir / relPath;
		std::ifstream data_table(directory);
		if (!data_table.is_open()) {
			//*ERROR*
			std::string errMsg = "Error reading json: cannot open data file at " + directory.u8string();
			theErrorFile.write(errMsg);
		}

		std::vector<double> vals_tmp;
		double samps = 0.0;
		while (data_table >> samps)
		{
			vals_tmp.push_back(samps);
			if (data_table.peek() == ',')
				data_table.ignore();
		}
		data_table.close();

		if (vals_tmp.size() < 1) { //*ERROR*
			//*ERROR*
			std::string errMsg = "Error reading json: data file of " + rvNames[i] + " has less then one sample.";
			theErrorFile.write(errMsg);

		}

		vals.push_back(vals_tmp);
		resampleCandidates.push_back(vals_tmp);
		rvNames.push_back(elem["name"]);
	}

	//
	// get constants
	//

	if (procno == 0)  std::cout << " - Getting constant variable names.. if any\n";
	//for (auto& elem : UQjson["randomVariables"])
	for (int i : constIdx)
	{
		// name of distribution
		auto elem = UQjson["randomVariables"][i];
		string distname = elem["distribution"];
		std::transform(distname.begin(), distname.end(), distname.begin(), ::tolower); // make lower case

		// input type
		std::string inpType = elem["inputType"];
		std::string inpTypeSub = inpType.substr(0, 3);
		std::transform(inpTypeSub.begin(), inpTypeSub.end(), inpTypeSub.begin(), ::toupper); // make upper case

		// parameter name
		std::vector<std::string> pnames;
		getPnames(distname, inpTypeSub, pnames);

		// If constant
		if (distname.compare("constant") == 0) 
		{
			// *name of random variable
			rvNames.push_back(elem["name"]);
			constants.push_back(elem[pnames[0]]);
		}
		// If constant (discrete)
		else if ((distname.compare("discrete") == 0) && (inpTypeSub.compare("PAR")) == 0) 
		{ 
			if (elem["value"].size() == 1) {
				// discrete distribution with only one quantity = constant
				rvNames.push_back(elem["name"]);
				constants.push_back(elem[pnames[0]][0]);
			}
		}
	}


	//
	// get discreteStr
	//

	if (procno == 0)  std::cout << " - Getting discreteStr variable names.. if any\n";
	//for (auto& elem : UQjson["randomVariables"])
	for (int i : discreteStrIdx)
	{
		// name of distribution
		auto elem = UQjson["randomVariables"][i];
		string distname = elem["distribution"];
		std::transform(distname.begin(), distname.end(), distname.begin(), ::tolower); // make lower case

		// input type
		std::string inpType = "Parameters";
		std::string inpTypeSub = inpType.substr(0, 3);
		std::transform(inpTypeSub.begin(), inpTypeSub.end(), inpTypeSub.begin(), ::toupper); // make upper case

		// parameter name
		std::vector<std::string> pnames;
		getPnames(distname, inpTypeSub, pnames);

		// If discrete_design_set_string
		if (distname.compare("discrete_design_set_string") == 0)
		{
			// *name of random variable
			rvNames.push_back(elem["name"]);
			discreteStrValues.push_back(elem["elements"]);
		}
	}



	//
	// get correlation matrix
	//

	//vector<vector<double>> corr;
	//for (int i = 0; i < nrv; i++)
	//{
	//	vector<double> corr_row(nrv, 0.2);
	//	corr_row[i] = 1.0; // overwrite diagonal elements
	//	corr.push_back(corr_row);
	//}

	//corr.reserve(nrv*nrv);

	//find the number of multimodels
	int multiModelCount = 0;
	for (int i = 0; i < nrv; i++) {
		if (rvNames[i].rfind("MultiModel-", 0) == 0) {
			multiModelCount++;
		}
	}


	//if ((multiModelCount > 1)&& (uqType.compare("Multi-fidelity Monte Carlo") == 0)) {
		//*ERROR*
	//	std::string errMsg = "Error reading json: SimCenterUQ engine supports only one model group";
	//	theErrorFile.write(errMsg);

	//}


	if (procno == 0)  std::cout << " - Checking correlation matrix\n";

	if (UQjson.find("correlationMatrix") != UQjson.end()) {
		corr = *new vector<vector<double>>(nrv, vector<double>(nrv, 0.0));
		// if key "correlationMatrix" exists
		for (int i = 0; i < nrv; i++) {
			for (int j = 0; j < nrv; j++) {

				if ((rvNames[i].rfind("MultiModel-", 0) == 0) | (rvNames[j].rfind("MultiModel-", 0) == 0)) 
				{
					if (i == j) {
						corr[i][j] = 1.0;
					}
					else {
						corr[i][j] = 0.0;
					}
				}
				else {
					corr[i][j] = UQjson["correlationMatrix"][randIdx[i] + randIdx[j] * (nrv + nco + nre - multiModelCount)];
				}
			}
		}
	}
	else
	{
		//corr.assign(nrv* nrv, 0);
		for (int i = 0; i < nrv; i++) {
			vector<double> corr_row(nrv, 0.0);
			corr_row[i] = 1.0; // overwrite diagonal elements
			corr.push_back(corr_row);
		}

//}
	}

	getGroupIdx(UQjson);

	//
	// get resampling group index matrix
	//

	fromTextToId(resampGroupTxt, rvNames, resamplingGroups);



	for (int i = 0; i < nreg; i++) {
		auto length_old = vals[resamplingGroups[i][0]].size();
		int length_data;
		for (int j = 1; j < resamplingGroups[i].size(); j++) {
			length_data =vals[resamplingGroups[i][j]].size();
			if (length_data != length_old)
			{
				std::string errMsg = "Error reading json: RVs in the same group do not have the same number of samples";
				theErrorFile.write(errMsg);
			}
			length_old = length_data;
		}
		resamplingSize.push_back(length_data);
	}

}

void
jsonInput::fromTextToId(string groupTxt, vector<string>& groupPool, vector<vector<int>>& groupIdVect)
{
	auto nrv = groupPool.size();
    std::regex re(R"(\{([^}]+)\})"); // will get string inside {}
    //std::regex re(""); // will get string inside {}
    //auto re = std::regex("Hello");
    std::sregex_token_iterator it(groupTxt.begin(), groupTxt.end(), re, 1);
	std::sregex_token_iterator end;
	while (it != end) {
		std::stringstream ss(*it++);
		std::vector<int> groupID;
		std::vector<string> groupString;
		while (ss.good()) {
			std::string substr;
			getline(ss, substr, ',');  // incase we have multiple strings inside {}
			groupString.push_back(substr);

			std::vector<std::string>::iterator itr = std::find(groupPool.begin(), groupPool.end(), substr);
			if (itr != groupPool.cend()) { // from names (a,b,{a,b}) to idx's (1,2,{1,2})		
				int index_rvn = std::distance(groupPool.begin(), itr); // start from 0
				groupID.push_back((int)index_rvn);
				std::string errMsg;
				if (index_rvn >= nrv) {
					std::string errMsg = "Error reading json: RV group (for Sobol) cannot contain constant variable: " + groupPool[index_rvn]  ;
					theErrorFile.write(errMsg);
				}
			}
			else {
				// *ERROR*
				std::string errMsg = "Error reading json: element " + substr + " inside the variable groups not found.";
				theErrorFile.write(errMsg);

			}
		}
		groupIdVect.push_back(groupID);
	}
}

void
jsonInput::fromTextToStr(string groupTxt, vector<vector<string>>& groupStringVector, vector<string>& flattenStringVect)
{
    int a=3;

    std::regex re(R"(\{([^}]+)\})"); // will get string inside {}
    //std::regex re("\\w+");
    //std::regex re;
	//re.assign("(soft)(.*)");
	std::sregex_token_iterator it(groupTxt.begin(), groupTxt.end(), re, 1);
	std::sregex_token_iterator end;
	while (it != end) {
		std::stringstream ss(*it++);
		std::vector<string> groupString;
		while (ss.good()) {
			std::string substr;
			getline(ss, substr, ',');  // incase we have multiple strings inside {}
			groupString.push_back(substr);
			flattenStringVect.push_back(substr);
		}
		groupStringVector.push_back(groupString);
	}
}

void 
jsonInput::getPnames(string distname, string optname, vector<std::string>& par_char)
{
	if (optname.compare("PAR") == 0) { // Get parameters

		if (distname.compare("binomial") == 0) {  // Not used
			par_char.push_back("n");
			par_char.push_back("p");
		}
		else if (distname.compare("geometric") == 0) {  // Not used
			par_char.push_back("p");
		}
		else if (distname.compare("negativebinomial") == 0) {  // Not used
			par_char.push_back("k");
			par_char.push_back("p");
		}
		else if (distname.compare("poisson") == 0) {
			par_char.push_back("lambda");
			par_char.push_back("t");
		}
		else if (distname.compare("uniform") == 0) {
			par_char.push_back("lowerbound");
			par_char.push_back("upperbound");
		}
		else if (distname.compare("normal") == 0) {
			par_char.push_back("mean");
			par_char.push_back("stdDev");
		}
		else if (distname.compare("lognormal") == 0) {
			par_char.push_back("lambda");
			par_char.push_back("zeta");
		}
		else if (distname.compare("exponential") == 0) {
			par_char.push_back("lambda");
		}
		else if (distname.compare("gamma") == 0) {
			par_char.push_back("k");
			par_char.push_back("lambda");
		}
		else if (distname.compare("beta") == 0) {
			par_char.push_back("alphas");
			par_char.push_back("betas");
			par_char.push_back("lowerbound");
			par_char.push_back("upperbound");
		}
		else if (distname.compare("gumbelMin") == 0) {  // Not used
			par_char.push_back("an");
			par_char.push_back("bn");
		}
		else if (distname.compare("gumbel") == 0) {
			par_char.push_back("alphaparam");
			par_char.push_back("betaparam");
		}
		else if (distname.compare("frechet") == 0) {  // Not used
			par_char.push_back("an");
			par_char.push_back("k");
		}
		else if (distname.compare("weibull") == 0) {
			par_char.push_back("scaleparam"); //an
			par_char.push_back("shapeparam"); //k
		}
		else if (distname.compare("gev") == 0) {  // Not used
			par_char.push_back("beta");
			par_char.push_back("alpha");
			par_char.push_back("epsilon");
		}
		else if (distname.compare("gevmin") == 0) {  // Not used
			par_char.push_back("beta");
			par_char.push_back("alpha");
			par_char.push_back("epsilon");
		}
		else if (distname.compare("pareto") == 0) {  // Not used
			par_char.push_back("x_m");
			par_char.push_back("alpha");
		}
		else if (distname.compare("rayleigh") == 0) {  // Not used
			par_char.push_back("alpha");
		}
		else if (distname.compare("chisquare") == 0) {
			par_char.push_back("k");
		}
		else if (distname.compare("discrete") == 0) {
			par_char.push_back("Values");
			par_char.push_back("Weights");
		}
		else if (distname.compare("truncatedexponential") == 0) {
			par_char.push_back("lambda");
			par_char.push_back("a");
			par_char.push_back("b");
		}
		else if (distname.compare("constant") == 0) {
			par_char.push_back("value");
		}
		else if (distname.compare("discrete_design_set_string") == 0) {
			// keep params empty
		}
		else {
			std::string errMsg = "Error reading json: cannot interpret distribution name: " + distname;
			theErrorFile.write(errMsg);
		}
	}
	else if (optname.compare("MOM") == 0) { // Get Moments
		if (distname.compare("normal") == 0) {  // Not used
			par_char.push_back("mean");
			par_char.push_back("stdDev");  // 
		}
		else if (distname.compare("lognormal") == 0) {  // Not used
			par_char.push_back("mean");
			par_char.push_back("stdDev");  // 
		}
		else if (distname.compare("geometric") == 0) {  // Not used
			par_char.push_back("mean");
		}
		else if (distname.compare("poisson") == 0) {
			par_char.push_back("mean");
		}
		else if (distname.compare("exponential") == 0) {
			par_char.push_back("mean");
		}
		else if (distname.compare("beta") == 0) {
			par_char.push_back("mean");
			par_char.push_back("standardDev");
			par_char.push_back("lowerbound");
			par_char.push_back("upperbound");
		}
		else if (distname.compare("gev") == 0) {
			par_char.push_back("mean");
			par_char.push_back("standardDev");
			par_char.push_back("epsilon");
		}
		else if (distname.compare("gevmin") == 0) {  // Not used
			par_char.push_back("mean");
			par_char.push_back("standardDev");
			par_char.push_back("epsilon");
		}
		else if (distname.compare("rayleigh") == 0) {  // Not used
			par_char.push_back("mean");
		}
		else if (distname.compare("chisquare") == 0) {
			par_char.push_back("mean");
		}
		else if (distname.compare("constant") == 0) {
			par_char.push_back("value");
		}
		else if (distname.compare("truncatedexponential") == 0) {
			par_char.push_back("mean");
			par_char.push_back("a");
			par_char.push_back("b");
		}
		else
		{
			par_char.push_back("mean");
			par_char.push_back("standardDev");
		}
	}
	else if (optname.compare("DAT") == 0) { // Get DATA	
		if (distname.compare("binomial") == 0) {
			par_char.push_back("n");
		}
		else if (distname.compare("beta") == 0) {
			par_char.push_back("lowerbound");
			par_char.push_back("upperbound");
		}
		else if (distname.compare("truncatedexponential") == 0) {
			par_char.push_back("a");
			par_char.push_back("b");
		}
		else if (distname.compare("constant") == 0) {
			par_char.push_back("value");
		}
	}
	else {
		std::string errMsg = "Error reading json: input type should be one of PAR/MOM/DAT";
		theErrorFile.write(errMsg);
	}
}

jsonInput::~jsonInput(void) {}

void
jsonInput::getGroupIdx(json UQjson) {

	//
	// get group index matrix
	//

	bool generate_default_RVsensitivityGroup = true;
	if (UQjson["UQ"].find("RVsensitivityGroup") != UQjson["UQ"].end()) {

		// if the key "sensitivityGroups" exists
		std::string groupTxt = UQjson["UQ"]["RVsensitivityGroup"];
		if (!groupTxt.empty()) {
			// if value of "sensitivityGroups" is nonempty
			groupTxt.erase(remove(groupTxt.begin(), groupTxt.end(), ' '), groupTxt.end()); // remove any white spaces
			fromTextToId(groupTxt, rvNames, groups);
			generate_default_RVsensitivityGroup = false;
		}

	}
	if (generate_default_RVsensitivityGroup) {

		for (int i = 0; i < nrv; i++) {
			groups.push_back({ i });
		}
		for (int i = 0; i < resamplingGroups.size(); i++) {
			for (int j = 0; j < resamplingGroups[i].size(); j++) {
				groups.push_back({ resamplingGroups[i][j] });
			}
		}

	}

	ngr = groups.size();
}
