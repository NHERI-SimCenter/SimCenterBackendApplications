#include <gtest/gtest.h>
#include "../writeErrors.h"
#include "../ERANataf.h"
#include "../jsonInput.h"
#include "../runForward.h"
#include "../runGSA.h"
#include "../discreteDist.h"
#include "../runMFMC.h"
writeErrors theErrorFile; // Error log

bool isIdenticalFiles(std::string fname1, std::string fname2, double diffPerc);
void removeFiles(std::string examplePath, int nsamp, std::vector<std::string> fnames);
void runTestForward(std::string examplePath, std::string workflowDriver, std::string inputJson, std::string osType, std::string runType, int nprocs);
void runTestGSA(std::string examplePath, std::string workflowDriver, std::string inputJson, std::string osType, std::string runType, int nprocs);

struct Test_quoFEM
	: public ::testing::Test
{

	std::string workflowDriver, osType, runType, inputJson;
	int procno, nprocs;

	virtual void SetUp() override {
		procno = 0;
		nprocs = 1;
		workflowDriver = "driver.bat";
		inputJson = "scInput.json";
		osType = "Windows";
		runType = "runningLocal";

	}
	virtual void TearDown() override {
		theErrorFile.close();
	}
};

struct Test_EEUQ
	: public ::testing::Test
{

	std::string workflowDriver, osType, runType, inputJson;
	int procno, nprocs;

	virtual void SetUp() override {
		procno = 0;
		nprocs = 1;
		workflowDriver = "sc_driver.bat";
		inputJson = "sc_scInput.json";
		osType = "Windows";
		runType = "runningLocal";
	}
	virtual void TearDown() override {
		theErrorFile.close();
	}
};

TEST_F(Test_quoFEM, RV) {

	std::string examplePath = "C:/Users/SimCenter/Sangri/SimCenterBackendApplications/modules/performUQ/SimCenterUQ/nataf_gsa/test/Examples/Test1";

	std::string workDir = examplePath;
	std::string inpFile = examplePath + "/templatedir/scInput.json";
	theErrorFile.getFileName(workDir + "/dakota.err", procno);

	int procno = 0;

	//
	//  (1) read JSON file
	//

	jsonInput inp(workDir, inpFile, procno);

	//
	//	(2) Construct Nataf Object
	//

	ERANataf T(inp, procno);

	double mean = 3.5;
	double std = 1.2;

    ASSERT_EQ(T.nrv,11) ; // continue even if false
	int i = 0;
	std::string name = "normal";
	ASSERT_NEAR(T.M[i].theDist->getMean(), mean ,0.05);
	ASSERT_NEAR(T.M[i].theDist->getStd(), std, 0.05);
	ASSERT_EQ(T.M[i].theDist->getName(), name);
	i = 1;
	name = "lognormal";
	ASSERT_NEAR(T.M[i].theDist->getMean(), mean, 0.05);
	ASSERT_NEAR(T.M[i].theDist->getStd(), std, 0.05);
	ASSERT_EQ(T.M[i].theDist->getName(), name);
	i = 2;
	name = "beta";
	ASSERT_NEAR(T.M[i].theDist->getMean(), mean, 0.05);
	ASSERT_NEAR(T.M[i].theDist->getStd(), std, 0.05);
	ASSERT_EQ(T.M[i].theDist->getName(), name);
	i = 3;
	name = "uniform";
	ASSERT_NEAR(T.M[i].theDist->getMean(), mean, 0.05);
	ASSERT_NEAR(T.M[i].theDist->getStd(), std, 0.05);
	ASSERT_EQ(T.M[i].theDist->getName(), name);
	i = 4;
	name = "weibull";
	ASSERT_NEAR(T.M[i].theDist->getMean(), mean, 0.05);
	ASSERT_NEAR(T.M[i].theDist->getStd(), std, 0.05);
	ASSERT_EQ(T.M[i].theDist->getName(), name);
	i = 5;
	name = "gumbel";
	ASSERT_NEAR(T.M[i].theDist->getMean(), mean, 0.05);
	ASSERT_NEAR(T.M[i].theDist->getStd(), std, 0.05);
	ASSERT_EQ(T.M[i].theDist->getName(), name);
	i = 6;
	name = "exponential";
	ASSERT_NEAR(T.M[i].theDist->getMean(), mean, 0.05);
	ASSERT_EQ(T.M[i].theDist->getName(), name);
	i = 7;
	name = "gamma";
	ASSERT_NEAR(T.M[i].theDist->getMean(), mean, 0.05);
	ASSERT_NEAR(T.M[i].theDist->getStd(), std, 0.05);
	ASSERT_EQ(T.M[i].theDist->getName(), name);
	i = 8;
	name = "chisquared";
	ASSERT_NEAR(T.M[i].theDist->getMean(), 3, 0.05);
	ASSERT_EQ(T.M[i].theDist->getName(), name);
	i = 9;
	name = "TruncatedExponential";
	ASSERT_NEAR(T.M[i].theDist->getMean(), mean, 0.05);
	ASSERT_EQ(T.M[i].theDist->getName(), name);
	i = 10;
	name = "discrete";
	ASSERT_NEAR(T.M[i].theDist->getMean(), 14.0/6.0, 0.05);
	ASSERT_EQ(T.M[i].theDist->getName(), name);

	removeFiles(examplePath, 0, { "dakota.err" });

}


TEST_F(Test_quoFEM, FORWARD) {

	std::string examplePath = "C:/Users/SimCenter/Sangri/SimCenterBackendApplications/modules/performUQ/SimCenterUQ/nataf_gsa/test/Examples/Test2";
	runTestForward(examplePath, workflowDriver, inputJson, osType, runType, nprocs);

	// TESTS
	ASSERT_TRUE(isIdenticalFiles(examplePath + "/dakotaTab_Test.out", examplePath + "/dakotaTab.out", 0)) << "TEST2 - FORWARD (1) RESULTS MISMATCH\n";

	removeFiles(examplePath, 5, { "dakotaTab.out", "dakota.err" });

}

TEST_F(Test_quoFEM, FORWARD_CORR) {

	std::string examplePath = "C:/Users/SimCenter/Sangri/SimCenterBackendApplications/modules/performUQ/SimCenterUQ/nataf_gsa/test/Examples/Test3";
	runTestForward(examplePath, workflowDriver, inputJson, osType, runType, nprocs);

	// TESTS
	ASSERT_TRUE(isIdenticalFiles(examplePath + "/dakotaTab_Test.out", examplePath + "/dakotaTab.out", 0)) << "TEST3 - FORWARD (2) RESULTS MISMATCH\n";

	removeFiles(examplePath, 5, { "dakotaTab.out", "dakota.err" });

}

TEST_F(Test_quoFEM, GSA) {

	std::string examplePath = "C:/Users/SimCenter/Sangri/SimCenterBackendApplications/modules/performUQ/SimCenterUQ/nataf_gsa/test/Examples/Test4";
	runTestGSA(examplePath, workflowDriver, inputJson, osType, runType, nprocs);

	// TESTS
	ASSERT_TRUE(isIdenticalFiles(examplePath + "/dakota_Test.out", examplePath + "/dakota.out", 0.05)) << "TEST4 - GSA RESULTS MISMATCH\n";

	removeFiles(examplePath, 500, { "dakotaTab.out", "dakota.out", "dakota.err" });

}

TEST_F(Test_quoFEM, GSA_PCA) {

	std::string examplePath = "C:/Users/SimCenter/Sangri/SimCenterBackendApplications/modules/performUQ/SimCenterUQ/nataf_gsa/test/Examples/Test5";
	runTestGSA(examplePath, workflowDriver, inputJson, osType, runType, nprocs);

	// TESTS
	ASSERT_TRUE(isIdenticalFiles(examplePath + "/dakota_Test.out", examplePath + "/dakota.out", 0.05)) << "TEST5 - GSA PCA RESULTS MISMATCH\n";

	removeFiles(examplePath, 300, { "dakotaTab.out", "dakota.out", "dakota.err" });

}



TEST_F(Test_EEUQ, FORWARD) {

	std::string examplePath = "C:/Users/SimCenter/Sangri/SimCenterBackendApplications/modules/performUQ/SimCenterUQ/nataf_gsa/test/Examples/EE_Test1";
	runTestForward(examplePath, workflowDriver, inputJson, osType, runType, nprocs);

	// TESTS
	ASSERT_TRUE(isIdenticalFiles(examplePath + "/dakotaTab_Test.out", examplePath + "/dakotaTab.out", 0)) << "EE TEST1 - FORWARD RESULTS MISMATCH\n";

	removeFiles(examplePath, 5, { "dakotaTab.out", "dakota.err" });

}


TEST(Test_RV, DISCRETE) {

	//discreteDist myDiscrete("PAR", { 1, 0.5, 2, 0.5 }, {});
	RVDist* myDiscrete = new discreteDist("PAR", { 1, 0.5, 2, 0.5 }, {});
    double p1 = myDiscrete->getCdf(1);
    double p2 = myDiscrete->getCdf(2);
	ASSERT_EQ(myDiscrete->getQuantile(p1), 1.0) << "RV TEST 1: DISCRETE CDFs";
	ASSERT_EQ(myDiscrete->getQuantile(p2), 2.0) << "RV TEST 1: DISCRETE CDFs";

}

void runTestForward(std::string examplePath, std::string workflowDriver, std::string inputJson, std::string osType, std::string runType, int nprocs) {
	int procno = 0;

	std::string workDir = examplePath;
	std::string inpFile = examplePath + "/templatedir/" + inputJson;
	theErrorFile.getFileName(workDir + "/dakota.err", procno);

	// (1) read JSON file
	jsonInput inp(workDir, inpFile, procno);
	// (2) Construct Nataf Object
	ERANataf T(inp, procno);
	// (3) Forward analysis
	runForward myForward(workflowDriver, osType, runType, inp, T, procno, nprocs);
	myForward.writeTabOutputs(inp, procno); 
}

void runTestGSA(std::string examplePath, std::string workflowDriver, std::string inputJson, std::string osType, std::string runType, int nprocs) {
	int procno = 0;

	std::string workDir = examplePath;
	std::string inpFile = examplePath + "/templatedir/" + inputJson;
	theErrorFile.getFileName(workDir + "/dakota.err", procno);

	// (1) read JSON file
	jsonInput inp(workDir, inpFile, procno);
	// (2) Construct Nataf Object
	ERANataf T(inp, procno);
	// (3) GSA
	runGSA myGSA(workflowDriver, osType, runType, inp, T, procno, nprocs);
	myGSA.writeOutputs(inp, 0, procno); 
}

bool isIdenticalFiles(std::string fname1, std::string fname2, double diffPerc) {

	std::fstream f1, f2;
	f1.open(fname1, std::ios::in);
	f2.open(fname2, std::ios::in);

	if (!f1 || !f2) {
		return false;
	}

	//bool is_same = true;
	char c1, c2;
	int count = 0;
	int countDiff = 0;
	while (1) {
		count += 1;
		c1 = f1.get();
		c2 = f2.get();
		if (c1 != c2) {
			//is_same = false;
			countDiff = 0;
			//break;
		}
		if ((c1 == EOF) || (c2 == EOF))
			break;
	}
	f1.close();
	f2.close();

	if (float(countDiff) / float(count) > diffPerc)
		return false;
	else
		return true;
}

void removeFiles(std::string examplePath, int nsamp, std::vector<std::string> fnames) {

	for (int i = 0; i < nsamp; i++) {
		std::filesystem::permissions(examplePath + "/workdir." + std::to_string(i+1) + "/driver.bat", std::filesystem::perms::others_all);
		std::filesystem::remove_all(examplePath + "/workdir." + std::to_string(i+1));
	}
	for (auto fname : fnames) {
		if (fname == "dakota.err") {
			theErrorFile.close();
		}
		std::filesystem::remove_all(examplePath + "/" + fname);
	}
}