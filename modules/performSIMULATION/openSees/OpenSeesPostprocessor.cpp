
#include "OpenSeesPostprocessor.h"
#include <jansson.h> 
#include <string.h>
#include <math.h>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>

#include "common/Units.h"


int main(int argc, char **argv)
{
  if (argc != 5) {
    printf("ERROR %d: correct usage: postprocessOpenSees fileNameAIM fileNameSAM fileNameEVENT filenameEDP\n", argc);
    return -1;
  }

  char *filenameAIM = argv[1];
  char *filenameSAM = argv[2];
  char *filenameEVENT = argv[3];
  char *filenameEDP = argv[4];

  OpenSeesPostprocessor thePostprocessor;

  thePostprocessor.processResults(filenameAIM, filenameSAM, filenameEDP);

  return 0;
}


OpenSeesPostprocessor::OpenSeesPostprocessor()
  :filenameEDP(0), filenameAIM(0), filenameSAM(0)
{

}

OpenSeesPostprocessor::~OpenSeesPostprocessor(){
  if (filenameEDP != 0)
    delete [] filenameEDP;
  if (filenameAIM != 0)
    delete [] filenameAIM;
  if (filenameSAM != 0)
    delete [] filenameSAM;  
}

int 
OpenSeesPostprocessor::processResults(const char *AIM, const char *SAM, const char *EDP)
{
  //
  // make copies of filenames in case methods need them
  //

  if (filenameEDP != 0)
    delete [] filenameEDP;
  if (filenameAIM != 0)
    delete [] filenameAIM;
  if (filenameSAM != 0)
    delete [] filenameSAM;  

  filenameEDP=(char*)malloc((strlen(EDP)+1)*sizeof(char));
  strcpy(filenameEDP,EDP);
  filenameAIM=(char*)malloc((strlen(AIM)+1)*sizeof(char));
  strcpy(filenameAIM,AIM);
  filenameSAM=(char*)malloc((strlen(SAM)+1)*sizeof(char));
  strcpy(filenameSAM,SAM);  

  json_error_t error;
  rootEDP = json_load_file(filenameEDP, 0, &error);

  //
  // if SAM has units, determine length scale factor to get to AIM units
  //
  
  rootSAM = json_load_file(filenameSAM, 0, &error);  
  unitConversionFactorAcceleration = 1.0;
  unitConversionFactorForce = 1.0;
  unitConversionFactorLength = 1.0;
  
  json_t* samUnitsJson = json_object_get(rootSAM, "units");
  
  if (samUnitsJson != NULL) {

    // read SAM units
    Units::UnitSystem samUnits;
    json_t* samLengthJson = json_object_get(samUnitsJson, "length");
    json_t* samForceJson = json_object_get(samUnitsJson, "force");    
    json_t* samTimeJson = json_object_get(samUnitsJson, "time");
    samUnits.lengthUnit = Units::ParseLengthUnit(json_string_value(samLengthJson));
    samUnits.forceUnit = Units::ParseForceUnit(json_string_value(samForceJson));
    samUnits.timeUnit = Units::ParseTimeUnit(json_string_value(samTimeJson));
    
    // read AIM  units
    rootAIM = json_load_file(filenameAIM, 0, &error);
    json_t* genInfoJson = json_object_get(rootAIM, "GeneralInformation");
    
    Units::UnitSystem bimUnits;    
    json_t* bimUnitsJson = json_object_get(genInfoJson, "units");
    json_t* bimLengthJson = json_object_get(bimUnitsJson, "length");
    json_t* bimForceJson = json_object_get(bimUnitsJson, "force");
    json_t* bimTimeJson = json_object_get(bimUnitsJson, "time");    
    bimUnits.lengthUnit = Units::ParseLengthUnit(json_string_value(bimLengthJson));
    bimUnits.forceUnit = Units::ParseForceUnit(json_string_value(bimForceJson));
    bimUnits.timeUnit = Units::ParseTimeUnit(json_string_value(bimTimeJson));

    // conversion factors
    unitConversionFactorForce = Units::GetForceFactor(samUnits, bimUnits);
    unitConversionFactorLength = Units::GetLengthFactor(samUnits, bimUnits);
    unitConversionFactorAcceleration = Units::GetAccelerationFactor(samUnits, bimUnits);    
  } 

  processEDPs();

  json_dump_file(rootEDP,filenameEDP,0);
  json_object_clear(rootEDP);  
  return 0;
}


int 
OpenSeesPostprocessor::processEDPs(){

  //
  // foreach EVENT
  //   processEDPs, i.e. open ouputfile, read data, write to edp and dump
  //

  int numTimeSeries = 1;
  int numPatterns = 1;

  int index;
  json_t *event;

  json_t *edps = json_object_get(rootEDP,"EngineeringDemandParameters");  
  
  int numEvents = json_array_size(edps);
  char edpEventName[50];

  //
  // try opening results.out file; unknown EDP results may be there or are already in ED"
  //

  ifstream resultsFile;  
  resultsFile.open("results.out");
  double valueResults = 0;

  for (int i=0; i<numEvents; i++) {

    // process event
    json_t *eventEDPs = json_array_get(edps,i);
    const char *eventName = json_string_value(json_object_get(eventEDPs,"name"));
    //const char *edpEventName = json_string_value(json_object_get(eventEDPs,"name"));
    sprintf(edpEventName,"%d",i);    

    json_t *eventEDP = json_object_get(eventEDPs,"responses");
    int numResponses = json_array_size(eventEDP);
    
    for (int k=0; k<numResponses; k++) {

      json_t *response = json_array_get(eventEDP, k);
      const char *type = json_string_value(json_object_get(response, "type"));

      if (strcmp(type,"max_abs_acceleration") == 0) {
	const char * cline = json_string_value(json_object_get(response, "cline"));
	const char * floor = json_string_value(json_object_get(response, "floor"));
	json_t *dofs = json_object_get(response, "dofs");
	int numDOFs = json_array_size(dofs);

	string fileString;
	ostringstream temp;  //temp as in temporary
	temp << filenameAIM << edpEventName << "." << type << "." << cline << "." << floor << ".out";
	fileString=temp.str(); 
	
	const char *fileName = fileString.c_str();

	//
	// open file & process data into a json array called: data
	//
	
	json_t *data = json_array();
	
	// open file
	ifstream myfile;
	myfile.open (fileName);
	double tmp;
	
	if (myfile.is_open()) {
	  
	  // read first 2 rows of useless data
	  for (int ii=0; ii<2; ii++)
	    for (int jj=0; jj<numDOFs; jj++) 
	      myfile >> tmp;
	  // read last row and add components to data
	  for (int jj=0; jj<numDOFs; jj++) {
	    myfile >> tmp;

	    tmp *= unitConversionFactorAcceleration;
	    
	    json_array_append(data, json_real(tmp));
	  }
	  myfile.close();
	}
	
	// set the response
	json_object_set(response,"scalar_data",data);

      }

      else if (strcmp(type,"rms_acceleration") == 0) {
	const char * cline = json_string_value(json_object_get(response, "cline"));
	const char * floor = json_string_value(json_object_get(response, "floor"));
	json_t *dofs = json_object_get(response, "dofs");
	int numDOFs = json_array_size(dofs);

	string fileString;
	ostringstream temp;  //temp as in temporary
	temp << filenameAIM << edpEventName << "." << type << "." << cline << "." << floor << ".out";
	fileString=temp.str(); 
	
	const char *fileName = fileString.c_str();
	
	//
	// open file & process data into a json array called: data
	//
	
	json_t *data = json_array();
	
	// open file
	ifstream myfile;
	myfile.open (fileName);
	double tmp;
	
	if (myfile.is_open()) {
	  
	  for (int jj=0; jj<numDOFs; jj++) {
	    myfile >> tmp;
	    tmp *= unitConversionFactorAcceleration;
	    json_array_append(data, json_real(tmp));
	  }
	  myfile.close();
	}
	
	// set the response
	json_object_set(response,"scalar_data",data);
	
      } 

      else if (strcmp(type,"max_rel_disp") == 0) {
	const char * cline = json_string_value(json_object_get(response, "cline"));
	const char * floor = json_string_value(json_object_get(response, "floor"));
	json_t *dofs = json_object_get(response, "dofs");
	int numDOFs = json_array_size(dofs);
	
	string fileString;
	ostringstream temp;  //temp as in temporary
	temp << filenameAIM << edpEventName << "." << type << "." << cline << "." << floor << ".out";
	fileString=temp.str(); 
	
	const char *fileName = fileString.c_str();
	
	//
	// opencfile & process data into a json array called: data
				//
	
	json_t *data = json_array();
	
	// open file
	ifstream myfile;
	myfile.open (fileName);
	double tmp;
	
	if (myfile.is_open()) {
	  
	  // read first 2 rows of useless data
	  for (int ii=0; ii<2; ii++)
	    for (int jj=0; jj<numDOFs; jj++) 
	      myfile >> tmp;
	  // read last row and add components to data
	  for (int jj=0; jj<numDOFs; jj++) {
	    myfile >> tmp;
	    tmp *= unitConversionFactorLength;
	    json_array_append(data, json_real(tmp));
	  }
	  myfile.close();
	}
	
	// set the response
	json_object_set(response,"scalar_data",data);
	
      } 

      else if ((strcmp(type,"max_drift") == 0) || 
	       (strcmp(type,"max_roof_drift") == 0)) {
	
	const char *cline = json_string_value(json_object_get(response, "cline"));
	const char *floor1 = json_string_value(json_object_get(response, "floor1"));
	const char *floor2 = json_string_value(json_object_get(response, "floor2"));
	json_t *dofs = json_object_get(response, "dofs");
	int numDOFs = json_array_size(dofs);
	
	json_t *data = json_array();	
	for (int ii=0; ii<numDOFs; ii++) {
	  int dof = json_integer_value(json_array_get(dofs,ii));
	  string fileString1;
	  ostringstream temp1;  //temp as in temporary
	  
	  temp1 << filenameAIM << edpEventName << "." << type << "." << cline << "." << floor1 << "." << floor2 << "." << dof << ".out";
	  fileString1=temp1.str(); 
	  
	  const char *fileName1 = fileString1.c_str();	
	  
	  // openfile & process data
	  ifstream myfile;
	  myfile.open (fileName1);
	  double absMin, absMax, absValue;
	  
	  absValue = 0.0;
	  if (myfile.is_open()) {
	    myfile >> absMin >> absMax >> absValue;
	    myfile.close();
	  } 
	  json_array_append(data, json_real(absValue));
	}
	
	// set the response
	json_object_set(response,"scalar_data",data);
      }
      
      else if (strcmp(type,"residual_disp") == 0) {
	
	printf("ERROR - OpenSeesPostprocessor needs to implement\n");
	exit(-1);
	
	const char *cline = json_string_value(json_object_get(response, "cline"));
	const char *floor = json_string_value(json_object_get(response, "floor"));
	
	string fileString;
	ostringstream temp;  //temp as in temporary
	temp << filenameAIM << edpEventName << "." << type << "." << cline << "." << floor << ".out";
	fileString=temp.str(); 
	
	const char *fileName = fileString.c_str();
	
	// openfile & process data
	ifstream myfile;
	myfile.open (fileName);
	double num1 = 0.; 
	double num2 = 0.;
	double num = 0.;
				
	if (myfile.is_open()) {
	  //	    std::vector<double> scores;
	  //keep storing values from the text file so long as data exists:
	  while (myfile >> num1 >> num2) {
	    //	      scores.push_back(num);
	  }
	  
	  // need to process to get the right value, for now just output last
	  num = fabs(num1);
	  if (fabs(num2) > num)
	    num = fabs(num2);
	  num *= unitConversionFactorLength;
	  myfile.close();
	}
	
	json_object_set(response,"scalar_data",json_real(num));
	/*
	  json_t *scalarValues = json_object_get(response,"scalar_data");
	  json_array_append(scalarValues,json_real(num));
	*/
      } 
      else if (strcmp(type,"max_pressure") == 0)
	{
          json_t *data = json_array();

          //json_t* floor2Json = json_object_get(response, "floor2");
          json_t* dofsArrayJson = json_object_get(response, "dofs");
          json_t* dofJson;
          size_t index;
          json_array_foreach(dofsArrayJson, index, dofJson)
          {
              //TODO: we need to read the pressure values from the event file, if they exist
              //For now we are adding 0 for each degree of freedom
              json_array_append(data, json_real(0.0));
          }

          json_object_set(response,"scalar_data",data);

      }
      else {
	fprintf(stderr, "%s\n",type);
	double valueResults = 0;
	if (resultsFile.is_open() && !resultsFile.eof()) {
	  resultsFile >> valueResults;
	  fprintf(stderr, "%f\n",valueResults);
	}

	json_t *data = json_array();	
	json_array_append(data, json_real(valueResults));
	json_object_set(response,"scalar_data",data);	  
      }
    }
  }

  if (resultsFile.is_open())
      resultsFile.close();

  return 0;
}


