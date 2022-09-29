#include <jansson.h> 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "common/Units.h"
#include <iostream>

//#include <string>
//#include <sstream>
//#include <map>

int
main(int argc, char **argv) {


  char *filenameAIM = NULL;
  char *filenameSAM = NULL;

  // MDOF --filenameAIM file? --filenameEVENT file? --filenameSAM file? <--getRV>

  if (argc == 8 || argc == 7) {

    int arg = 1;
    while(arg < argc)
    {
        if (strcmp(argv[arg], "--filenameAIM") == 0)
        {
            arg++;
            filenameAIM = argv[arg];
        }
        else if (strcmp(argv[arg], "--filenameSAM") == 0)
        {
            arg++;
            filenameSAM = argv[arg];
        }
        arg++;
    }

  } else {
    fprintf(stderr, "ERROR - MDOF - incorrect # args want: MDOF --filenameAIM file? --filenameEVENT file? --filenameSAM file? <--getRV>\n");
  }


  json_error_t error;
  json_t *rootBIM = json_load_file(filenameAIM, 0, &error);
  json_t *rootSAM = json_object();

  //
  // stuff that we add to the SAM that need to be filled out as we parse the file
  //

  json_t *properties = json_object();
  json_t *geometry = json_object();
  json_t *materials = json_array();
  json_t *nodes = json_array();
  json_t *elements = json_array();
  

  // ensure this is correct type
  json_t *SIM = json_object_get(rootBIM,"Modeling");
  if (SIM == 0) {
      fprintf(stderr, "Modeling Section not found in input\n");
      exit(-1);
  }


  json_t *typeSIM = json_object_get(SIM,"type");  
  if ((typeSIM == 0) || (strcmp("MDOF_BuildingModel",json_string_value(typeSIM)) != 0)) {
    if (typeSIM != 0) 
      fprintf(stderr, "type: %s\n", json_string_value(typeSIM));
    else
      fprintf(stderr, "type: NONE PROVIDED\n");

    fprintf(stderr, "ERROR - MDOF_BuildingModel - incorrect type\n");
    exit(-1);    
  }
  
  //
  // get base info that must be there
  //

  json_t *numStoriesType = json_object_get(SIM,"numStories");
  int numStory = json_integer_value(numStoriesType);
  int numNodes = numStory + 1;


  //
  // check for eccentricty in response location
  //

  bool eResponse = false;
  double responseX = 0.;
  double responseY = 0.;

  json_t *resXobj = json_object_get(SIM,"responseX");  
  if (resXobj != NULL) {
    responseX = json_number_value(resXobj);
  }
  json_t *resYobj = json_object_get(SIM,"responseY");  
  if (resYobj != NULL) {
    responseX = json_number_value(resYobj);
  }
  
  if (responseX != 0. || responseY != 0.)
    eResponse = true;


  if (strcmp("--getRV", argv[argc-1]) != 0) {

    // 
    // output the model
    //

    // get ModelData
    json_t *modelData = json_object_get(SIM,"ModelData");  

    //ReadBIMUnits
    json_t* genInfoJson = json_object_get(rootBIM, "GeneralInformation");
    json_t* bimUnitsJson = json_object_get(genInfoJson, "units");
    json_t* bimLengthJson = json_object_get(bimUnitsJson, "length");
    json_t* bimTimeJson = json_object_get(bimUnitsJson, "time");
    Units::UnitSystem bimUnits;
    bimUnits.lengthUnit = Units::ParseLengthUnit(json_string_value(bimLengthJson));
    bimUnits.timeUnit = Units::ParseTimeUnit(json_string_value(bimTimeJson));

    // initialize some variablles
    double height = 0;
    double weight = 0;
    double G = Units::GetGravity(bimUnits);//used to be 386.41 before converting the units

    int ndf = 2;
    double floorHeight = 0.0;
    double buildingWeight = 0.0;
    int nodeTag = 2;
    int eleTag = 1;
    int matTag = 1;

    double *zeta = new double[numStory];


    //
    // check for eccentricty in mass location
    //
    
    double massX = 0.;
    double massY = 0.;
    bool eMass = false;

    json_t *massXobj = json_object_get(SIM,"massX");  
    if (massXobj != NULL) {
      massX = json_number_value(massXobj);
    }
    json_t *massYobj = json_object_get(SIM,"massY");  
    if (massYobj != NULL) {
      massY = json_number_value(massYobj);
    }

    if (massX != 0. || massY != 0.)
	eMass = true;
    
    //
    // add nodes, elements and materials for each floor and story to roof
    //

    int index = 0;
    json_t *floorData;
    json_array_foreach(modelData, index, floorData) {

      double kx = json_number_value(json_object_get(floorData, "kx"));
      double ky = json_number_value(json_object_get(floorData, "ky"));
      double Fyx = json_number_value(json_object_get(floorData, "Fyx"));
      double Fyy = json_number_value(json_object_get(floorData, "Fyy"));
      double bx = json_number_value(json_object_get(floorData, "bx"));
      double by = json_number_value(json_object_get(floorData, "by"));
      double height = json_number_value(json_object_get(floorData, "height"));
      double weight = json_number_value(json_object_get(floorData, "weight"));
      double ktheta = json_number_value(json_object_get(floorData, "Ktheta"));
      if (ktheta != 0.)
	ndf = 6;

      floorHeight += height;
      double floorMass = weight/G;

      // create a node, element and 2 materials for each story
      json_t *node = json_object();
      json_t *element = json_object();
      json_t *material1 = json_object();
      json_t *material2 = json_object();
      json_t *material3 = json_object();

      json_object_set(node, "name", json_integer(index+2)); // +2 as we need node at 1 
      json_t *nodePosn = json_array();      
      json_array_append(nodePosn,json_real(0));
      json_array_append(nodePosn,json_real(0));
      json_array_append(nodePosn,json_real(floorHeight));
      json_object_set(node, "crd", nodePosn);
      json_object_set(node, "ndf", json_integer(ndf));
      if (eMass == false) 
	json_object_set(node, "mass", json_real(floorMass));

      if (ndf == 6) {
	json_t *constraints = json_array();
	json_array_append(constraints, json_integer(0));
	json_array_append(constraints, json_integer(0));
	json_array_append(constraints, json_integer(1));
	json_array_append(constraints, json_integer(1));
	json_array_append(constraints, json_integer(1));
	json_array_append(constraints, json_integer(0));
	json_object_set(node, "constraints", constraints);
      }

      json_array_append(nodes,node);

      if (eMass == true) {
	json_t *massNode = json_object();
	json_object_set(massNode, "name", json_integer(index+2 + 2 * numStory)); // +2 as ground nodes start at 1 and these nodes are above ground
	json_t *nodePosn = json_array();      
	json_array_append(nodePosn,json_real(massX));
	json_array_append(nodePosn,json_real(massY));
	json_array_append(nodePosn,json_real(floorHeight));
	json_object_set(massNode, "crd", nodePosn);
	json_object_set(massNode, "ndf", json_integer(ndf));
	json_object_set(massNode, "mass", json_real(floorMass));
	json_object_set(massNode, "constrainedToNode", json_integer(index+2));

	if (ndf == 6) {
	  json_t *constraints = json_array();
	  json_array_append(constraints, json_integer(0));
	  json_array_append(constraints, json_integer(0));
	  json_array_append(constraints, json_integer(1));
	  json_array_append(constraints, json_integer(1));
	  json_array_append(constraints, json_integer(1));
	  json_array_append(constraints, json_integer(0));
	  json_object_set(massNode, "constraints", constraints);
	}

	json_array_append(nodes,massNode);
      }

      if (eResponse == true) {
	json_t *respNode = json_object();
	json_object_set(respNode, "name", json_integer(index+2 + 4 * numStory)); // +2 as we need node at 1 
	json_t *nodePosn = json_array();      
	json_array_append(nodePosn,json_real(massX));
	json_array_append(nodePosn,json_real(massY));
	json_array_append(nodePosn,json_real(floorHeight));
	json_object_set(respNode, "crd", nodePosn);
	json_object_set(respNode, "ndf", json_integer(ndf));
	json_object_set(respNode, "constrainedToNode", json_integer(index+2));

	if (ndf == 6) {
	  json_t *constraints = json_array();
	  json_array_append(constraints, json_integer(0));
	  json_array_append(constraints, json_integer(0));
	  json_array_append(constraints, json_integer(1));
	  json_array_append(constraints, json_integer(1));
	  json_array_append(constraints, json_integer(1));
	  json_array_append(constraints, json_integer(0));
	  json_object_set(respNode, "constraints", constraints);
	}

	json_array_append(nodes,respNode);
      }


      json_object_set(element, "name", json_integer(eleTag));
      json_object_set(element, "type", json_string("shear_beam2d"));
      json_t *eleMats = json_array();      
      json_array_append(eleMats,json_integer(matTag));
      json_array_append(eleMats,json_integer(matTag+1));
      if (ndf == 3 || ndf == 6)
	json_array_append(eleMats,json_integer(matTag+2));
      json_object_set(element, "uniaxial_material", eleMats);
      json_t *eleNodes = json_array();      
      json_array_append(eleNodes,json_integer(index+1));
      json_array_append(eleNodes,json_integer(index+2));
      json_object_set(element, "nodes", eleNodes);
      
      json_object_set(material1,"name",json_integer(matTag));
      json_object_set(material1,"type",json_string("bilinear"));
      json_object_set(material1,"K",json_real(kx));
      json_object_set(material1,"Fy",json_real(Fyx));
      json_object_set(material1,"beta",json_real(bx));

      json_object_set(material2,"name",json_integer(matTag+1));
      json_object_set(material2,"type",json_string("bilinear"));
      json_object_set(material2,"K",json_real(ky));
      json_object_set(material2,"Fy",json_real(Fyy));
      json_object_set(material2,"beta",json_real(by));

      if (ndf == 6 || ndf == 3) {
	json_object_set(material3,"name",json_integer(matTag+2));
	json_object_set(material3,"type",json_string("elastic"));
	json_object_set(material3,"K",json_real(ktheta));
      }

      json_array_append(materials,material1);
      json_array_append(materials,material2);

      if (ndf == 3 || ndf == 6) 
	json_array_append(materials,material3);

      json_array_append(elements, element);

      // increment node, ele and mat tags
      nodeTag++;
      matTag+=3;
      eleTag++;
    }

    //
    // add node at the ground
    //

    json_t *node = json_object();
    json_object_set(node, "name", json_integer(1)); // +2 as we need node at 1 
    json_t *nodePosn = json_array();      
    json_array_append(nodePosn,json_real(0.0));
    json_array_append(nodePosn,json_real(0.0));
    json_array_append(nodePosn,json_real(0.0));
    json_object_set(node, "crd", nodePosn);
    json_object_set(node, "ndf", json_integer(ndf));
    json_t *constraints = json_array();
    for (int i=0; i<ndf; i++)
      json_array_append(constraints, json_integer(1));
    json_object_set(node, "constraints", constraints);

    json_array_append(nodes,node);

    json_t *typeDR = json_object_get(SIM,"dampingRatio");
    if (typeDR == NULL) {
      json_object_set(properties,"dampingRatio",json_real(0.02));
    }     else {
      std::cerr << "SETTING DAMPING RATIO\n" << json_dumps(typeDR, JSON_ENCODE_ANY);
      json_object_set(properties,"dampingRatio",typeDR);

    }
  }
  
  //
  // now the node mapping array, needed for EDP
  // create array, loop over nodes abd create the mapping
  //

  json_t *mappingArray = json_array();
  int nodeTag = 1; // node tags start at 0
  char floorString[16];
  
  for (int i=0; i<numNodes; i++) {

      sprintf(floorString,"%d",nodeTag-1); // floors start at 0
      json_t *nodeEntry = json_object();
      json_object_set(nodeEntry,"node",json_integer(nodeTag));
      json_object_set(nodeEntry,"cline",json_string("centroid"));
      //  itoa(floor, floorString, floor); NOT IN STANDARD
      json_object_set(nodeEntry,"floor",json_string(floorString));
      json_array_append(mappingArray, nodeEntry);
      
      if (eResponse == true) {
	if (i != 0) {
	  json_t *respNode = json_object();
	  json_object_set(respNode, "node", json_integer(nodeTag + 4 * numStory));
	  json_object_set(respNode,"cline",json_string("response"));
	  json_object_set(respNode,"floor",json_string(floorString));
	  json_array_append(mappingArray, respNode);
	} else { // assign response nodes cline to node 1
	  json_t *respNode = json_object();
	  json_object_set(respNode, "node", json_integer(nodeTag));
	  json_object_set(respNode,"cline",json_string("response"));
	  json_object_set(respNode,"floor",json_string(floorString));
	  json_array_append(mappingArray, respNode);
	}
      } else {
	json_t *nodeEntry = json_object();
	json_object_set(nodeEntry,"node",json_integer(nodeTag));
	json_object_set(nodeEntry,"cline",json_string("response"));
	//  itoa(floor, floorString, floor); NOT IN STANDARD
	json_object_set(nodeEntry,"floor",json_string(floorString));
	json_array_append(mappingArray, nodeEntry);
      }

      nodeTag++;
  }

  json_object_set(rootSAM,"NodeMapping",mappingArray);
  json_object_set(rootSAM,"numStory",json_integer(numStory));
  json_object_set(rootSAM,"ndm", json_integer(2));

  json_t *theRVs = json_object_get(SIM,"randomVar");

  json_t *rvObj;
  json_t *rvArray = json_array();
  int index = 0;
  json_array_foreach(theRVs, index, rvObj) {
    json_array_append(rvArray, rvObj);
  }

  json_object_set(rootSAM,"type",json_string("generic"));
  json_object_set(properties,"uniaxialMaterials",materials);
  
  json_object_set(geometry,"nodes",nodes);
  json_object_set(geometry,"elements",elements);

  json_object_set(rootSAM,"Properties",properties);
  json_object_set(rootSAM,"Geometry",geometry);
  json_object_set(rootSAM,"NodeMapping",mappingArray);
  json_object_set(rootSAM,"randomVar",rvArray);

  //
  // dump json to file
  //
  
  json_dump_file(rootSAM,filenameSAM,0);

  return 0;
}
