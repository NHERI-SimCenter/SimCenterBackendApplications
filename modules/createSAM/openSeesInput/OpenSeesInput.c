#include <jansson.h> 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

//#include <string>
//#include <sstream>
//#include <map>

int
main(int argc, char **argv) {

  // OpenSeesInput --filenameAIM file? --filenameEVENT file? --filenameSAM file? --filePath path? --fileName file? <--getRV>
  fprintf(stderr,"HELLO WORLD! %d\n", argc);

  char *filenameAIM = NULL;
  // NOT USED: char *filenameEVENT = argv[4]; 
  char *filenameSAM = NULL;
  // NOT USED: char *filePath = argv[8];
  char *fileName = NULL;  
  
  int arg = 1;
  while(arg < argc) {
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
    else if (strcmp(argv[arg], "--fileName") == 0)
      {
	arg++;
	fileName = argv[arg];
      }
    arg++;
  }

  fprintf(stderr, "SAM: %s\n", filenameSAM);
  
  if (filenameAIM == NULL || filenameSAM == NULL) {
    fprintf(stderr, "Missing filenameSAM or fileNameAIM arg\n");
    exit(-1);
  }

  json_error_t error;
  json_t *rootAIM = json_object();
  json_t *rootSAM = json_object();
  
    //
    // now the node mapping array, needed for EDP
    // create array, lop over nodes in list and fill in
    //

    json_t *mappingArray = json_array();

    //
    // load input file & get the array of nodes
    //

    rootAIM = json_load_file(filenameAIM, 0, &error);
    int length = 0;

    if (fileName == 0) { // not passed in input, should be in AIM
      json_t *theApplications = json_object_get(rootAIM,"Applications");
      if (theApplications == NULL) {
        fprintf(stderr, "AIM file missing Applications");
        exit(-1);
      }
      json_t *theModeling = json_object_get(theApplications,"Modeling");
      if (theModeling == NULL) {
        fprintf(stderr, "AIM file Applications missing Modeling");
        exit(-1);
      }
      json_t *theAppData = json_object_get(theModeling,"ApplicationData");
      if (theAppData == NULL) {
        fprintf(stderr, "AIM file Applications missing AppData");
        exit(-1);
      }
      json_t *theFileName = json_object_get(theAppData,"fileName");
      if (theFileName == NULL && !json_is_string(theFileName)) {
        fprintf(stderr, "AIM file AppData missing fileName");
        exit(-1);
      }
      const char *fileName2 = json_string_value(theFileName);
      length = json_string_length(theFileName);
      fileName = (char *)malloc((length+1)*sizeof(char));
      fprintf(stderr,"LEN %d\n", length);
      strncpy(fileName, fileName2, length);
      fileName[length]='\0';
    } else {
      length = strlen(fileName);
    }

    json_object_set(rootSAM,"mainScript",json_stringn(fileName, length));
    json_object_set(rootSAM,"type",json_string("OpenSeesInput"));

    json_t *theSIM = json_object_get(rootAIM,"Modeling");
    json_t *theResponseNodes = json_object_get(theSIM,"responseNodes");

    // check nodes exists
    if (theSIM == NULL ||  theResponseNodes == NULL) {
      fprintf(stderr,"OpenSeesInput - no nodes section found ");    
      return -1;
    }
    
    // loop over each node in nodes list, creating nodeMapping entry
    int index;
    json_t *intObj;
    int floor = 0;  // ground floor floor 0
    char floorString[10];

    json_array_foreach(theResponseNodes, index, intObj) {
      json_t *nodeEntry =json_object();
      int tag = json_integer_value(intObj);
      json_object_set(nodeEntry,"node",json_integer(tag));
      json_object_set(nodeEntry,"cline",json_string("response"));
      sprintf(floorString,"%d",floor);
      json_object_set(nodeEntry,"floor",json_string(floorString));
      floor++;
      json_array_append(mappingArray, nodeEntry);
    }

    // check for centroid nodes
    json_t *theCentroidNodes = json_object_get(theSIM,"centroidNodes");
    if (theCentroidNodes != NULL) {
      floor = 0;  // ground floor floor 0
      
      json_array_foreach(theCentroidNodes, index, intObj) {
        json_t *nodeEntry =json_object();
        int tag = json_integer_value(intObj);
        json_object_set(nodeEntry,"node",json_integer(tag));
        json_object_set(nodeEntry,"cline",json_string("centroid"));
        sprintf(floorString,"%d",floor);
        json_object_set(nodeEntry,"floor",json_string(floorString));
        floor++;
        json_array_append(mappingArray, nodeEntry);
      }
    }
    
    json_object_set(rootSAM,"NodeMapping",mappingArray);

    json_t *theDampingRatio = json_object_get(theSIM,"dampingRatio");
    
    // add #story and ndm
    int nStory = floor -1;

    json_object_set(rootSAM,"numStory",json_integer(nStory));

    json_t *ndm = json_object_get(theSIM, "ndm");
    json_object_set(rootSAM,"ndm", ndm);

    json_t *ndf = json_object_get(theSIM, "ndf");
    if (ndf != NULL)
      json_object_set(rootSAM,"ndf", ndf);

    json_t *inputType = json_object_get(theSIM, "type");
    if (inputType != NULL) {
      json_object_set(rootSAM,"subType", inputType);
    }
    json_t *useDamping = json_object_get(theSIM, "useDamping");
    if (useDamping != NULL) {
      json_object_set(rootSAM,"useDamping", useDamping);
    }
    json_t *theRVs = json_object_get(theSIM,"randomVar");

    // check nodes exists
    if (theSIM == NULL ||  theRVs == NULL) {
      fprintf(stdout,"OpenSeesInput - no randomVar section found ");    
      return -1;
    }

    // loop over each node in nodes list, creating nodeMapping entry
    json_t *rvObj;
    json_t *rvArray = json_array();
    json_array_foreach(theRVs, index, rvObj) {
      //      json_t *nodeEntry = json_object();
      //      int tag = json_integer_value(intObj);
      json_array_append(rvArray, rvObj);
    }
    json_object_set(rootSAM,"randomVar",rvArray);
    if (theDampingRatio != NULL)
      json_object_set(rootSAM,"dampingRatio",theDampingRatio);      
    
    //
    // dump json to file
    //
    json_dump_file(rootSAM,filenameSAM,0);

    return 0;
}
