/*
 * getMultipleLocationWindSpeedsATC.c 
 * Program to make call to ATC's online hazard database https://hazards.atcouncil.org
 * to obtain site specific windspeed given the sites latitude and longitude. The program
 * takes 2 mandatory args (latitude and longitude), and some optional args: 
 * version, outfile. Program is called as follows:
 *
 *    programName latitude longitude <--version 7-10> <--outfile filename>
 *
 * --version: ATC version, valid options: 7-10 or 7-16 (default 7-16)
 * --outfile: output file name (default: out.json)
 *
 * written: fmckenna@berkeley.edu
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <curl/curl.h>
#include <jansson.h>

static size_t write_data(void *ptr, size_t size, size_t nmemb, void *stream)
{
  size_t written = fwrite(ptr, size, nmemb, (FILE *)stream);
  return written;
}

struct MemoryStruct {
  char *memory;
  size_t size;
};
 
static size_t
WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
  size_t realsize = size * nmemb;
  struct MemoryStruct *mem = (struct MemoryStruct *)userp;
 
  char *ptr = realloc(mem->memory, mem->size + realsize + 1);
  if(ptr == NULL) {
    /* out of memory! */ 
    printf("not enough memory (realloc returned NULL)\n");
    return 0;
  }
 
  mem->memory = ptr;
  memcpy(&(mem->memory[mem->size]), contents, realsize);
  mem->size += realsize;
  mem->memory[mem->size] = 0;
 
  return realsize;
}

int main(int argc, char *argv[])
{
  CURLcode ret;
  CURL *hnd;

  struct MemoryStruct chunk;
  
  //
  // parse args
  //

  

  // check correct #
  if (argc < 3) {
    fprintf(stderr,"Correct usage: program inputCSV outputCSV <--year 7-10>\n");
    exit(-1);
  }

  const char *inputFile = argv[1];
  const char *outputFile = argv[2];  
  char *version = (char *)malloc(5*sizeof(char));

  strcpy(version,"7-16");
  
  int currentArg = 3;
  while (currentArg < argc-1) { // -1 as each option requires 2 args
    if (strcmp(argv[currentArg],"--version") == 0) {
      currentArg++;
      if (strcmp(argv[currentArg],"7-10") == 0)
	strcpy(version,"7-10");
      else
	strcpy(version,"7-16");
      currentArg++;
    } else {
      fprintf(stderr,"UNKNOWN ARG: %s\n",argv[currentArg]);
      exit(-1);
    }
  }  

  
  FILE *fpIN, *fpOUT;
  fpIN = fopen(inputFile, "r");
  fpOUT = fopen(outputFile, "w");
  char  line[1056];
  char *URI = (char *)malloc(1000*sizeof(char)); URI[0]='\n';
  char *outfile = (char *)malloc(10*sizeof(char)); strcpy(outfile,"out.tmp");  
  const char *id;
  const char *latitude;
  const char *longitude;

  int count = 0;
  while (fgets(line, sizeof(line), fpIN) != NULL)
    {
      line[strcspn(line, "\r\n")] = 0; // remove line endings
      const char* id = strtok(line, ",");
      const char* latitude = strtok(NULL, ",");
      const char* longitude = strtok(NULL, ",");
      
      printf("%d: %s %s %s\n",count, id, latitude, longitude);
      
      if (count == 0) {
	
	fprintf(fpOUT,"%s,%s,%s, DWS1, DWS2, DWS3, DWS4\n",id, latitude, longitude);
	
      } else {


	double level1 = 0.;
	double level2 = 0.;
	double level3 = 0.;
	double level4 = 0.;
	
	//
	// create URI
	//

	strcpy(URI,"https://api-hazards.atcouncil.org/wind.json?group=asce");
	strcat(URI,version);
	strcat(URI,"&subgroup=&siteclass=&lat=");
	strcat(URI,latitude);
	strcat(URI,"&lng=");
	strncat(URI,longitude, strlen(longitude));
	
	//	fprintf(stderr,"calling: %s\n",URI);
	
	//
	// setup curl call
	//

	hnd = curl_easy_init();
	curl_easy_setopt(hnd, CURLOPT_URL, URI);
	curl_easy_setopt(hnd, CURLOPT_NOPROGRESS, 1L);
	curl_easy_setopt(hnd, CURLOPT_USERAGENT, "curl/7.54.0");
	curl_easy_setopt(hnd, CURLOPT_MAXREDIRS, 50L);
	curl_easy_setopt(hnd, CURLOPT_HTTP_VERSION, (long)CURL_HTTP_VERSION_2TLS);
	curl_easy_setopt(hnd, CURLOPT_TCP_KEEPALIVE, 1L);

	chunk.memory = malloc(1);  /* will be grown as needed by the realloc above */ 
	chunk.size = 0;
	
	curl_easy_setopt(hnd, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
	curl_easy_setopt(hnd, CURLOPT_WRITEDATA, (void *)&chunk);

	//
	// make curl call
	//
	
	ret = curl_easy_perform(hnd);

	curl_easy_cleanup(hnd);

	json_error_t jsonError;
	
	json_t *root = json_loads(chunk.memory, chunk.size, &jsonError);
	json_t *datasets = json_object_get(root,"datasets");
	if (datasets != NULL) {
	  int numData = json_array_size(datasets);

	  int numFound = 0;
	  int arrayCount = 0;
	  while (arrayCount < numData && numFound < 4) {
	    json_t *ele = json_array_get(datasets, arrayCount);
	    json_t *grp = json_object_get(ele,"group");
	    if ((grp != NULL) &&
		(strcmp(json_string_value(grp),"ASCE 7-16") == 0)) {
	      json_t *name = json_object_get(ele,"name");
	      if (name != NULL) {
		const char *nameValue = json_string_value(name);
		if (strcmp(nameValue,"Risk Category I") == 0) {
		  json_t *data = json_object_get(ele,"data");
		  json_t *value = json_object_get(data,"value");
		  level1 = json_real_value(value);
		  numFound++;
		}
		else if (strcmp(nameValue,"Risk Category II") == 0) {
		  json_t *data = json_object_get(ele,"data");
		  json_t *value = json_object_get(data,"value");
		  level2 = json_real_value(value);
		  numFound++;
		}
		else if (strcmp(nameValue,"Risk Category III") == 0) {
		  json_t *data = json_object_get(ele,"data");
		  json_t *value = json_object_get(data,"value");
		  level3 = json_real_value(value);
		  numFound++;
		}
		else if (strcmp(nameValue,"Risk Category IV") == 0) {
		  json_t *data = json_object_get(ele,"data");
		  json_t *value = json_object_get(data,"value");
		  level4 = json_real_value(value);
		  numFound++;
		}		
	      }
	    }
	    arrayCount++;
	  }

	  // print results tyo file
	  
	  fprintf(fpOUT, "%s,%s,%s,%f,%f,%f,%f\n",id, latitude, longitude, level1, level2, level3, level4);

	}
	
	//
	// cleanup
	//
	
	free(chunk.memory);
	
      }
      count++;
    }
  fclose(fpOUT);
  fclose(fpIN);
  
  free(version);
  free(URI);

  // done
  return (int)ret;
}
