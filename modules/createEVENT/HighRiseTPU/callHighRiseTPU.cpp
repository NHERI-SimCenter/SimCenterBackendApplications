#include <curl/curl.h>
#include <iostream>
#include <string.h>
#include <sstream>
#include <stdlib.h>

static 
size_t write_data(void *ptr, size_t size, size_t nmemb, void *stream)
{
  size_t written = fwrite(ptr, size, nmemb, (FILE *)stream);
  return written;
}


/*
 * function to call TU database to obtain experimental test data
 */


int 
callHighRise_TPU(const char *alpha,   
		const char *breadthDepth, 
		const char *depthHeight, 
		int incidenceAngle,
		const char *outputFilename)
{

  curl_global_init(CURL_GLOBAL_ALL);

  CURLcode ret;
  CURL *hnd;
  struct curl_slist *slist1;

  slist1 = NULL;

  //
  // create url
  //
  char url[128];
  strcpy(url,"http://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/highrise/Test_Data/T");    

  if (strcmp(breadthDepth,"1:1") == 0)
    strcat(url,"11");
  else if (strcmp(breadthDepth,"2:1") == 0)
    strcat(url,"21");
  else
    strcat(url,"31");

  if (strcmp(depthHeight,"1:1") == 0)
    strcat(url, "1");
  else if (strcmp(depthHeight, "1:2") == 0)
    strcat(url, "2");
  else if (strcmp(depthHeight, "1:3") == 0)
    strcat(url, "3");
  else if (strcmp(depthHeight, "1:4") == 0)
    strcat(url, "4");
  else if (strcmp(depthHeight, "1:5") == 0)
    strcat(url, "5");
  
  if (strcmp(alpha,  "1/4") == 0)
    strcat(url, "_4_0");
  else
    strcat(url, "_6_0");
  
  // incidence angle
  if (incidenceAngle == 0) 
    strcat(url,"00");
  else if (incidenceAngle == 5) 
    strcat(url,"05");
  else if (incidenceAngle == 10) 
    strcat(url,"10");
  else if (incidenceAngle == 15) 
    strcat(url,"15");
  else if (incidenceAngle == 20) 
    strcat(url,"20");
  else if (incidenceAngle == 25) 
    strcat(url,"25");    
  else if (incidenceAngle == 30) 
    strcat(url,"30");
  else if (incidenceAngle == 35) 
    strcat(url,"35");
  else if (incidenceAngle == 40) 
    strcat(url,"40");
  else if (incidenceAngle == 45) 
    strcat(url,"45");
  else if (incidenceAngle == 50) 
    strcat(url,"50");
  else if (incidenceAngle == 55) 
    strcat(url,"55");
  else if (incidenceAngle == 60) 
    strcat(url,"60");
  else if (incidenceAngle == 65) 
    strcat(url,"65");
  else if (incidenceAngle == 70) 
    strcat(url,"70");
  else if (incidenceAngle == 75) 
    strcat(url,"75");
  else if (incidenceAngle == 80) 
    strcat(url,"80");
  else if (incidenceAngle == 85) 
    strcat(url,"85");
  else if (incidenceAngle == 90) 
    strcat(url,"90");    
  
  else {
    std::cerr << "Unknown incidence angle : " << incidenceAngle << "\n";
    return -3;
  }
  
  strcat(url,"_1.mat");


  std::cerr << url << "\n";
  
  //
  // set curl options
  //
  
  hnd = curl_easy_init();
  curl_easy_setopt(hnd, CURLOPT_URL, url);
  curl_easy_setopt(hnd, CURLOPT_NOPROGRESS, 1L);
    
  /*
  curl_easy_setopt(hnd, CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t)326);
  curl_easy_setopt(hnd, CURLOPT_USERAGENT, "curl/7.54.0");
  curl_easy_setopt(hnd, CURLOPT_HTTPHEADER, slist1);
  curl_easy_setopt(hnd, CURLOPT_MAXREDIRS, 50L);
  curl_easy_setopt(hnd, CURLOPT_HTTP_VERSION, (long)CURL_HTTP_VERSION_2TLS);
  curl_easy_setopt(hnd, CURLOPT_COOKIEJAR, "compressed");
  curl_easy_setopt(hnd, CURLOPT_CUSTOMREQUEST, "POST");
  curl_easy_setopt(hnd, CURLOPT_TCP_KEEPALIVE, 1L);
  */

  //
  // store the output data in a file
  //

  FILE *pagefile = fopen(outputFilename, "wb");
  if(pagefile) {
    curl_easy_setopt(hnd, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(hnd, CURLOPT_WRITEDATA, pagefile);
  } else {
      std::cerr << "Could not open file: " << outputFilename << "\n";
      return -4;
  }

  //
  // go get data
  //
  
  ret = curl_easy_perform(hnd);

  if (ret != CURLE_OK) {
    const char *str = curl_easy_strerror(ret);
    std::cerr <<  "tpuCall FAILED" << str << "\n";
    return -1;
  }

  fclose(pagefile);

  curl_easy_cleanup(hnd);
  hnd = NULL;
  curl_slist_free_all(slist1);
  slist1 = NULL;

  return (int)ret;
}
		 

#ifdef _TESTIT

//
// example main:
//

int main(int argc, char *argv[])
{
  return callHighRise_TPU("1/4","2:1","1:3",10,"test.mat");
}

#endif


