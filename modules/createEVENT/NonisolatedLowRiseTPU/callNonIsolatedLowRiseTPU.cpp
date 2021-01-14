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
callNonIsolatedLowRise_TPU(const char *arrangement,   
			   const char *heightBreadth, 
			   const char *density, 
			   const char *roofType,
			   const char *incidenceAngle,
			   const char *outputFilename)
{

  curl_global_init(CURL_GLOBAL_ALL);

  std::cerr << "arrangement: " << arrangement << " heightBreadth: " << heightBreadth;
  std::cerr << " density: " << density << " roofType: " << roofType;
  std::cerr << " angle: " << incidenceAngle << " \n";
  
  CURLcode ret;
  CURL *hnd;
  struct curl_slist *slist1;

  slist1 = NULL;

  //
  // create url
  //
  char url[128];
  strcpy(url,"http://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/grouplowrise/Cp_ts_10");    

  if (strcmp(heightBreadth,"3:8") == 0)
    strcat(url,"3");
  else if (strcmp(heightBreadth,"6:8") == 0)
    strcat(url,"6");
  else
    strcat(url,"9");

  if (strcmp(heightBreadth,"3:8") == 0)
    strcat(url,"3");
  else if (strcmp(heightBreadth,"6:8") == 0)
    strcat(url,"6");
  else
    strcat(url,"9");
  
  
  if (strcmp(density,".00") == 0)
    strcat(url, "1");
  else if (strcmp(density,".10") == 0)
    strcat(url, "2");
  else if (strcmp(density, ".15") == 0)
    strcat(url, "3");
  else if (strcmp(density, ".20") == 0)
    strcat(url, "4");
  else if (strcmp(density, ".25") == 0)
    strcat(url, "5");
  else if (strcmp(density, ".30") == 0)
    strcat(url, "6");
  else if (strcmp(density, ".40") == 0)
    strcat(url, "8");
  else if (strcmp(density, ".50") == 0)
    strcat(url, "A");
  else if (strcmp(density, ".60") == 0)
    strcat(url, "C");
  
  if (strcmp(arrangement,"Regular") == 0)
    strcat(url, "1");
  else if (strcmp(arrangement,"Staggered") == 0)
    strcat(url, "2");
  else
    strcat(url, "3");


  if (strcmp(heightBreadth,"3:8") == 0)
    strcat(url,"3_deg0");
  else if (strcmp(heightBreadth,"6:8") == 0)
    strcat(url,"6_deg0");
  else
    strcat(url,"9_deg0");
  
  // incidence angle
  if (strcmp(incidenceAngle,"00") == 0)
    strcat(url,"00.mat");
  else if (strcmp(incidenceAngle, "23") == 0) 
    strcat(url,"23.mat");
  else if (strcmp(incidenceAngle, "45") == 0)
    strcat(url,"45.mat");
  else
    strcat(url,"90.mat");    
  
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


