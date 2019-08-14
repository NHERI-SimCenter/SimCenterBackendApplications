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
callLowRise_TPU(const char *shape,   
		const char *heightBreadth, 
		const char *depthBreadth, 
		const char *pitch,
		int incidenceAngle,
		const char *outputFilename)        
{   
  curl_global_init(CURL_GLOBAL_ALL);

  CURLcode ret;
  CURL *hnd;
  struct curl_slist *slist1;

  slist1 = NULL;
  char url[128];
  
  if ((strcmp(shape,"flat") == 0) || (strcmp(shape,"gable") == 0)) {

    // "http://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/lowrise/Cp_ts_g12020500.mat"
    strcpy(url,"http://www.wind.arch.t-kougei.ac.jp/info_center/windpressure/lowrise/Cp_ts_g");    

    // depthBreadth
    if (strcmp(depthBreadth,"2:2") == 0)
      strcat(url,"08");
    else if (strcmp(depthBreadth,"3:2") == 0)
      strcat(url,"12");
    else if (strcmp(depthBreadth,"5:2") == 0)
      strcat(url,"20");
    else {
      std::cerr << "Unknown depth/breadth : " << depthBreadth << "\n";
      return -1;
    }

    // heightBreadth addition
    if (strcmp(heightBreadth,"1:4") == 0)
      strcat(url,"02");
    else if (strcmp(heightBreadth,"2:4") == 0)
      strcat(url,"04");
    else if (strcmp(heightBreadth,"3:4") == 0)
      strcat(url,"06");
    else if (strcmp(heightBreadth,"4:4") == 0)
      strcat(url,"08");
    else {
      std::cerr << "Unknown height/breadth : " << heightBreadth << "\n";
      return -2;
    }

    // pitch
    if (strcmp(pitch,"0.0") == 0) 
      strcat(url,"00");
    else if (strcmp(pitch,"4.8") == 0) 
      strcat(url,"05");
    else if (strcmp(pitch,"9.4") == 0) 
      strcat(url,"10");
    else if (strcmp(pitch,"14.5") == 0)
      strcat(url,"14");
    else if (strcmp(pitch,"18.4") == 0)
      strcat(url,"18");
    else if (strcmp(pitch,"21.8") == 0)
      strcat(url,"22");
    else if (strcmp(pitch,"26.7") == 0)
      strcat(url,"27");
    else if (strcmp(pitch,"30.0") == 0)
      strcat(url,"30.0");
    else if (strcmp(pitch,"45.0") == 0)
      strcat(url,"45");
    else {
      std::cerr << "Unknown pitch : " << pitch << "\n";
      return -3;
    }

    // incidence angle
    if (incidenceAngle == 0) 
      strcat(url,"00");
    else if (incidenceAngle == 15) 
      strcat(url,"15");
    else if (incidenceAngle == 45) 
      strcat(url,"45");
    else if (incidenceAngle == 60) 
      strcat(url,"60");
    else if (incidenceAngle == 75) 
      strcat(url,"75");
    else if (incidenceAngle == 90) 
      strcat(url,"90");
    else {
      std::cerr << "Unknown incidence angle : " << incidenceAngle << "\n";
      return -3;
    }


    strcat(url,".mat");
  } else {
      std::cerr << "Unknown shape : " << shape << "\n";
      return -4;
  }

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


  ret = curl_easy_perform(hnd);

  if (ret != CURLE_OK) {
    const char *str = curl_easy_strerror(ret);
    std::cerr <<  "tpuCall FAILED" << str << "\n";
    return -1;
  }

  curl_easy_cleanup(hnd);
  hnd = NULL;
  curl_slist_free_all(slist1);
  slist1 = NULL;

  return (int)ret;
}
		 

/* 
 * example main:
 *

int main(int argc, char *argv[])
{
  return callLowRise_TPU("flat","2:4","2:2","0.0",15,"test.mat");
}
*/

