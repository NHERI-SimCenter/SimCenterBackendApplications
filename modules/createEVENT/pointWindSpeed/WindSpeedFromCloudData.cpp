// createEDP.cpp
// purpose - given a building, return an WIND_EVENT
// written: fmckenna

#include <iostream>

#include <stdio.h>
#include <stdlib.h>

#include <jansson.h>     // for writing json
#include <nanoflann.hpp> // for searching for nearest point

#include <map>
#include <string>
#include <cstring>

using namespace nanoflann;

struct locations {
  locations():x(0),y(0) {}
  locations(double speed,double a,double b):windSpeed(speed),x(a),y(b) {}
  double windSpeed;
  double x;
  double y;
};


template <typename T>
struct PointCloud
{
  struct Point
  {
    Point(): pointTag(0),x(0.),y(0.) {}
    Point(int tag, T(a), T(b)): pointTag(tag),x(a),y(b) {}
    int pointTag;
    T  x,y;
  };
  
  std::vector<Point>  pts;
  
  inline size_t kdtree_get_point_count() const { return pts.size(); }
  
  inline T kdtree_distance(const T *p1, const size_t idx_p2,size_t /*size*/) const
  {
    const T d0=p1[0]-pts[idx_p2].x;
    const T d1=p1[1]-pts[idx_p2].y;
    return d0*d0+d1*d1;
  }
  
  inline T kdtree_get_pt(const size_t idx, int dim) const
  {
    if (dim==0) return pts[idx].x;
    else return pts[idx].y;
  }
  
  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

int main(int argc, char **argv) {

  const char *filenameAIM =0;
  const char *filenameEVENT =0;
  const char *filenameWindCloudData =0;

  bool getRV = false;

  // 
  // parse the inputs
  //

  int arg = 1;
  while (arg < argc) {
    if (strcmp(argv[arg], "--filenameAIM") ==0) {
      arg++;
      filenameAIM = argv[arg];
    }
    else if (strcmp(argv[arg], "--filenameEVENT") ==0) {
      arg++;
      filenameEVENT = argv[arg];
    }
    else if (strcmp(argv[arg], "--filenameWindSpeedCloudData") ==0) {
      arg++;
      filenameWindCloudData = argv[arg];
    }
    else if (strcmp(argv[arg], "--getRV") ==0) {
      getRV = true;
    } else {
      std::cerr << "WIndSpeedFRomCloudData - unknown arg: " << argv[arg] << "\n";
    }
    arg++;
  }

  // check inputs all there
  if(filenameWindCloudData == 0 || filenameAIM == 0 || filenameEVENT == 0) {
    if (filenameWindCloudData == 0)
      std::cerr << "no cloudDataFile file\n";
    else if (filenameAIM == 0)
      std::cerr << "no BIM file\n";
    else
      std::cerr << "no EVENT file\n";

    exit(-1);
  } 

  //
  // only do anything if getRV is passed
  // 

  if (getRV == true) {

    std::map<int, locations> stationLocations;
    PointCloud<float> cloud;
    
    //
    // first parse wind data file and put each point into the cloud of points
    //
    
    // open file
    json_error_t error;
    json_t *input = json_load_file(filenameWindCloudData, 0, &error);
    if (input == NULL) {
      std::cerr << "FATAL ERROR - input file does not exist or not a JSON file\n";
      std::cerr << filenameWindCloudData;
      exit(-1);
    }
    
    json_t *windDataArray = json_object_get(input, "wind"); 
    if (windDataArray == NULL) {
      std::cerr << "FATAL ERROR - no wind data\n";
      exit(-1);
    }
    
    // parse each event in input:
    int indexJ;
    json_t *windDataPoint;
    int numEDP = 0;
    int count = 0;
    json_array_foreach(windDataArray, indexJ, windDataPoint) {
      json_t *latJO = json_object_get(windDataPoint,"lat");
      json_t *longJO = json_object_get(windDataPoint,"long");
      json_t *wsJO = json_object_get(windDataPoint,"windSpeed");
      double lat = json_number_value(latJO);
      double longitude = json_number_value(longJO);
      double ws = json_number_value(wsJO);
      
      stationLocations[count]=locations(ws,lat,longitude);
      cloud.pts.resize(count+1);
      cloud.pts[count].pointTag = count;
      cloud.pts[count].x = lat;
      cloud.pts[count].y = longitude;
      count++;
    }   
    
    std::cerr << "NUMBER CLOUD POINTS: " << count << "\n";
    
    //
    // now parse the bim file for the location and 
    //
    
    json_t *root = json_load_file(filenameAIM, 0, &error);
    
    if(!root) {
      printf("ERROR reading BIM file: %s\n", filenameAIM);
    }
    
    json_t *GI = json_object_get(root,"GeneralInformation");
    json_t *location = json_object_get(GI,"location");
    
    float buildingLoc[2];
    buildingLoc[0] = json_number_value(json_object_get(location,"latitude"));
    buildingLoc[1] = json_number_value(json_object_get(location,"longitude"));
    
    json_object_clear(root);  
    
    //
    // now find nearest point in the cloud
    //
    
    // build the kd tree
    typedef KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<float, PointCloud<float> >,
      PointCloud<float>,
      2
      > my_kd_tree_t;
    
    my_kd_tree_t   index(2, cloud, KDTreeSingleIndexAdaptorParams(10) );
    index.buildIndex();
    
    //
    // do a knn search to find nearest point
    //
    
    long unsigned int num_results = 1;
    size_t ret_index;
    float out_dist_sqr;
    nanoflann::KNNResultSet<float> resultSet(num_results);
    resultSet.init(&ret_index, &out_dist_sqr);
    index.findNeighbors(resultSet, &buildingLoc[0], nanoflann::SearchParams(10));
    
    // 
    // create the event
    //
    
    int pointTag = ret_index;
    
    std::map<int, locations>::iterator stationIter;
    
    stationIter = stationLocations.find(pointTag);
    std::string stationName;
    
    double windSpeed = 0;
    if (stationIter != stationLocations.end()) {
      //std::cerr << stationIter->second.station;
      windSpeed = stationIter->second.windSpeed;
      std::cerr << windSpeed;
    }
    
    //
    // add acceleration record at station to event array in events file
    //
    
    root = json_load_file(filenameEVENT, 0, &error);
    json_t *eventsArray;
    
    if(!root) {
      root = json_object();    
      eventsArray = json_array();    
      json_object_set(root,"Events",eventsArray);
      json_t *rvArray=json_array();    
      json_object_set(root,"RandomVariables",rvArray);
      
    } else {
      eventsArray = json_object_get(root,"Events");
    }
    
    json_t *newEvent = json_object();
    
    json_object_set(newEvent,"type",json_string("Wind"));
    json_object_set(newEvent,"subtype",json_string("WindSpeed"));
    json_object_set(newEvent,"peak_wind_gust_speed",json_real(windSpeed));
    json_object_set(newEvent,"unit",json_string("mph"));
    
    json_array_append(eventsArray, newEvent);
    
    json_dump_file(root,filenameEVENT,0);
    json_object_clear(root);

    std::cerr << "Wrote EVENT file: " << filenameEVENT << "\n";
  }

  return 0;
}
