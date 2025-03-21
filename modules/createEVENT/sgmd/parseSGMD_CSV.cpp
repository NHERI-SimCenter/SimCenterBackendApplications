
/* ********************************************************************************************
**
** Appplication to parse .csv files provided by SGMD developers into SimCenter EVENT file
** the provided CSV files are of following format units m,m/s, m/s^2 for mag 7

Time,Vel FP,Acc FP,Dis FP,Vel FN,Acc FN,Dis FN,Vel Vert,Acc Vert,Dis Vert
0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.000000,0.000000,0.000000
0.001217,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.000000,0.000000,-0.000000
0.002433,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,-0.000000,0.000000,-0.000000

*********************************************************************************************** */

// written: fmk
// 02/25
//


#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <jansson.h>  

struct DataPoint {
    double time;
    double vel_fp, acc_fp, dis_fp;
    double vel_fn, acc_fn, dis_fn;
    double vel_vert, acc_vert, dis_vert;
};

std::vector<DataPoint> parseCSV(const std::string& filename) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return data;
    }

    // Read the header line and ignore it
    std::getline(file, line);

    // Read data lines
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        DataPoint point;
        char comma; // To discard commas while reading

        ss >> point.time >> comma
           >> point.vel_fp >> comma >> point.acc_fp >> comma >> point.dis_fp >> comma
           >> point.vel_fn >> comma >> point.acc_fn >> comma >> point.dis_fn >> comma
           >> point.vel_vert >> comma >> point.acc_vert >> comma >> point.dis_vert;

        data.push_back(point);
    }

    file.close();
    return data;
}

int main(int argc, const char **argv) {

  std::string inputFilename = argv[1]; // Get filename from arguments
  std::string outputFilename; 
  
  size_t lastdot = inputFilename.find_last_of(".");
  if (lastdot == std::string::npos) {
    outputFilename = inputFilename + ".json"; // No extension found, append .json
  } else
    outputFilename = inputFilename.substr(0, lastdot) + ".json";
  
  // parse inputfile
  std::vector<DataPoint> data = parseCSV(inputFilename);  
  
  //
  // create timeseries and pattern for 1 and 2 dof dirns
  //   - fill in some known data, then loop and add data to timeSeries
  //
  
  json_t *timeseriesObj1 = json_object();
  json_t *patternObj1 = json_object();
  json_t *timeseriesObj2 = json_object();
  json_t *patternObj2 = json_object();

  double factor = 1.0;
  json_object_set(timeseriesObj1,"name",json_string("dirn1"));
  json_object_set(timeseriesObj1,"type",json_string("Value"));    
  json_object_set(timeseriesObj1,"factor",json_real(factor));
  json_object_set(patternObj1,"timeSeries",json_string("dirn1"));
  json_object_set(patternObj1,"type",json_string("UniformAcceleration"));
  json_object_set(patternObj1,"dof",json_integer(1));
  
  json_object_set(timeseriesObj2,"name",json_string("dirn2"));
  json_object_set(timeseriesObj2,"type",json_string("Value"));    
  json_object_set(timeseriesObj2,"factor",json_real(factor));
  json_object_set(patternObj2,"timeSeries",json_string("dirn2"));
  json_object_set(patternObj2,"type",json_string("UniformAcceleration"));
  json_object_set(patternObj2,"dof",json_integer(1));
  
  json_t *dataArray1 = json_array();
  json_t *dataArray2 = json_array();        
  
  double dT = 0.0;
  int numPoints = 0;    
  
  // add data to data arrays
  for (const auto& dp : data) {
    dT = dp.time;
    json_array_append(dataArray1, json_real(dp.acc_fp));
    json_array_append(dataArray1, json_real(dp.acc_fn));            
    numPoints++;
  }

  dT = dT/(numPoints-1);
  
  //
  // create arrays for timeseries and load patterns
  //
  
  json_t *timeSeriesArray = json_array();
  json_t *patternArray = json_array();
  
  json_object_set(timeseriesObj1,"data",dataArray1);    
  json_object_set(timeseriesObj1,"dT",json_real(dT));
  json_object_set(timeseriesObj1,"numSteps",json_integer(numPoints));
  
  json_object_set(timeseriesObj2,"data",dataArray2);    
  json_object_set(timeseriesObj2,"dT",json_real(dT));
  json_object_set(timeseriesObj2,"numSteps",json_integer(numPoints));    
  
  json_array_append(timeSeriesArray, timeseriesObj1);
  json_array_append(timeSeriesArray, timeseriesObj2);    
  json_array_append(patternArray, patternObj1);
  json_array_append(patternArray, patternObj2);    
  
  //
  // create an event
  //
  
  json_t *event = json_object();
  
  // units
  double magnitude = 7.0;    
  json_t *units = json_object();
  json_object_set(units,"length",json_string("m"));
  json_object_set(units,"time",json_string("sec"));        
  
  json_object_set(event,"units",units);
  json_object_set(event,"pattern",patternArray);
  json_object_set(event,"timeSeries",timeSeriesArray);                
  json_object_set(event,"dT",json_real(dT));
  json_object_set(event,"numSteps",json_integer(numPoints));
  json_object_set(event,"magnitude",json_real(magnitude));            
  
  //
  // add Events array
  //
  
  json_t *eventsArray = json_array();
  json_array_append(eventsArray, event);    
  
  //
  // finally create JSON object and write to file
  //
  
  json_t *obj = json_object();
  json_object_set(obj,"Events", eventsArray);
  
  // write file
  json_dump_file(obj, outputFilename.c_str(), JSON_COMPACT);
    
  return 0;
}
