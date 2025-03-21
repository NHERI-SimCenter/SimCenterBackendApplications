
/* ********************************************************************************************
**
** Appplication to parse .csv files provided by SGMD developers into EventGrid.csv file format
**   entails: creating EventGrid.csv and for each site .csv and .json

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

std::vector<DataPoint>
parseGM_CSV(const std::string& filename, int *result) {
  
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
	*result = -1;
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


int
writeJSON(std::string inputFilename, std::string outputFilename,double elev) {

  // parse inputfile
  int result = 0;
  std::vector<DataPoint> data = parseGM_CSV(inputFilename, &result);  

  if (result != 0)
    return result;
  
  json_t *dataArray1 = json_array();
  json_t *dataArray2 = json_array();        
  
  double dT = 0.0;
  int numPoints = 0;    
  
  // add data to data arrays
  // 0.10197162129779283 is conversion from m/sec^2 to g 1/9.8whatever
  for (const auto& dp : data) {
    dT = dp.time;
    json_array_append(dataArray1, json_real(dp.acc_fp*0.10197162129779283));
    json_array_append(dataArray2, json_real(dp.acc_fn*0.10197162129779283));            
    numPoints++;
  }

  dT = dT/(numPoints-1);
  
  //
  // create arrays for timeseries and load patterns
  //

  json_t *obj = json_object();
  json_object_set(obj,"dT",json_real(dT));
  json_object_set(obj,"elev",json_real(elev));  
  json_object_set(obj,"data_x",dataArray1);
  json_object_set(obj,"data_y",dataArray2);      
  
  // write file
  json_dump_file(obj, outputFilename.c_str(), JSON_COMPACT);
    
  return result;
}

int main(int argc, const char **argv) {

  std::string inputFilename = argv[1]; // Get filename from arguments
  std::string outputDirname = argv[2];
  std::string outputEventFile = outputDirname + "/EventGrid.csv";

  std::ifstream inputFile(inputFilename);
  std::ofstream eventFile(outputEventFile);
  
  std::string line;
  if (!inputFile.is_open()) {
    std::cerr << "Error: Could not open file " << inputFilename << std::endl;
    exit(0);
  }
  if (!eventFile.is_open()) {
    std::cerr << "Error: Could not open file " << outputEventFile << std::endl;
    exit(0);    
  }  

  // Read the header line and ignore it
  std::getline(inputFile, line);

  // Write Header for EventGrid
  eventFile << "GP_file,Longitude,Latitude\n";
  
  // Read data lines
  // sta,lon,lat,elev
  // S_01_01,-122.1836,38.3523,257.5273

  double minLat = 91;
  double maxLat = -91;
  double minLng = 361;
  double maxLng = -361;
  
  
  while (std::getline(inputFile, line)) {
    std::stringstream ss(line);
    std::string stationName;
    double lat, lng, elev;
    char comma; // To discard commas while reading

    std::getline(ss, stationName, ',');  // Read first value as string
    ss >> lng;
    ss.ignore();  // Ignore comma
    ss >> lat;
    ss.ignore();  // Ignore comma
    ss >> elev;

    if (lat < minLat) minLat = lat;
    if (lat > maxLat) maxLat = lat;    
    if (lng < minLng) minLng = lng;
    if (lng > maxLng) maxLng = lng;    
    
    // create the json file for the entry, if error don't add entry
    
    std::string outputJSONFilename = outputDirname + "/" + stationName + ".json";
    std::string inputCSVFilename =  stationName + ".csv";
    /*
    if (writeJSON(inputCSVFilename, outputJSONFilename,elev) == 0) {

      // write data to EventGrid file
      eventFile << stationName << ".csv," << lng << "," <<  lat << "\n";
      
      // create the csv file for the entry
      std::string outputCSVFilename = outputDirname + "/" + stationName + ".csv";    
      std::ofstream outputCSVFile(outputCSVFilename);
      if (!outputCSVFile.is_open()) {
	std::cerr << "Error: Could not open file " << outputCSVFilename << std::endl;
      }
      outputCSVFile << "TH_file\n" << stationName;
      outputCSVFile.close();      
    }
    */
  }
  
  inputFile.close();
  eventFile.close();

  std::cerr << "Lat(min,max) Long(min,max)";
  std::cerr << minLat << " " << maxLat << " " << minLng << " " << maxLng <<"\n";
}
  
  
