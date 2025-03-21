
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
parseGM_CSV(const std::string& filename, int &result) {

    std::vector<DataPoint> data;
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
	result = -1;
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

    result = 0;
    std::cerr << " RES: " << result << " ";
    return data;
}


double
getMax(std::string inputFilename, int &result) {

  // parse inputfile
  std::vector<DataPoint> data = parseGM_CSV(inputFilename, result);  

  std::cerr << " RES: " << result << " ";
  if (result != 0) {
    return 0.0;
  } 
  
  json_t *dataArray1 = json_array();
  json_t *dataArray2 = json_array();        
  
  double maxValue;
  for (const auto& dp : data) {
    double fp = std::abs(dp.acc_fp*0.10197162129779283);
    double fn = std::abs(dp.acc_fp*0.10197162129779283);
    if (fp > maxValue) maxValue = fp;
    if (fn > maxValue) maxValue = fn;    
  }

  std::cerr << " MAX: " << maxValue << " error end getMAx " << result << " ";
  return maxValue;
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


    int error;
    std::cerr << inputCSVFilename << "\t";
    double pga = getMax(inputCSVFilename, error);
    
    std::cerr << " error " << error << " PGA " << pga << "\t";
    
    if (error == 0) {// write data to EventGrid file

      std::cerr << " PGA2 " << pga << "\t";
      
      eventFile << stationName << ".csv," << lng << "," <<  lat << "\n";
      
      // create the csv file for the entry
      std::string outputCSVFilename = outputDirname + "/" + stationName + ".csv";
      std::cerr << outputCSVFilename << "\n";      
      std::ofstream outputCSVFile(outputCSVFilename);
      if (!outputCSVFile.is_open()) {
	std::cerr << "Error: Could not open file " << outputCSVFilename << std::endl;
      } else {
	outputCSVFile << "PGA\n" << pga;
	outputCSVFile.close();
      }
    }

  }
  
  inputFile.close();
  eventFile.close();

  std::cerr << "Lat(min,max) Long(min,max)";
  std::cerr << minLat << " " << maxLat << " " << minLng << " " << maxLng <<"\n";
}
  

