#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cmath>

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char* argv[]) {

  std::string paramsFileName = argv[1];
  std::string driverFileName = argv[2];
  std::string multiModelString = argv[3];

  std::string line;
  std::ifstream infile(paramsFileName);

  while (std::getline(infile, line)) {
    if (line.find(multiModelString) != std::string::npos) {
      std::string modelIndex;
      size_t start = line.find_last_of(' ');
      if (start != std::string::npos) modelIndex = line.substr(start + 1);
      float value = std::stof(modelIndex);
      int roundedValue = std::round(value);
      modelIndex = std::to_string(roundedValue);

      if (!modelIndex.empty()) {
        std::string multiModelDriverFileName = multiModelString + "_" + modelIndex + "_" + driverFileName;
        std::string command = "";
        #ifdef _WIN32
          command = "cmd /c " + multiModelDriverFileName;
        #else
          command = "./" + multiModelDriverFileName;
        #endif
        std::system(command.c_str());
        return 0;
      }
    }
  }
}
