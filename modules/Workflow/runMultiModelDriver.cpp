// #include <string>
// #include <filesystem>
// #include <iostream>
// #include <cmath>

// int main() {
//     // std::string paramsFileName(argv[1]);

//     // if (!std::filesystem::exists(paramsFileName)) {
//     //     std::cerr << "runMultiModelDriver:: params file: " << paramsFileName << " does not exist\n";
//     //     exit(801);
//     // }

//     std::string line = "MultiModel 1.001";
//     std::cout << "line: " << line << std::endl;
//     std::string lastWord;
//     size_t start = line.find_last_of(' ');
//     if (start != std::string::npos) lastWord = line.substr(start + 1);
//     float lw_float = std::stof(lastWord);
//     int lw_int = std::round(lw_float);
//     std::string modelIndex = std::to_string(lw_int);
//     std::cout << "modelIndex: " << modelIndex << std::endl;
//     std::cout << "Done! " << std::endl;
// }


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
      std::string lastWord;
      size_t start = line.find_last_of(' ');
      if (start != std::string::npos) lastWord = line.substr(start + 1);
      float value = std::stof(lastWord);
      int roundedValue = std::round(value);
      lastWord = std::to_string(roundedValue);

      if (!lastWord.empty()) {
        std::string multiModelDriverFileName = multiModelString + "_" + lastWord + "_" + driverFileName;
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
