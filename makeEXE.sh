#!/bin/bash

echo "Building SimCenterBackendApplications ..."
mkdir -p build
cd build
conan install .. --build missing
status=$?; if [[ $status != 0 ]]; then echo "conan failed"; exit $status; fi
cmake -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DCMAKE_CXX_EXTENSIONS=OFF .. -DCMAKE_LIBRARY_PATH="/usr/local/opt/libomp/lib" -DCMAKE_INCLUDE_PATH="/usr/local/opt/libomp/include"
status=$?; if [[ $status != 0 ]]; then echo "cmake failed"; exit $status; fi
cmake --build . --config Release
status=$?; if [[ $status != 0 ]]; then echo "make failed"; exit $status; fi
cmake --install .
status=$?; if [[ $status != 0 ]]; then echo "make install failed"; exit $status; fi
make install .
cd ..




