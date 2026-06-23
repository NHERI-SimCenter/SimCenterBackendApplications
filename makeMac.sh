
# NOTE if fail in nataf_gsa on Mac it is because clang still not working with omp, you need a brew install of libomp, issue:
#     brew install libomp

# define the compilers
export CC=clang
export CXX=clang++
export CMAKE_ARGS=-DCMAKE_POLICY_VERSION_MINIMUM=3.5


#
# following is temp until conan2 fully tested
#

cp conanfile.py conanfile.py.ORIG
cp CMakeLists.txt CMakeLists.txt.ORIG
cp cmake/SimCenterFunctions.cmake ./cmake/SimCenterFunctions.cmake.ORIG

cp conanfile2.py conanfile.py
cp CMakeLists2.txt CMakeLists.txt
cp cmake/SimCenterFunctions2.cmake ./cmake/SimCenterFunctions.cmake

#
# run conan to install dependencies
#

conan install . --output-folder=build --build=missing

#
# run cmake to generate CMakefiles, then build & finally install
#

cmake -B build -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix ./applications

#
# Bundle libomp.dylib with nataf_gsa so it runs without needing a brew install on user machine
#

NATAF_DIR="applications/performUQ/SimCenterUQ"
LIBOMP_DIR="$(brew --prefix libomp)/lib"
LIBOMP_SRC="${LIBOMP_DIR}/libomp.dylib"

if [ -f "${NATAF_DIR}/nataf_gsa" ] && [ -f "${LIBOMP_SRC}" ]; then
    cp "${LIBOMP_SRC}" "${NATAF_DIR}/libomp.dylib"
    install_name_tool -change "${LIBOMP_SRC}" "@executable_path/libomp.dylib" "${NATAF_DIR}/nataf_gsa"
    echo "Bundled libomp.dylib into ${NATAF_DIR}"
else
    echo "WARNING: nataf_gsa or libomp.dylib not found — skipping libomp bundling"
fi

#
# again temp until migration to conan2 complete
#

mv conanfile.py.ORIG conanfile.py
mv CMakeLists.txt.ORIG CMakeLists.txt
mv cmake/SimCenterFunctions.cmake.ORIG ./cmake/SimCenterFunctions.cmake



