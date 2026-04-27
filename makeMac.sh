
# NOTE if fails in nataf_gsa on Mac it is because clang still not working with omp,
# SOLN: you need a brew install of libomp, issue:
#           brew install libomp

# define the compilers
export CC=clang
export CXX=clang++
export CMAKE_ARGS=-DCMAKE_POLICY_VERSION_MINIMUM=3.5

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
# Bundle libomp.dylib with nataf_gsa and run install_name_tool on natf_gsa so it runs without needing a brew install on user machine
#

NATAF_DIR="${PWD}/applications/performUQ/SimCenterUQ"
LIBOMP_DIR="$(brew --prefix libomp)/lib"
LIBOMP_SRC="${LIBOMP_DIR}/libomp.dylib"
echo "$PWD ${LIBOMP_SRC} ${NATAF_DIR}/libomp.dylib "


if [ -f "${NATAF_DIR}/nataf_gsa" ] && [ -f "${LIBOMP_SRC}" ]; then
    cp "${LIBOMP_SRC}" "${NATAF_DIR}"
    install_name_tool -change "${LIBOMP_SRC}" "@executable_path/libomp.dylib" "${NATAF_DIR}/nataf_gsa"
    echo "Bundled libomp.dylib into ${NATAF_DIR}"
else
    echo "WARNING: nataf_gsa or libomp.dylib not found — skipping libomp bundling"
fi




