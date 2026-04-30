#!/bin/bash

# Usage: makeMac.sh [--arch <x86_64 or arm64>]
# Defaults: arch = ""

# NOTE if fails in nataf_gsa on Mac it is because clang still not working with omp,
# SOLN: you need a brew install of libomp, issue:
#           brew install libomp

# define the compilers
export CC=clang
export CXX=clang++
export CMAKE_ARGS=-DCMAKE_POLICY_VERSION_MINIMUM=3.5

CONAN_PROFILE="default"
BUILD_DIR="build"
OUTPUT_DIR="applications"
ARCH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch|-a)
            # 1. Capture the value
            ARCH="$2"
            
            # 2. VALIDATION: Check if it's one of the allowed types
            if [[ "$ARCH" != "x86_64" && "$ARCH" != "arm64" ]]; then
                echo "Error: Invalid architecture '$ARCH'."
                echo "Supported architectures: x86_64, arm64"
                exit 1
            fi

            if [[ "$ARCH" != "x86_64" ]]; then
		CMAKE_DAKOTA_VERSION=619
	    else
		CMAKE_DAKOTA_VERSION=620		
            fi	    

            # 3. If valid, set the variables
            CONAN_PROFILE="macos-$ARCH"
            BUILD_DIR="build_$ARCH"
            OUTPUT_DIR="${OUTPUT_DIR}_$ARCH"	    
            
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --arch <arm64|x86_64>"
            exit 1
            ;;
    esac
done

# Define a helper to print error and stop without closing the terminal
die() {
    echo "$1"
    exit 1
}

echo "Cleaning up old build directory..."
rm -rf "${BUILD_DIR}"

#
# run conan to install dependencies
#

conan install . --output-folder="${BUILD_DIR}" --build missing -s build_type=Release -pr "${CONAN_PROFILE}" || die "FAIL: Conan install failed."

#
# run cmake to generate CMakefiles, then build & finally install

CMAKE_ARCH_FLAG=""
if [ -n "${ARCH}" ]; then
    CMAKE_ARCH_FLAG="-DCMAKE_OSX_ARCHITECTURES=${ARCH}"
fi

cmake -B "${BUILD_DIR}" -S . \
    -DCMAKE_TOOLCHAIN_FILE="${BUILD_DIR}/conan_toolchain.cmake" ${CMAKE_ARCH_FLAG}  -DDAKOTA_VERSION=${CMAKE_DAKOTA_VERSION} \
    -DCMAKE_BUILD_TYPE=Release || die "FAIL: CMake configure failed."

cmake --build "${BUILD_DIR}" --parallel 8 || die "FAIL: CMake build failed."

cmake --install "${BUILD_DIR}" --prefix "${OUTPUT_DIR}" || die "FAIL: CMake install failed."

#
# Bundle libomp.dylib with nataf_gsa and run install_name_tool on natf_gsa so it runs without needing a brew install on user machine
#

NATAF_DIR="${PWD}/${OUTPUT_DIR}/performUQ/SimCenterUQ"
if [ "${ARCH}" = "x86_64" ]; then
    LIBOMP_DIR="/usr/local/opt/libomp/lib"
else
    LIBOMP_DIR="$(brew --prefix libomp)/lib"
fi
LIBOMP_SRC="${LIBOMP_DIR}/libomp.dylib"
echo "$PWD ${LIBOMP_SRC} ${NATAF_DIR}/libomp.dylib "

if [ -f "${NATAF_DIR}/nataf_gsa" ] && [ -f "${LIBOMP_SRC}" ]; then
    cp -f "${LIBOMP_SRC}" "${NATAF_DIR}"
    install_name_tool -change "${LIBOMP_SRC}" "@executable_path/libomp.dylib" "${NATAF_DIR}/nataf_gsa"
    echo "Bundled libomp.dylib into ${NATAF_DIR}"
else
    echo "WARNING: nataf_gsa or libomp.dylib not found — skipping libomp bundling"
fi




