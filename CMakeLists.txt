cmake_minimum_required(VERSION 3.15)
project(SimCenterBackendApplications C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)

# On macOS, Apple Clang ships without OpenMP. Homebrew's libomp provides it but
# is keg-only, so we must hint CMake before find_package(OpenMP) is called anywhere.
if(APPLE)
    find_program(_brew brew)
    if(_brew)
        execute_process(COMMAND ${_brew} --prefix libomp
                        OUTPUT_VARIABLE _libomp_prefix
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET)
        if(_libomp_prefix AND EXISTS "${_libomp_prefix}")
            # Flags for FindOpenMP — no -I here, handled separately below
            set(OpenMP_CXX_FLAGS     "-Xpreprocessor -fopenmp")
            set(OpenMP_C_FLAGS       "-Xpreprocessor -fopenmp")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_C_LIB_NAMES   "omp")
            set(OpenMP_omp_LIBRARY   "${_libomp_prefix}/lib/libomp.dylib")
            # Make omp.h findable for any target in the build
            include_directories(SYSTEM "${_libomp_prefix}/include")
            link_directories("${_libomp_prefix}/lib")
            message(STATUS "macOS: using Homebrew libomp at ${_libomp_prefix}")
        else()
            message(WARNING "macOS: libomp not found via Homebrew. Run: brew install libomp")
        endif()
    endif()
endif()


find_package(jansson       REQUIRED)
find_package(ZLIB          REQUIRED)
find_package(CURL          REQUIRED)
find_package(Eigen3        REQUIRED)
find_package(clara         REQUIRED)
find_package(nanoflann     REQUIRED)
find_package(NLopt         REQUIRED)
find_package(kissfft       REQUIRED)
find_package(Boost         REQUIRED)
find_package(nlohmann_json REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/modules)
add_subdirectory(modules)
