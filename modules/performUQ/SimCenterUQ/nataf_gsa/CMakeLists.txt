cmake_minimum_required(VERSION 3.10)

ADD_DEFINITIONS("-DBOOST_ALL_NO_LIB")

# project(nataf_gsa_project)
# Because we need CXX_STANDARD 17 instead of 14, we don't use simcenter_add_executable() here.
if(UNIX)
  if(NOT APPLE)
    set(LINUX TRUE)
  endif()
endif()


if(LINUX)
  find_package(MPI)
  if (MPI_FOUND)
    set (PARALLEL_LIB "MPI_RUN")
  endif()
else()
  find_package(OpenMP)
  if (OpenMP_FOUND)
    set (PARALLEL_LIB "OPENMP_RUN")
  endif()
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCES main.cpp
	    ERADist.cpp
	    exponentialDist.cpp
	    normalDist.cpp
	    ERANataf.cpp
	    gammaDist.cpp
	    runGSA.cpp
	    runForward.cpp
	    RVDist.cpp
	    gumbelDist.cpp
	    truncExponentialDist.cpp
	    betaDist.cpp
	    jsonInput.cpp
	    uniformDist.cpp
	    chiSquaredDist.cpp
	    lognormalDist.cpp
	    weibullDist.cpp
	    discreteDist.cpp
	    writeErrors.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib_armadillo/armadillo-10.1.0/include)

if(MPI_FOUND)

  message(STATUS "nataf_gsa: MPI")
  include_directories(${MPI_INCLUDE_PATH})
  add_executable(nataf_gsa ${SOURCES})
  
  target_link_libraries(nataf_gsa
    CONAN_PKG::boost
    CONAN_PKG::jsonformoderncpp	
    CONAN_PKG::nlopt
    CONAN_PKG::eigen
    CONAN_PKG::mkl-static
    ${MPI_LIBRARIES}
    stdc++fs
    )
  add_compile_definitions(MPI_RUN)
  
elseif(OpenMP_FOUND)

  find_package(LAPACK REQUIRED)
  message(STATUS "nataf_gsa: OpenMP")	
  add_executable(nataf_gsa ${SOURCES})
  target_link_libraries(nataf_gsa
    CONAN_PKG::boost
    CONAN_PKG::jsonformoderncpp	
    CONAN_PKG::nlopt
    CONAN_PKG::eigen
    ${LAPACK_LIBRARIES}
    OpenMP::OpenMP_CXX)
    
  add_compile_definitions(OPENMP_RUN)
  
else()

  find_package(LAPACK REQUIRED)
  message(STATUS "nataf_gsa: no MPI or OpenMP")	  
  add_executable(nataf_gsa ${SOURCES})
  target_link_libraries(nataf_gsa
    CONAN_PKG::boost
    CONAN_PKG::jsonformoderncpp	
    CONAN_PKG::nlopt
    CONAN_PKG::eigen
    ${LAPACK_LIBRARIES})
    
endif() 

get_current_module_dir(moduleDir)
get_current_module_name(moduleName)    
install(TARGETS nataf_gsa DESTINATION "${PROJECT_SOURCE_DIR}/applications/performUQ/SimCenterUQ")
