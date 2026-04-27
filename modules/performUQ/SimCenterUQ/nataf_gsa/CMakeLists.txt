cmake_minimum_required(VERSION 3.10)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	#
	# Only for debugging... I need mkl
	#

	project(nataf_gsa_project)
	enable_language(Fortran)
	set(gtest_force_shared_crt on)
	enable_testing()

	if(UNIX)
		if(NOT APPLE)
			set(LINUX TRUE)
		endif()
	endif()


	if(LINUX)
		set ( PARALLEL_LIB "MPI_RUN")
	else()
		set ( PARALLEL_LIB "OPENMP_RUN")
	endif()


	# sy- # Find mp package

	if(PARALLEL_LIB STREQUAL  "MPI_RUN")
		find_package(MPI REQUIRED)
	else()
		find_package(OpenMP REQUIRED) # Find the package
	endif() 

	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED True)

	include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)
	find_package(Boost         REQUIRED)
	find_package(nlohmann_json REQUIRED)
	find_package(NLopt         REQUIRED)
	find_package(Eigen3        REQUIRED)

	add_subdirectory("${PROJECT_SOURCE_DIR}/ARPACK")
	set (ARPACK_LIBRARIES ARPACK)

	set(SOURCES main.cpp
			ERADist.cpp
			exponentialDist.cpp
			normalDist.cpp
			ERANataf.cpp
			gammaDist.cpp
			runGSA.cpp
			runForward.cpp
			runMFMC.cpp
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

	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
	if(PARALLEL_LIB STREQUAL  "MPI_RUN")
		include_directories(${MPI_INCLUDE_PATH})
		add_executable(nataf_gsa ${SOURCES})
		target_link_libraries(nataf_gsa
			Boost::boost
			nlohmann_json::nlohmann_json
			NLopt::nlopt
			Eigen3::Eigen
			${MPI_LIBRARIES}
			stdc++fs
      		${ARPACK_LIBRARIES}
			)
		add_compile_definitions(MPI_RUN)

	else()
		add_executable(nataf_gsa ${SOURCES})
		target_link_libraries(nataf_gsa
			Boost::boost
			nlohmann_json::nlohmann_json
			NLopt::nlopt
			Eigen3::Eigen
			OpenMP::OpenMP_CXX
      		${ARPACK_LIBRARIES}
			)
		add_compile_definitions(OPENMP_RUN)
	endif() 

	install(TARGETS nataf_gsa DESTINATION ../..)

	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
	add_library(nataf_gsa2 STATIC ${SOURCES} ${HEADERS})
	target_link_libraries(nataf_gsa2
		Boost::boost
		nlohmann_json::nlohmann_json
		NLopt::nlopt
		Eigen3::Eigen
		OpenMP::OpenMP_CXX
    	${ARPACK_LIBRARIES}
			)
	add_compile_definitions(OPENMP_RUN)
	
	set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL$<$<CONFIG:Debug>:Debug>")
	add_compile_definitions($<$<CONFIG:Debug>:_ITERATOR_DEBUG_LEVEL=2>)

else()

	ADD_DEFINITIONS("-DBOOST_ALL_NO_LIB")

	# Uses plain CMake — CXX_STANDARD 17 is set per-target below.
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
			runMFMC.cpp
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
		Boost::boost
		nlohmann_json::nlohmann_json
		NLopt::nlopt
		Eigen3::Eigen
		${MPI_LIBRARIES}
		stdc++fs
		)
	  add_compile_definitions(MPI_RUN)
  
	elseif(OpenMP_FOUND)

	  find_package(LAPACK REQUIRED)
	  message(STATUS "nataf_gsa: OpenMP")	
	  add_executable(nataf_gsa ${SOURCES})
	  target_link_libraries(nataf_gsa
		Boost::boost
		nlohmann_json::nlohmann_json
		NLopt::nlopt
		Eigen3::Eigen
		${LAPACK_LIBRARIES}
		OpenMP::OpenMP_CXX)
    
	  add_compile_definitions(OPENMP_RUN)
  
	else()

	  find_package(LAPACK REQUIRED)
	  message(STATUS "nataf_gsa: no MPI or OpenMP")	  
	  add_executable(nataf_gsa ${SOURCES})
	  target_link_libraries(nataf_gsa
		Boost::boost
		nlohmann_json::nlohmann_json
		NLopt::nlopt
		Eigen3::Eigen
		${LAPACK_LIBRARIES})
    
	endif() 


	file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
	get_filename_component(_install_dir ${_rel} DIRECTORY)
	install(TARGETS nataf_gsa DESTINATION ${_install_dir})

endif()
