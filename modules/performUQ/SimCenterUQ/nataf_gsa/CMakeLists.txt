cmake_minimum_required(VERSION 3.10)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	#
	# Only for debuging... I need mkl
	#
	project(nataf_gsa_project)
	enable_language(Fortran)

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


	# sy - optimization flags
	#if(MSVC)
	#	add_compile_options("/W4" "$<$<CONFIG:RELEASE>:/O2>")
	#endif()

	# sy - to check if filesystem is working

	if(LINUX)
		try_compile(HAS_FS "${CMAKE_BINARY_DIR}/temp" 
		"${CMAKE_SOURCE_DIR}/tmp/has_filesystem.cc" 
					CMAKE_FLAGS -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON
					LINK_LIBRARIES stdc++fs)
		if(HAS_FS)
			message(STATUS "Compiler has filesystem support")
		else()
			message(FATAL_ERROR "Compiler is missing filesystem capabilities")
		endif(HAS_FS)
	endif()


	include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
	conan_basic_setup(TARGETS)
	
	add_subdirectory("${PROJECT_SOURCE_DIR}/ARPACK")
	set (ARPACK_LIBRARIES ARPACK)

	# for visual studio, do on commandline: conan install . -s build_type=Debug -if=C:/Users/SimCenter/Sangri/nataf_gsa_cpp_mpi/out/build/x64-Debug/

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

	if(PARALLEL_LIB STREQUAL  "MPI_RUN")
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
       		${ARPACK_LIBRARIES}
			)
		add_compile_definitions(MPI_RUN)

	else()
		add_executable(nataf_gsa ${SOURCES})
		target_link_libraries(nataf_gsa
			CONAN_PKG::boost
			CONAN_PKG::jsonformoderncpp	
			CONAN_PKG::nlopt
			CONAN_PKG::eigen
			CONAN_PKG::mkl-static
			OpenMP::OpenMP_CXX
       		${ARPACK_LIBRARIES}
			)
		add_compile_definitions(OPENMP_RUN)
	endif() 

	install(TARGETS nataf_gsa DESTINATION ../..)
	#add_compile_definitions($<$<CONFIG:Debug>:_ITERATOR_DEBUG_LEVEL=0>)

else()

	ADD_DEFINITIONS("-DBOOST_ALL_NO_LIB")

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

	# sy - optimization flags
	# if(MSVC)
	#    add_compile_options("/W4" "$<$<CONFIG:RELEASE>:/O2>")
	# endif()

	# sy - to check if filesystem is working

	#if(LINUX)
	#	try_compile(HAS_FS "${CMAKE_BINARY_DIR}/temp" 
	#	"${CMAKE_SOURCE_DIR}/tmp/has_filesystem.cc" 
	#				CMAKE_FLAGS -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON
	#				LINK_LIBRARIES stdc++fs)
	#	if(HAS_FS)
	#		message(STATUS "Compiler has filesystem support")
	#	else()
	#		message(FATAL_ERROR "Compiler is missing filesystem capabilities")
	#	endif(HAS_FS)
	#endif()

	# for visual studio IDE, do on commandline: conan install . -s build_type=Debug -if=C:/Users/SimCenter/Sangri/SimCenterBackendApplications/modules/performUQ/SimCenterUQ/nataf_gsa/out/build/x64-Debug/

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

	  #find_package(LAPACK REQUIRED)
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


	install(TARGETS nataf_gsa DESTINATION ${PROJECT_SOURCE_DIR}/applications/performUQ/SimCenterUQ)
	
endif()
