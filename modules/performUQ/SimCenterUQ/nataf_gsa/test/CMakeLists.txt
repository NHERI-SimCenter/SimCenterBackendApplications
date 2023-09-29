cmake_minimum_required(VERSION 3.10)

set(Sources
	nataf_gsa_test.cpp
)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

add_executable(nataf_gsa_test ${Sources})
target_link_libraries(nataf_gsa_test PUBLIC
					gtest_main
					nataf_gsa2
)

add_test(
	NAME nataf_gsa_test
	COMMAND nataf_gsa_test
)
#add_compile_definitions($<$<CONFIG:Debug>:_ITERATOR_DEBUG_LEVEL=2>)