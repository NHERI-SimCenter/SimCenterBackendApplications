# static library (the "common" module)
file(GLOB _srcs *.cpp *.cc *.c)
if(_srcs)
    add_library(common STATIC ${_srcs})
    target_include_directories(common PUBLIC ${CMAKE_SOURCE_DIR}/modules)
endif()

file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES simcenter_common.py DESTINATION ${_rel})

add_subdirectory(groundMotionIM)
