cmake_minimum_required(VERSION 3.15)

file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES NNGM.py DESTINATION ${_rel})
