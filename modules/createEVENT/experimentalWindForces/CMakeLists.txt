file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES experimentalWindForces.py convertWindMat.py DESTINATION ${_rel})
