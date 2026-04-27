file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES SimCenterUQ.py surrogateBuild.py UQengine.py runPLoM.py DESTINATION ${_rel})

add_subdirectory(PLoM)
add_subdirectory(nataf_gsa)
