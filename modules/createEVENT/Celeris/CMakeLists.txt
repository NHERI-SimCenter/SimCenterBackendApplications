file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES Celeris.py setrun.py setrun_old.py irrWaves.txt test_curve.xyz config.json DESTINATION ${_rel})

add_subdirectory(celeris)
