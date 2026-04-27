add_subdirectory(database)
add_subdirectory(gmpe)
add_subdirectory(lib)

file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES
    ComputeIntensityMeasure.py CreateScenario.py CreateStation.py
    FetchOpenQuake.py FetchOpenSHA_25_4.py HazardSimulation.py HazardSimulationEQ.py
    SelectGroundMotion.py HazardOccurrence.py USGS_API.py ScenarioForecast.py
    liquefaction.py landslide.py GMSimulators.py GlobalVariable.py
    DESTINATION ${_rel})
