file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES
    IstanbulStations.py IstanbulRun.py IstanbulApp.py IstanbulApp2.py
    Istanbul_AllSites_grid.html All_Stations_Lat_Lon_Vs30_BedrockDepth.csv
    DESTINATION ${_rel})

add_subdirectory(TapisFiles)
