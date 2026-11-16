file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES
    M9Run.py M9_sites.csv M9Stations.py M9API.py M9App.py M9App2.py
    A_grid.html B_grid.html C_grid.html D_grid.html E_grid.html All_grid.html
    DESTINATION ${_rel})

add_subdirectory(TapisFiles)
