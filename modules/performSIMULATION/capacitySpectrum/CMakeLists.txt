file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES
    CapacityModels.py DampingModels.py DemandModels.py runCMS.py runCMS2.py
    hazus_capacity_alpha2.csv hazus_kappa_data.csv hazus_typical_roof_height.csv
    HC_capacity_data.csv MC_capacity_data.csv LC_capacity_data.csv
    PC_capacity_data.csv SC_capacity_data.csv VC_capacity_data.csv
    DESTINATION ${_rel})
