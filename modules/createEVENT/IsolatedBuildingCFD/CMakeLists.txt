file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES
    foam_dict_reader.py setup_case.py post_process_output.py IsolatedBuildingCFD.py
    generate_component_geometry.py generate_wind_profiles.py
    DESTINATION ${_rel})

add_subdirectory(templateOF10Dicts)
