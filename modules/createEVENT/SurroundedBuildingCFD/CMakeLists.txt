file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES foam_file_processor.py setup_case.py post_process_output.py SurroundedBuildingCFD.py DESTINATION ${_rel})

add_subdirectory(templateOF10Dicts)
