add_subdirectory(BaselineTclFiles)
add_subdirectory(Preprocessing)

file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES AllSectionDatabase.csv BeamDatabase.csv ColumnDatabase.csv DESTINATION ${_rel})
install(FILES
    main_program.py seismic_design.py building_information.py main_design.py
    column_component.py main_generation.py connection_part.py elastic_analysis.py
    model_generation.py elastic_output.py nonlinear_analysis.py global_variables.py
    beam_component.py help_functions.py steel_material.py
    DESTINATION ${_rel})
