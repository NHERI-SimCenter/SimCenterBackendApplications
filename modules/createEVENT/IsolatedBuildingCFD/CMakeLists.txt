#Python scripts for pre and post processing
simcenter_add_python_script(SCRIPT foam_dict_reader.py)
simcenter_add_python_script(SCRIPT setup_case.py)
simcenter_add_python_script(SCRIPT process_output_data.py)

add_subdirectory(templateOF10Dicts)
