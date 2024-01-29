#Python scripts for pre and post-processing
simcenter_add_python_script(SCRIPT foam_file_processor.py)
simcenter_add_python_script(SCRIPT setup_case.py)
simcenter_add_python_script(SCRIPT post_process_output.py)
simcenter_add_python_script(SCRIPT EmptyDomainCFD.py)

add_subdirectory(templateOF10Dicts)
