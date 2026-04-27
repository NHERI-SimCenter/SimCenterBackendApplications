file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(FILES quofemDTOs.py UQpyDTO.py DESTINATION ${_rel})

add_subdirectory(distributions)
add_subdirectory(modules)
add_subdirectory(reliability)
add_subdirectory(runmodel)
add_subdirectory(sampling)
