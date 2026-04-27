add_executable(extractEDP extractEDP.cpp)
target_include_directories(extractEDP PRIVATE ${CMAKE_SOURCE_DIR}/modules)
target_link_libraries(extractEDP PRIVATE jansson::jansson)
file(RELATIVE_PATH _rel ${CMAKE_SOURCE_DIR}/modules ${CMAKE_CURRENT_SOURCE_DIR})
install(TARGETS extractEDP DESTINATION ${_rel})

add_executable(createStandardUQ_Input createStandardUQ_Input.cpp)
target_include_directories(createStandardUQ_Input PRIVATE ${CMAKE_SOURCE_DIR}/modules)
target_link_libraries(createStandardUQ_Input PRIVATE jansson::jansson)
install(TARGETS createStandardUQ_Input DESTINATION ${_rel})

add_library(commonUQ STATIC parseWorkflowInput.cpp)
target_include_directories(commonUQ PUBLIC
    ${CMAKE_SOURCE_DIR}/modules
    $<TARGET_PROPERTY:jansson::jansson,INTERFACE_INCLUDE_DIRECTORIES>)

install(FILES
    uq_utilities.py quoFEM_RV_models.py parallel_runner_mpi4py.py
    adaptive_doe.py common_datamodels.py convergence_metrics.py
    gp_ab_algorithm.py gp_model.py principal_component_analysis.py
    space_filling_doe.py tmcmc.py safer_cholesky.py
    kernel_density_estimation.py config_utilities.py logging_utilities.py
    log_likelihood_functions.py
    DESTINATION ${_rel})

add_subdirectory(ERAClasses)
