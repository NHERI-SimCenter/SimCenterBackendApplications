simcenter_add_executable(NAME extractEDP
                         FILES extractEDP.cpp
                         DEPENDS CONAN_PKG::jansson)

simcenter_add_executable(NAME createStandardUQ_Input
                         FILES createStandardUQ_Input.cpp
                         DEPENDS CONAN_PKG::jansson)

add_library(commonUQ STATIC
               parseWorkflowInput.cpp)

target_include_directories(commonUQ PUBLIC ${CONAN_INCLUDE_DIRS_JANSSON})

simcenter_add_python_script(SCRIPT uq_utilities.py)
simcenter_add_python_script(SCRIPT quoFEM_RV_models.py)
simcenter_add_python_script(SCRIPT parallel_runner_mpi4py.py)
add_subdirectory(ERAClasses)

simcenter_add_python_script(SCRIPT adaptive_doe.py)
simcenter_add_python_script(SCRIPT common_datamodels.py)
simcenter_add_python_script(SCRIPT convergence_metrics.py)
simcenter_add_python_script(SCRIPT gp_ab_algorithm.py)
simcenter_add_python_script(SCRIPT gp_model.py)
simcenter_add_python_script(SCRIPT principal_component_analysis.py)
simcenter_add_python_script(SCRIPT space_filling_doe.py)
simcenter_add_python_script(SCRIPT tmcmc.py)
simcenter_add_python_script(SCRIPT safer_cholesky.py)
simcenter_add_python_script(SCRIPT kernel_density_estimation.py)
simcenter_add_python_script(SCRIPT config_utilities.py)
simcenter_add_python_script(SCRIPT logging_utilities.py)
simcenter_add_python_script(SCRIPT log_likelihood_functions.py)