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