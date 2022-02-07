simcenter_add_executable(NAME extractEDP
                         FILES extractEDP.cpp
                         DEPENDS CONAN_PKG::jansson)

simcenter_add_executable(NAME createStandardUQ_Input
                         FILES createStandardUQ_Input.cpp
                         DEPENDS CONAN_PKG::jansson)

add_library(commonUQ OBJECT
               parseWorkflowInput.cpp)

target_include_directories(commonUQ PUBLIC ${CONAN_INCLUDE_DIRS_JANSSON})
