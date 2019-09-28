simcenter_add_executable(NAME OpenSeesInput
                         FILES OpenSeesInput.c
                         DEPENDS CONAN_PKG::jansson)
