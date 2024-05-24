simcenter_add_executable(NAME PhysicsBasedMotion
                         FILES PhysicsBasedMotion.cpp
                         DEPENDS CONAN_PKG::jansson)

set_property(TARGET PhysicsBasedMotion PROPERTY CXX_STANDARD 17)
