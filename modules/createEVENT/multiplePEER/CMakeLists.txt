simcenter_add_executable(NAME MultiplePEER
                         FILES MultiplePEER_Events.cpp
                         DEPENDS CONAN_PKG::jansson)

set_property(TARGET MultiplePEER PROPERTY CXX_STANDARD 17)
