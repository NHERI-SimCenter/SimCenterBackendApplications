simcenter_add_executable(NAME MultipleSCEvents
                         FILES MultipleSimCenterEvents.cpp
                         DEPENDS CONAN_PKG::jansson)

set_property(TARGET MultipleSCEvents PROPERTY CXX_STANDARD 17)
