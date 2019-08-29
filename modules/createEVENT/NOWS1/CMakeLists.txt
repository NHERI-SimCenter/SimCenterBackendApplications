simcenter_add_executable(NAME NOWS1
                         FILES NOWS1.cpp rtGetInf.cpp rtGetNaN.cpp rt_nonfinite.cpp WindSim.cpp
                               windsim_dk1_main.cpp windsim_dk1_main_emxAPI.cpp windsim_dk1_main_emxutil.cpp
                         DEPENDS CONAN_PKG::jansson common)
