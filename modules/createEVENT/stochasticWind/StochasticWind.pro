TEMPLATE = app
CONFIG += console
CONFIG -= debug_and_release

OBJECTS_DIR = $${OUT_PWD}/obj

INCLUDEPATH += $$PWD/include \
               $$PWD/external/Clara \
               $$PWD/external \
               $$PWD/../../common\
               $$PWD/external/smelt/include

                     
SOURCES += $$PWD/src/command_parser.cc \
           $$PWD/src/wind_generator.cc \
           $$PWD/src/floor_forces.cc \           
           $$PWD/src/main.cc \
           $$PWD/../../common/Units.cpp           

#Assuming jansson library share the same parent folder with the app and is built with the same compiler
win32{
    INCLUDEPATH+="$$PWD/../../../jansson/build/include"
    LIBS += -L"$$PWD/../../../jansson/build/x64/Release" -ljansson
    LIBS += -L"$$PWD/../../../Stochastic-Loading-Module/build/Release" -lsmelt
    LIBS += -L"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/mkl/lib/intel64_win" -lmkl_core -lmkl_intel_lp64 -lmkl_tbb_thread
    LIBS += -L"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/tbb/lib/intel64/vc_mt"  -ltbb
    LIBS += -L"C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows/ipp/lib/intel64_win"  -lippcoremt -lippvmmt -lippsmt
}

unix{
    INCLUDEPATH += $$PWD/external/smelt/include
    LIBS += $$PWD/external/smelt/lib/libsmelt.so
}
unix:macx{
    CONFIG-=app_bundle
    INCLUDEPATH += $$PWD/external/smelt/include
    LIBS += $$PWD/external/smelt/lib/libsmelt.dylib
}
