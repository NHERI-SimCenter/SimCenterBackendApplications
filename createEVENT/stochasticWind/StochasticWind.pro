TEMPLATE = app
CONFIG += console
CONFIG += debug_and_release

DESTDIR  = $$PWD

INCLUDEPATH += $$PWD/include \
               $$PWD/external/Clara \
               $$PWD/external \
               $$PWD/../../common
                     
SOURCES += $$PWD/src/command_parser.cc \
           $$PWD/src/wind_generator.cc \
           $$PWD/src/floor_forces.cc \           
           $$PWD/src/main.cc \
           $$PWD/../../common/Units.cpp           

unix{
    INCLUDEPATH += $$PWD/external/smelt/include
    LIBS += $$PWD/external/smelt/lib/libsmelt.so
}
unix:macx{
    INCLUDEPATH += $$PWD/external/smelt/include
    LIBS += $$PWD/external/smelt/lib/libsmelt.dylib
}
