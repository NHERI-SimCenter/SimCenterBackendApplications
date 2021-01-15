include ../../Makefile.inc

SRCS=OpenSeesPreprocessor.cpp OpenSeesPostprocessor.cpp

OBJS_PRE=OpenSeesPreprocessor.o mainPreprocessor.o OpenSeesConcreteShearWalls.o

OBJS_UNC=OpenSeesPreprocessor.o getUncertainty.o OpenSeesConcreteShearWalls.o

OBJS_POST=OpenSeesPostprocessor.o mainPostprocessor.o OpenSeesConcreteShearWalls.o

all: mainPreprocessor mainPostprocessor getUncertainty

mainPreprocessor: $(OBJS_PRE)
	$(CXX) -c ../../common/Units.cpp $(CPPFLAGS) -o Units.o
	$(CXX) $(LDFLAGS) -o mainPreprocessor $(OBJS_PRE) Units.o $(LDLIBS)


getUncertainty: $(OBJS_UNC)
	$(CXX) -c ../../common/Units.cpp $(CPPFLAGS) -o Units.o
	$(CXX) $(LDFLAGS) -o getUncertainty $(OBJS_UNC) Units.o $(LDLIBS)


mainPostprocessor: $(OBJS_POST)
	$(CXX) $(LDFLAGS) -o mainPostprocessor $(OBJS_POST) $(LDLIBS)

clean:
	$(RM) *~ \#* *.o *.out

distclean: clean
	$(RM) mainPostprocessor mainPreprocessor getUncertainty


