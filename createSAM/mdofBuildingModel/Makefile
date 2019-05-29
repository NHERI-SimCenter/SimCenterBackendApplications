RM=rm -f
CPPFLAGS= -I /usr/local/jansson/include -I ./nanoflann/include -I ../../common -std=c++11
LDFLAGS= -static-libgcc
LDLIBS=/usr/lib/x86_64-linux-gnu/libjansson.a

SRCS=MDOF_BuildingModelToSAM.cpp
OBJS=MDOF_BuildingModelToSAM.o

all: MDOF

MDOF: $(OBJS)
	$(CXX) -c ../../common/Units.cpp $(CPPFLAGS) -o Units.o
	$(CXX) $(LDFLAGS) -o MDOF_BuildingModel MDOF_BuildingModelToSAM.o Units.o $(LDLIBS)

clean:
	$(RM) *~ \#* *.o *.out

distclean: clean
	$(RM) MDOF_BuildingModel
