include ../../Makefile.inc

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