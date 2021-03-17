include ../../Makefile.inc

OBJS=InterstoryParam.cpp HazusSAM_Generator.o FloorParam.o Building.o MDOF-LU.o

all: MDOF

MDOF: $(OBJS)
	$(CXX) -c ../../common/Units.cpp $(CPPFLAGS) -o Units.o
	$(CXX) $(LDFLAGS) -o MDOF-LU Units.o $(OBJS) $(LDLIBS)

clean:
	$(RM) *~ \#* *.o *.out

distclean: clean
	$(RM) MDOF-LU
