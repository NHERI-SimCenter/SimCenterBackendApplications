include ../../Makefile.inc

all: StandardHydroEDP 

StandardHydroEDP.o : StandardHydroEDP.cpp

StandardHydroEDP: StandardHydroEDP.o
	$(CXX) $(LDFLAGS) -o StandardHydroEDP StandardHydroEDP.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) StandardHydroEDP


