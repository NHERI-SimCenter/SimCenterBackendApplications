include ../../Makefile.inc

all: StandardEQEDP 

StandardEarthquakeEDP.o : StandardEarthquakeEDP.cpp

StandardEQEDP: StandardEarthquakeEDP.o
	$(CXX) $(LDFLAGS) -o StandardEQEDP StandardEarthquakeEDP.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) StandardEQEDP


