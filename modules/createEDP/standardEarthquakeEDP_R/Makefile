include ../../Makefile.inc

all: StandardEarthquakeEDP

StandardEarthquakeEDP.o : StandardEarthquakeEDP.cpp

StandardEQEDP: StandardEarthquakeEDP.o
	$(CXX) $(LDFLAGS) -o StandardEarthquakeEDP StandardEarthquakeEDP.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) StandardEarthquakeEDP



