RM=rm -f
CPPFLAGS= -I /usr/local/jansson/include -I ./nanoflann/include
LDFLAGS= -static-libgcc
LDLIBS=/usr/local/jansson/lib/libjansson.a

all: StandardEarthquakeEDP 

StandardEarthquakeEDP.o : StandardEarthquakeEDP.cpp

StandardEarthquakeEDP: StandardEarthquakeEDP.o
	$(CXX) $(LDFLAGS) -o StandardEarthquakeEDP StandardEarthquakeEDP.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) StandardEarthquakeEDP


