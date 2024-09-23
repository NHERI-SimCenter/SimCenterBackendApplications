include ../../Makefile.inc

all: StandardStormSurgeEDP 

StandardStormSurgeEDP.o : StandardStormSurgeEDP.cpp

StandardStormSurgeEDP: StandardStormSurgeEDP.o
	$(CXX) $(LDFLAGS) -o StandardStormSurgeEDP StandardStormSurgeEDP.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) StandardStormSurgeEDP


