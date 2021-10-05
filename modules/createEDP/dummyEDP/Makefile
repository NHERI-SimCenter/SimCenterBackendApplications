include ../../Makefile.inc

all: DummyEDP 

DummyEarthquakeEDP.o : DummyEDP.cpp

DummyEDP: DummyEDP.o
	$(CXX) $(LDFLAGS) -o DummyEDP DummyEDP.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) DummyEDP



