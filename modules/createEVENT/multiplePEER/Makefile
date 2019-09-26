include ../../Makefile.inc

all: MultiplePEER_Events 

MultiplePEER_Events.o : MultiplePEER_Events.cpp

MultiplePEER_Events: MultiplePEER_Events.o
	$(CXX) $(LDFLAGS) -o MultiplePEER_Events MultiplePEER_Events.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) MultiplePEER_Events


