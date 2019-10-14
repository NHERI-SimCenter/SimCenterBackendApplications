include ../../Makefile.inc

all: MultiplePEER

MultiplePEER_Events.o : MultiplePEER_Events.cpp

MultiplePEER: MultiplePEER_Events.o
	$(CXX) $(LDFLAGS) -o MultiplePEER MultiplePEER_Events.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) MultiplePEER_Events


