include ../../Makefile.inc

all: StandardTsunamiEDP 

StandardTsunamiEDP.o : StandardTsunamiEDP.cpp

StandardTsunamiEDP: StandardTsunamiEDP.o
	$(CXX) $(LDFLAGS) -o StandardTsunamiEDP StandardTsunamiEDP.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) StandardTsunamiEDP


