include ../../Makefile.inc

all: StandardEDP 

StandardEDP.o : StandardEDP.cpp

StandardEDP: StandardEDP.o
	$(CXX) $(LDFLAGS) -o StandardEDP StandardEDP.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) StandardEDP


