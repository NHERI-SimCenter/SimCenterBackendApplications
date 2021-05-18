include ../../Makefile.inc

all: StandardEDP 

StandardWindEDP.o : StandardEDP.cpp

StandardWindEDP: StandardWindEDP.o
	$(CXX) $(LDFLAGS) -o StandardEDP StandardEDP.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) StandardEDP


