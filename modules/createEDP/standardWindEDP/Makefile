include ../../Makefile.inc

all: StandardWindEDP 

StandardWindEDP.o : StandardWindEDP.cpp

StandardWindEDP: StandardWindEDP.o
	$(CXX) $(LDFLAGS) -o StandardWindEDP StandardWindEDP.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) StandardWindEDP


