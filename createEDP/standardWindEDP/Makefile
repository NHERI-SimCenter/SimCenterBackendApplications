RM=rm -f
CPPFLAGS= -I /usr/local/jansson/include -I ./nanoflann/include
LDFLAGS= -static-libgcc
LDLIBS=/usr/local/jansson/lib/libjansson.a

all: StandardWindEDP 

StandardWindEDP.o : StandardWindEDP.cpp

StandardWindEDP: StandardWindEDP.o
	$(CXX) $(LDFLAGS) -o StandardWindEDP StandardWindEDP.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) StandardWindEDP


