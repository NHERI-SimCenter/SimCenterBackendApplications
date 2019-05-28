RM=rm -f
CPPFLAGS= -I /usr/local/jansson/include -I ./nanoflann/include
LDFLAGS= -static-libgcc
LDLIBS=/usr/lib/x86_64-linux-gnu/libjansson.a


all: StandardGMT_EDP 

StandardGMT_EDP.o : StandardGMT_EDP.cpp

StandardGMT_EDP: StandardGMT_EDP.o
	$(CXX) $(LDFLAGS) -o StandardGMT_EDP StandardGMT_EDP.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean

	$(RM) StandardGMT_EDP


