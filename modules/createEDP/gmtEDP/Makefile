include ../../Makefile.inc

all: StandardGMT_EDP 

StandardGMT_EDP.o : StandardGMT_EDP.cpp

StandardGMT_EDP: StandardGMT_EDP.o
	$(CXX) $(LDFLAGS) -o StandardGMT_EDP StandardGMT_EDP.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean

	$(RM) StandardGMT_EDP


