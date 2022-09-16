include ../../Makefile.inc

OBJS = callNonIsolatedLowRiseTPU.o 

all: NonIsolatedLowRiseTPU

NonIsolatedLowRiseTPU: $(OBJS) NonIsolatedLowRiseTPU.o
	$(CXX) $(CPPFLAGS) -c ../../common/Units.cpp -o Units.o
	$(CXX) $(LDFLAGS) -o NonIsolatedLowRiseTPU NonIsolatedLowRiseTPU.o Units.o $(OBJS) $(LDLIBS) -lpython


test: NonIsolatedLowRiseTPU
	time ./NonIsolatedLowRiseTPU --filenameAIM dakota.json --filenameEVENT event.json --getRV
	time ./NonIsolatedLowRiseTPU --filenameAIM dakota.json --filenameEVENT event.json

test1: testCallTPU.o $(OBJS)
	$(CXX) $(CPPFLAGS) -c ../../common/Units.cpp -o Units.o
	$(CXX) $(LDFLAGS) -o testCallTPU testCallTPU.o Units.o $(OBJS) $(LDLIBS) -lPython
	time ./testCallTPU
	time ./NonIsolatedLowRiseTPU --filenameAIM dakota.json --filenameEVENT event.json --getRV
	time ./NonIsolatedLowRiseTPU --filenameAIM dakota.json --filenameEVENT event.json

clean:
	$(RM) $(OBJS) *~ \#* *.o 

distclean: clean 
	$(RM) NonIsolatedLowRiseTPU testCallTPU
