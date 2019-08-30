include ../../Makefile.inc

OBJS = callTPU.o 

all: LowRiseTPU

LowRiseTPU: $(OBJS) LowRiseTPU.o
	$(CXX) $(CPPFLAGS) -c ../../common/Units.cpp -o Units.o
	$(CXX) $(LDFLAGS) -o LowRiseTPU LowRiseTPU.o Units.o $(OBJS) $(LDLIBS) -lpython


test: testCallTPU.o $(OBJS)
	$(CXX) $(CPPFLAGS) -c ../../common/Units.cpp -o Units.o
	$(CXX) $(LDFLAGS) -o testCallTPU testCallTPU.o Units.o $(OBJS) $(LDLIBS) -lPython
	time ./testCallTPU
	time ./LowRiseTPU --filenameBIM testTPU.json --filenameEVENT event.json --getRV
	time ./LowRiseTPU --filenameBIM testTPU.json --filenameEVENT event.json

clean:
	$(RM) $(OBJS) *~ \#* *.o

distclean: clean 
	$(RM) LowRiseTPU testCallTPU
