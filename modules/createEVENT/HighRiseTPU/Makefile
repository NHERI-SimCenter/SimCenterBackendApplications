include ../../Makefile.inc

OBJS = callHighRiseTPU.o 

all: HighRiseTPU

HighRiseTPU: $(OBJS) HighRiseTPU.o
	echo $(CXX) $(CPPFLAGS) -c ../../common/Units.cpp -o Units.o
	$(CXX) $(CPPFLAGS) -c ../../common/Units.cpp -o Units.o
	$(CXX) $(LDFLAGS) -o HighRiseTPU HighRiseTPU.o Units.o $(OBJS) $(LDLIBS) -lpython


test: testCallTPU.o $(OBJS)
	$(CXX) $(CPPFLAGS) -c ../../common/Units.cpp -o Units.o
	$(CXX) $(LDFLAGS) -o testCallTPU testCallTPU.o Units.o $(OBJS) $(LDLIBS) -lPython
	time ./testCallTPU
	time ./HighRiseTPU --filenameAIM dakota.json --filenameEVENT event.json --getRV
	time ./HighRiseTPU --filenameAIM dakota.json --filenameEVENT event.json

clean:
	$(RM) $(OBJS) *~ \#* *.o

distclean: clean 
	$(RM) HighRiseTPU testCallTPU
