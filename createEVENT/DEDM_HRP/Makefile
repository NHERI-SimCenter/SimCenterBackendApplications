RM=rm -f
CPPFLAGS= -I /usr/local/jansson/include -I ./nanoflann/include -I ../../common -std=c++11
LDFLAGS= -static-libgcc
LDLIBS=/usr/local/jansson/lib/libjansson.a -lcurl

OBJS = dedmCall.o

all: DEDM_HRP

DEDM_HRP: DEDM_HRP.o $(OBJS)
	$(CXX) $(CPPFLAGS) -c ../../common/Units.cpp -o Units.o
	$(CXX) $(LDFLAGS) -o DEDM_HRP DEDM_HRP.o Units.o $(OBJS) $(LDLIBS)


test: DEDM_HRP testDedmCall.o
	$(CXX) $(LDFLAGS) -o testDedmCall testDedmCall.o Units.o $(OBJS) $(LDLIBS)
	./testDedmCall
	python ./DEDM_HRP.py --filenameBIM testInput.json --filenameEVENT evt.json --getRV
	python ./DEDM_HRP.py --filenameBIM testInput.json --filenameEVENT evt.json 

clean:
	$(RM) $(OBJS) *~ \#* *.o

distclean: clean 
	$(RM) DEDM_HRP testDedmCall
