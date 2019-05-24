RM=rm -f
CPPFLAGS= -I /usr/local/jansson/include -I ./nanoflann/include -I ../../common -std=c++11
LDFLAGS= -static-libgcc
LDLIBS=/usr/local/jansson/lib/libjansson.a

OBJS = WindForces.o \
	windsim_dk1_main.o \
	windsim_dk1_main_emxAPI.o \
	windsim_dk1_main_emxutil.o \
	rtGetInf.o \
	rtGetNaN.o \
	rt_nonfinite.o

all: NOWS1


NOWS1: NOWS1.o $(OBJS)
	$(CXX) $(CPPFLAGS) -c ../../common/Units.cpp -o Units.o
	$(CXX) $(LDFLAGS) -o NOWS1 NOWS1.o Units.o $(OBJS) $(LDLIBS)


test: NOWS1
	./NOWS1 --filenameBIM testInput.json --filenameEVENT evt.json --getRV
	./NOWS1 --filenameBIM testInput.json --filenameEVENT evt.json 
	./NOWS1 --filenameBIM blah.json --filenameEVENT evt.json

debug:

clean:
	$(RM) $(OBJS) *~ \#* *.o

distclean: clean 
	$(RM) NOWS1
