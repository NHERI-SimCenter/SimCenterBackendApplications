include ../../Makefile.inc

OBJS = 

all: WindTunnelExperiment

WindTunnelExperiment: $(OBJS) WindTunnelExperiment.o
	$(CXX) $(CPPFLAGS) -c ../../common/Units.cpp -o Units.o
	$(CXX) $(LDFLAGS) -o WindTunnelExperiment WindTunnelExperiment.o Units.o $(OBJS) $(LDLIBS)


test: WindTunnelExperiment
	time ./WindTunnelExperiment --filenameBIM testInput.json --filenameEVENT event.json --getRV
	time ./WindTunnelExperiment --filenameBIM testInput.json --filenameEVENT event.json

clean:
	$(RM) $(OBJS) *~ \#* *.o

distclean: clean 
	$(RM) WindTunnelExperiment event.json
