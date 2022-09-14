include ../../Makefile.inc

OBJS = WindSpeedFromCloudData.o 

all: WindSpeedFromCloudData

WindSpeedFromCloudData: $(OBJS) 
	$(CXX) $(LDFLAGS) -o WindSpeedFromCloudData $(OBJS) $(LDLIBS) -lpython

test: WindSpeedFromCloudData $(OBJS)
	time python ./parseHurricaneScenario.py scenario.mat scenario.json
	time ./WindSpeedFromCloudData -filenameAIM 2-BIM.json -filenameEVENT 1-EVENT.json -filenameWindSpeedCloudData scenario.json -getRV
	time ./WindSpeedFromCloudData -filenameAIM 2-BIM.json -filenameEVENT 1-EVENT.json -filenameWindSpeedCloudData scenario.json 

clean:
	$(RM) $(OBJS) *~ \#* *.o

distclean: clean 
	$(RM) WindSpeedFromCloud scenario.json
