include ../../Makefile.inc

all: simCenterDprepro simCenterSub

debug: CPPFLAGS += -DDEBUG -g
debug: all

simCenterDprepro.o: simCenterDprepro.cpp

simCenterSub.o: simCenterSub.cpp

simCenterDprepro: simCenterDprepro.o
	$(CXX) $(LDFLAGS) -o simCenterDprepro simCenterDprepro.o $(LDLIBS)

simCenterSub: simCenterSub.o
	$(CXX) $(LDFLAGS) -o simCenterSub simCenterSub.o $(LDLIBS)


clean:
	$(RM) $(OBJS) *~ \#* extractEDP postprocessDAKOTA *.o preprocessDakota

distclean: clean
	$(RM) extractEDP postprocessDAKOTA workdir.* templatedir LHS*.out *.json dakota.*


