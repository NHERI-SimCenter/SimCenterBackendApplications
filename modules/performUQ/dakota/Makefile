include ../../Makefile.inc

SRCS=extractEDP.cpp postprocessDakota.cpp preprocessDakota.cpp dakotaProcedures.cpp
OBJS=extractEDP.o  postprocessDakota.o simCenterDprepro.o preprocessDakota.o dakotaProcedures.o

all: extractEDP simCenterDprepro preprocessDakota

debug: CPPFLAGS += -DDEBUG -g
debug: all

extractEDP.o: extractEDP.cpp 

postprocessDakota.o: postprocessDakota.cpp 

preprocessDakota.o: preprocessDakota.cpp 

dakotaProcedures.o: dakotaProcedures.cpp 

postprocessDakota: $(OBJS)
	$(CXX) $(LDFLAGS) -o postprocessDakota postprocessDakota.o $(LDLIBS)

preprocessDakota: $(OBJS)
	$(CXX) $(LDFLAGS) -o preprocessDakota preprocessDakota.o dakotaProcedures.o $(LDLIBS)

simCenterDprepro: $(OBJS)
	$(CXX) $(LDFLAGS) -o simCenterDprepro simCenterDprepro.o $(LDLIBS)

extractEDP: $(OBJS)
	$(CXX) $(LDFLAGS) -o extractEDP extractEDP.o $(LDLIBS)

clean:
	$(RM) $(OBJS) *~ \#* extractEDP postprocessDAKOTA *.o preprocessDakota

distclean: clean
	$(RM) extractEDP postprocessDAKOTA workdir.* templatedir LHS*.out *.json dakota.*


