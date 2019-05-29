include ../../Makefile.inc

SRCS=extractEDP.cpp postprocessDakota.cpp
OBJS=extractEDP.o  postprocessDakota.o

all: extractEDP

debug: CPPFLAGS += -DDEBUG -g
debug: all

extractEDP.o: extractEDP.cpp 

postprocessDakota.o: postprocessDakota.cpp 

postprocessDakota: $(OBJS)
	$(CXX) $(LDFLAGS) -o postprocessDakota postprocessDakota.o $(LDLIBS)

extractEDP: $(OBJS)
	$(CXX) $(LDFLAGS) -o extractEDP extractEDP.o $(LDLIBS)

clean:
	$(RM) $(OBJS) *~ \#* extractEDP postprocessDAKOTA *.o

distclean: clean
	$(RM) extractEDP postprocessDAKOTA workdir.* templatedir LHS*.out *.json dakota.*


