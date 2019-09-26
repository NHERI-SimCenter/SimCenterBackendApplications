include ../../Makefile.inc

all: MultipleSimCenterEvents 

MultipleSimCenterEvents.o : MultipleSimCenterEvents.cpp

MultipleSimCenterEvents: MultipleSimCenterEvents.o
	$(CXX) $(LDFLAGS) -o MultipleSimCenterEvents MultipleSimCenterEvents.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) MultipleSimCenterEvents


