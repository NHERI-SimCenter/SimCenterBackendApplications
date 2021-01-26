include ../../Makefile.inc

all: MultipleSCEvents 

MultipleSimCenterEvents.o : MultipleSimCenterEvents.cpp

MultipleSCEvents: MultipleSimCenterEvents.o
	$(CXX) $(LDFLAGS) -o MultipleSCEvents MultipleSimCenterEvents.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) MultipleSCEvents


