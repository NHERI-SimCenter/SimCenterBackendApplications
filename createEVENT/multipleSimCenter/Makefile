RM=rm -f
CPPFLAGS= -I /usr/local/jansson/include -I ./nanoflann/include
LDFLAGS= -static-libgcc
LDLIBS=/usr/local/jansson/lib/libjansson.a

all: MultipleSimCenterEvents 

MultipleSimCenterEvents.o : MultipleSimCenterEvents.cpp

MultipleSimCenterEvents: MultipleSimCenterEvents.o
	$(CXX) $(LDFLAGS) -o MultipleSimCenterEvents MultipleSimCenterEvents.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) MultipleSimCenterEvents


