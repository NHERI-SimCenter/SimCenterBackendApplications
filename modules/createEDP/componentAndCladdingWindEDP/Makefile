include ../../Makefile.inc

all:  ComponentAndCladdingEDP

StandardWindEDP.o : ComponentAndCladdingEDP.cpp

StandardWindEDP: ComponentAndCladdingEDP.o
	$(CXX) $(LDFLAGS) -o ComponentAndCladdingEDP ComponentAndCladdingEDP.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) ComponentAndCladdingEDP


