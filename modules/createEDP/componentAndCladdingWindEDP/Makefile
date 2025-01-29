include ../../Makefile.inc

all:  ComponentAndCladdingEDP

ComponentAndCladdingEDP.o : ComponentAndCladdingEDP.cpp

ComponentAndCladdingEDP: ComponentAndCladdingEDP.o
	$(CXX) $(LDFLAGS) -o ComponentAndCladdingEDP ComponentAndCladdingEDP.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) ComponentAndCladdingEDP


