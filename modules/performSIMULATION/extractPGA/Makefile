include ../../Makefile.inc

all: ExtractPGA 

ExtractPGA.o : ExtractPGA.cpp

ExtractPGA: ExtractPGA.o
	$(CXX) $(LDFLAGS) -o ExtractPGA ExtractPGA.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) ExtractPGA


