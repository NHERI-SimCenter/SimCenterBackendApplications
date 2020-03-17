include ../../Makefile.inc

all: UserDefinedEDP

UserDefinedEDP.o : UserDefinedEDP.cpp

UserDefinedEDP: UserDefinedEDP.o
	$(CXX) $(LDFLAGS) -o UserDefinedEDP UserDefinedEDP.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) UserDefinedEDP 


