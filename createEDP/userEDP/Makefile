RM=rm -f
CPPFLAGS= -I /usr/local/jansson/include -I ./nanoflann/include
LDFLAGS= -static-libgcc
LDLIBS=/usr/local/jansson/lib/libjansson.a

all: UserDefinedEDP

UserDefinedEDP.o : UserDefinedEDP.cpp

UserDefinedEDP: UserDefinedEDP.o
	$(CXX) $(LDFLAGS) -o UserDefinedEDP UserDefinedEDP.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) UserDefinedEDP 


