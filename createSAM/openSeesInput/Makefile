RM=rm -f
CFLAGS= -I /usr/local/jansson/include -I ./nanoflann/include -I ../../common
LDFLAGS= -static-libgcc
LDLIBS=/usr/local/jansson/lib/libjansson.a

SRCS=OpenSeesInput.c 
OBJS=OpenSeesInput.o 

all: OpenSeesInput

OpenSeesInput: $(OBJS)
	$(CXX) $(LDFLAGS) -o OpenSeesInput  OpenSeesInput.o $(LDLIBS)

clean:
	$(RM) *~ \#* *.o *.out

distclean: clean
	$(RM) OpenSeesInput