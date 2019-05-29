include ../../Makefile.inc

SRCS=OpenSeesInput.c 
OBJS=OpenSeesInput.o 

all: OpenSeesInput

OpenSeesInput: $(OBJS)
	$(CXX) $(LDFLAGS) -o OpenSeesInput  OpenSeesInput.o $(LDLIBS)

clean:
	$(RM) *~ \#* *.o *.out

distclean: clean
	$(RM) OpenSeesInput