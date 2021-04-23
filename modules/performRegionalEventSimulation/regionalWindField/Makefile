include ../../Makefile.inc

OBJS=WindFieldModel.o

all: WindFieldSimulation

WindFieldSimulation: $(OBJS)
	$(CXX) $(LDFLAGS) -o WindFieldSimulation $(OBJ) $(LDLIBS)

clean:
	$(RM) *~ \#* *.o *.out

distclean: clean
	$(RM) WindFieldSimulation