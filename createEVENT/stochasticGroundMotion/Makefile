include ../../Makefile.inc

SRC = src/command_parser.cc \
	src/eq_generator.cc \
	src/main.cc

OBJS = command_parser.o eq_generator.o main.o

all: StochasticGroundMotion

StochasticGroundMotion: $(SRC)
	$(CXX) $(LDFLAGS) -c src/command_parser.cc $(CPPFLAGS) $(STOCHASTIC_INC)
	$(CXX) $(LDFLAGS) -c src/eq_generator.cc $(CPPFLAGS) $(STOCHASTIC_INC)
	$(CXX) $(LDFLAGS) -c src/main.cc $(STOCHASTIC_INC)
	$(CXX) -o StochasticGroundMotion $(OBJS) $(LDLIBS) $(STOCHASTIC_LIBS)

clean:
	$(RM) $(OBJS) *~ \#* *.o

distclean: clean 
	$(RM) StochasticGroundMotion
