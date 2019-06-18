include ../../Makefile.inc

SRC = src/command_parser.cc \
	src/wind_generator.cc \
	src/floor_forces.cc \
	src/main.cc

OBJS = command_parser.o wind_generator.o floor_forces.o main.o Units.o

all: StochasticGroundMotion

StochasticGroundMotion: $(SRC)
	$(CXX) $(LDFLAGS) -c src/command_parser.cc $(CPPFLAGS) $(STOCHASTIC_INC)
	$(CXX) $(LDFLAGS) -c src/wind_generator.cc $(CPPFLAGS) $(STOCHASTIC_INC)
	$(CXX) $(LDFLAGS) -c src/floor_forces.cc $(CPPFLAGS) $(STOCHASTIC_INC)
	$(CXX) $(LDFLAGS) -c ../../common/Units.cpp $(CPPFLAGS) $(STOCHASTIC_INC)
	$(CXX) $(LDFLAGS) -c src/main.cc $(STOCHASTIC_INC)
	$(CXX) -o StochasticWind $(OBJS) $(LDLIBS) $(STOCHASTIC_LIBS)

clean:
	$(RM) $(OBJS) *~ \#* *.o

distclean: clean 
	$(RM) StochasticGroundMotion
