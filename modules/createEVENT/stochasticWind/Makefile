include ../../Makefile.inc

SRC = command_parser.cpp \
	wind_generator.cpp \
	floor_forces.cpp \
	main.cpp

OBJS = command_parser.o wind_generator.o floor_forces.o main.o Units.o

all: StochasticGroundMotion

StochasticGroundMotion: $(SRC)
	$(CXX) -c command_parser.cpp $(CPPFLAGS) $(STOCHASTIC_INC)
	$(CXX) -c wind_generator.cpp $(CPPFLAGS) $(STOCHASTIC_INC)
	$(CXX) -c floor_forces.cpp $(CPPFLAGS) $(STOCHASTIC_INC)
	$(CXX) -c ../../common/Units.cpp $(CPPFLAGS) $(STOCHASTIC_INC)
	$(CXX) -c main.cpp $(STOCHASTIC_INC)
	$(CXX) $(LDFLAGS) -o StochasticWind $(OBJS) $(LDLIBS) $(STOCHASTIC_LIBS)

clean:
	$(RM) $(OBJS) *~ \#* *.o

distclean: clean 
	$(RM) StochasticGroundMotion
