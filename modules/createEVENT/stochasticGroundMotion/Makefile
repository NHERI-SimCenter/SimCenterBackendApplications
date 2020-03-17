include ../../Makefile.inc

SRC = command_parser.cpp \
	eq_generator.cpp \
	main.cpp

OBJS = command_parser.o eq_generator.o main.o

all: StochasticGM

StochasticGM: $(SRC)
	$(CXX) -c command_parser.cpp $(CPPFLAGS) $(STOCHASTIC_INC)
	$(CXX) -c eq_generator.cpp $(CPPFLAGS) $(STOCHASTIC_INC)
	$(CXX) -c main.cpp $(STOCHASTIC_INC)
	$(CXX) $(LDFLAGS) -o StochasticGM $(OBJS) $(LDLIBS) $(STOCHASTIC_LIBS)

clean:
	$(RM) $(OBJS) *~ \#* *.o

distclean: clean 
	$(RM) StochasticGM
