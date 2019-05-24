RM=rm -f
CPPFLAGS= -I /usr/local/jansson/include -I ./nanoflann/include -std=c++11
LDFLAGS= -static-libgcc
LDLIBS=/usr/local/jansson/lib/libjansson.a

STOCHASTIC_INC = -I ./include \
	-I ../../external \
	-I ../../external/nlohmann_json \
	-I ../../external/stochastic_loading/include $(CPPFLAGS)

STOCHASTIC_LIBS = /opt/intel/mkl/lib/libmkl_intel_lp64.a /opt/intel/mkl/lib/libmkl_intel_thread.a /opt/intel/mkl/lib/libmkl_core.a /opt/intel/compilers_and_libraries/mac/lib/libiomp5.a /opt/intel/ipp/lib/libippcore.a /opt/intel/ipp/lib/libippvm.a /opt/intel/ipp/lib/libipps.a ../../../smelt/build/liblstoc.a

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
