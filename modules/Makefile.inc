RM=rm -f
CPPFLAGS= -I /usr/local/jansson/include -I ./nanoflann/include -std=c++11 -I ./../../common
CXXFLAGS= -I /usr/local/jansson/include -I ./nanoflann/include
LDFLAGS= -static-libgcc
LDLIBS=/usr/local/jansson/lib/libjansson.a -lcurl

STOCHASTIC_INC = -I ./include \
	-I ../../external \
	-I ../../external/nlohmann_json \
	-I ../../external/stochastic_loading/include $(CPPFLAGS)

STOCHASTIC_LIBS = /opt/intel/mkl/lib/libmkl_intel_lp64.a /opt/intel/mkl/lib/libmkl_intel_thread.a /opt/intel/mkl/lib/libmkl_core.a /opt/intel/compilers_and_libraries/mac/lib/libiomp5.a /opt/intel/ipp/lib/libippcore.a /opt/intel/ipp/lib/libippvm.a /opt/intel/ipp/lib/libipps.a ../../../smelt/build/liblstoc.a


