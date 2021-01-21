RM=rm -f
CC=icc
CXX=icc
RM=rm -f
CPPFLAGS= -I$(HOME)/jansson-2.11/include -I./nanoflann/include -I ./../../common -I ./../../common/Clara -I ./../.. -I /home1/00477/tg457427/SimCenterBackendApplications/modules/external/Clara/include -std=c++11 
LDFLAGS= 
LDLIBS=$(HOME)/jansson-2.11/build/lib/libjansson.a -lcurl
LINKER           = $(CXX)

MKL_LIB = $(MKLROOT)/lib/intel64
IPP_LIB = $(IPPROOT)/lib/intel64

STOCHASTIC_INC = -I ./include \
	-I ../../.. \
	-I ../.. \
	-I ../../external \
	-I ../../external/nlohmann_json \
	-I /home1/00477/tg457427/smelt/include \
	-I /home1/00477/tg457427/smelt/external \
	-I ../../external/stochastic_loading/include $(CPPFLAGS)

STOCHASTIC_LIBS = -L${MKLROOT}/lib/intel64 -L${IPPLIB}/lib/intel64 \
$(HOME)/smelt/build/libsmelt.a \
-lmkl_rt -lippcore -lippvm -lipps 


