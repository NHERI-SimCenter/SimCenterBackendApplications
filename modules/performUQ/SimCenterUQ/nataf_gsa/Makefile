CC=mpicc
CXX=mpicxx
-Dxxx=MPI_RUN
CFLAGS=-mkl -stdc++17 -I. -I./nlopt/include -I/home1/apps/intel19/eigen/3.3.7/include/eigen3 -I./lib_armadillo/armadillo-10.1.0/include

LINK_LIBS=-L./nlopt/lib -lnlopt -lstdc++fs

OBJ = main.o \
	ERADist.o \
	exponentialDist.o \
	normalDist.o \
	ERANataf.o \
	gammaDist.o \
	runGSA.o \
	runForward.o \
	RVDist.o \
	gumbelDist.o \
	truncExponentialDist.o \
	betaDist.o \
	jsonInput.o \
	uniformDist.o \
	chiSquaredDist.o \
	lognormalDist.o \
	weibullDist.o \
	discreteDist.o \
	writeErrors.o

%.o: %.c 
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cpp 
	$(CXX) -c -o $@ $< $(CFLAGS)

nataf_gsa: $(OBJ)
	$(CXX) -o $@ $^ $(CFLAGS) $(LINK_LIBS)

clean :
	rm -fr *.o *~

ready:
	mkdir nlohmann
	cp -avr ~/.conan/data/jsonformoderncpp/3.7.0/vthiery/stable/package/5ab84d6acfe1f23c4fae0ab88f26e3a396351ac9/include/* ./
	mkdir nlopt
	cp -avr ~/.conan/data/nlopt/2.6.2/_/_/package/6855b024b359009f69c23257dfa495789e05df16/include ./nlopt
	cp -avr ~/.conan/data/nlopt/2.6.2/_/_/package/6855b024b359009f69c23257dfa495789e05df16/lib ./nlopt
