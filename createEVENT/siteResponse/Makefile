RM=rm -f
CPPFLAGS= -I /usr/local/jansson/include -I ./nanoflann/include
LDFLAGS= -static-libgcc
LDLIBS=/usr/local/jansson/lib/libjansson.a

all: SiteResponse 

SiteREsponse.o : SiteResponse.cpp

SiteREsponse: SiteREsponse.o
	$(CXX) $(LDFLAGS) -o SiteResponse SiteResponse.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) SiteResponse


