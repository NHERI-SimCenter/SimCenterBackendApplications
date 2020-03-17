include ../../Makefile.inc

all: SiteResponse 

SiteREsponse.o : SiteResponse.cpp

SiteREsponse: SiteREsponse.o
	$(CXX) $(LDFLAGS) -o SiteResponse SiteResponse.o $(LDLIBS)

clean:
	$(RM) *.o *~

distclean: clean
	$(RM) SiteResponse


