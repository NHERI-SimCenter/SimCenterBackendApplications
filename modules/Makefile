all:         
	(cd createEDP; make all)
	(cd createEVENT; make all)
	(cd createSAM; make all)
	(cd performRegionalEventSimulation; make all)
	(cd performSIMULATION; make all)
	(cd performUQ; make all)
	(cd Workflow; make all)

clean: 
	rm -f *.o *~
	(cd createEDP; make clean)
	(cd createEVENT; make clean)
	(cd createSAM; make clean)
	(cd performRegionalEventSimulation; make clean)
	(cd performSIMULATION; make clean)
	(cd performUQ; make clean)
	(cd Workflow; make clean)

distclean: clean
	(cd createEDP; make distclean)
	(cd createEVENT; make distclean)
	(cd createSAM; make distclean)
	(cd performRegionalEventSimulation; make distclean)
	(cd performSIMULATION; make distclean)
	(cd performUQ; make distclean)
	(cd Workflow; make distclean)

# DO NOT DELETE THIS LINE -- make depend depends on it.
