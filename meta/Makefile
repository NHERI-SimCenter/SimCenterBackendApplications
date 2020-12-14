SIMCENTER_DEVDIR=../..
GALLERY=$(SIMCENTER_DEVDIR)/SimCenterDocumentation/docs/common/developer_manual/architecture/desktop/applications/_static/html
AURORE=aurore #-D ./aurore.db.xml 

CACHE = .aurore/aurore.cache.json
RENDRE = cat $(CACHE) | rendre

APPS = quoFEM WE-UQ EE-UQ RDT
MODULES = createBIM createEDP createEVENT createSAM
MODULES += performDL performRegionalMapping performSIMULATION performUQ Workflow


.PHONY: cache all

all:
	for app in $(APPS); do \
	  for mod in $(MODULES); do \
	    $(RENDRE) -d index.yaml#/$$app -q gallery -i \%c:$$mod > $(GALLERY)/$$app/$$mod.html; \
	  done \
	done \

cache:
	$(AURORE) -q get > $(CACHE)

