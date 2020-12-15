SIMCENTER_DEVDIR=../..
GALLERY=$(SIMCENTER_DEVDIR)/SimCenterDocumentation/docs/common/developer_manual/architecture/desktop/applications/_static/html
AURORE=aurore #-D ./aurore.db.xml 

CACHE = .aurore/aurore.cache.json
# RENDRE = cat $(CACHE) | rendre
RENDRE = rendre

APPS = quoFEM WE-UQ EE-UQ RDT
MODULES = createBIM createEDP createEVENT createSAM
MODULES += performDL performRegionalMapping performSIMULATION performUQ Workflow


.PHONY: cache all

all:
	for app in $(APPS); do \
	  echo "$${app}-------------------------------------------------"; \
	  for mod in $(MODULES); do \
		echo "MODULE: $$mod"; \
	    $(RENDRE) -d index.yaml#/$$app -q list -e \%c:$$mod ; \
	    $(RENDRE) -d index.yaml#/$$app -q gallery -e \%c:$$mod > $(GALLERY)/$$app/$$mod.html; \
		echo ''; \
	  done \
	done \

dry:
	for app in $(APPS); do \
	  echo "$${app}-------------------------------------------------"; \
	  for mod in $(MODULES); do \
		echo "$$mod"; \
	    $(RENDRE) -d index.yaml#/$$app -q list -e \%c:$$mod ; \
		echo ''; \
	  done \
	done \

cache:
	$(AURORE) -q get > $(CACHE)

