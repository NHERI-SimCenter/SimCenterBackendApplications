SHELL= /bin/sh

SIMCENTER_DEVDIR=../..
GALLERY=$(SIMCENTER_DEVDIR)/SimCenterDocumentation/docs/common/developer_manual/architecture/desktop/applications/_static/html
AURORE=aurore

CACHE = .aurore/aurore.cache.json
RENDRE = rendre

APPS = quoFEM WE-UQ EE-UQ RDT PBE
MODULES = createBIM createEDP createEVENT createSAM
MODULES += performDL performRegionalMapping performSIMULATION performUQ Workflow
MODULES += preprocSIM preprocUQ


.PHONY: cache all

all:
	for app in $(APPS); do \
	  echo "$${app}-------------------------------------------------"; \
	  for mod in $(MODULES); do \
		echo "MODULE: $$mod"; \
		if [ "$$($(RENDRE) -l index.yaml#/$$app -q list -e \%c:$$mod )" ] ; then \
			$(RENDRE) -l index.yaml#/$$app -q gallery -e \%c:$$mod > $(GALLERY)/$$app/$$mod.html; \
			echo ''; \
		fi; \
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

