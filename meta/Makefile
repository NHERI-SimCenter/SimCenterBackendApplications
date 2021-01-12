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


.PHONY: cache

cache:
	$(AURORE) -q get > $(CACHE)

