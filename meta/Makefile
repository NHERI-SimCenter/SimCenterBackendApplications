SIMCENTER_DEVDIR=../..
# GALLERY=$(SIMCENTER_DEVDIR)/SimCenterDocumentation/docs/common/developer_manual/architecture/desktop/backendApplications/_static/html
GALLERY=$(SIMCENTER_DEVDIR)/SimCenterDocumentation/docs/common/developer_manual/architecture/desktop/applications/_static/html
AURORE=aurore #-D ./aurore.db.xml 

MODULES = createBIM createEDP createEVENT createSAM
MODULES += performDL performRegionalMapping performSIMULATION performUQ Workflow
MODULES += wind-events earthquake-events

CACHE = .aurore/aurore.cache.json
RENDRE = cat $(CACHE) | rendre

.PHONY: cache

all:
	for mod in $(MODULES); do \
		$(RENDRE) -q gallery -i \%c:$$mod > $(GALLERY)/$$mod.html; \
	done

cache:
	$(AURORE) -q get > $(CACHE)




4:
	$(AURORE) -vvv -C ./categories.xml get

wind-events:
	rendre -d defaults.yaml#/wind-events gallery < $(CACHE)


