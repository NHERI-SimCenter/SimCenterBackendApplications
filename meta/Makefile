SIMCENTER_DEVDIR=../..
# GALLERY=$(SIMCENTER_DEVDIR)/SimCenterDocumentation/docs/common/developer_manual/architecture/desktop/backendApplications/_static/html
GALLERY=$(SIMCENTER_DEVDIR)/cmp-12-6/docs/common/developer_manual/architecture/desktop/applications/_static/html
AURORE=aurore -D ./aurore.db.xml 

MODULES = createBIM createEDP createEVENT createSAM
MODULES += performDL performRegionalMapping performSIMULATION performUQ Workflow

CACHE = cache/metadata-cache.json
RENDRE = "cat $(CACHE) | rendre"

.PHONY: cache

all:
	for mod in $(MODULES); do \
		$(AURORE) -F category:term:$$mod -q feed -t tmpl-0003 > $(GALLERY)/$$mod-gallery.html; \
	done

cache:
	$(AURORE) -C categories.xml get > $(CACHE)


gallery:
	for mod in $(MODULES); do \
		$(RENDRE) -F categories/workflow-module:$$mod -q feed -t tmpl-0003 > $(GALLERY)/$$mod-gallery.html; \
	done


1:
	aurore -I '$(SIMCENTER_DEVDIR)/SimCenterBackendApplications/meta/entities.xml#bknd-0001' get


2:
	aurore -I '$(SIMCENTER_DEVDIR)/SimCenterBackendApplications/meta/entities.xml#bknd-0001' \
	       -I '$(SIMCENTER_DEVDIR)/SimCenterBackendApplications/meta/entities.xml#bknd-0002' \
		   get
3:
	$(AURORE) get


4:
	$(AURORE) -C ./categories.xml get

wind-events:
	rendre -d defaults.yaml#/wind-events gallery < $(CACHE)