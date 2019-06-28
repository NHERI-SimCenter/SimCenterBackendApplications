TEMPLATE = subdirs

SUBDIRS =   createEDP/standardEarthquakeEDP/StandardEarthquakeEDP.pro \
            createEDP/standardWindEDP/StandardWindEDP.pro \
            createEDP/gmtEDP/StandardGMTEDP.pro \
            createEDP/userEDP/UserDefinedEDP.pro \
            createSAM/openSeesInput/OpenSeesInput.pro \
            createSAM/mdofBuildingModel/MDOF_BuildingModel.pro\
            createEVENT/multipleSimCenter/MultipleSimCenterEvents.pro \
            createEVENT/multiplePEER/MultiplePEER_Events.pro \
            createEVENT/siteResponse/SiteResponse.pro\
            createEVENT/stochasticGroundMotion/StochasticGroundMotion.pro\
            createEVENT/stochasticWind/StochasticWind.pro\
            performSIMULATION/openSees/OpenSeesPreprocessor.pro \
            performSIMULATION/openSees/OpenSeesPostprocessor.pro \
            performUQ/dakota/extractEDP.pro \
            performUQ/dakota/postprocessDAKOTA.pro
