# written: Michael Gardner

# DO NOT CHANGE THE FACTORY, JUST IMPORT IT INTO ADDITIONAL DERIVED CLASSES
# Polymorhophic factory for running UQ apps
class UqRunnerFactory:
    factories = {}

    def addFactory(id, runnerFactory):
        UqRunnerFactory.factories.put[id] = runnerFactory

    # A Template Method:
    def createRunner(id):
        if id not in UqRunnerFactory.factories:
            UqRunnerFactory.factories[id] = eval(id + '.Factory()')
        return UqRunnerFactory.factories[id].create()


# Abstract base class
class UqRunner:
    pass
