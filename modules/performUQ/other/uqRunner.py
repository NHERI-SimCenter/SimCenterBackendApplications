# written: Michael Gardner  # noqa: INP001, D100

# DO NOT CHANGE THE FACTORY, JUST IMPORT IT INTO ADDITIONAL DERIVED CLASSES
# Polymorhophic factory for running UQ apps
class UqRunnerFactory:  # noqa: D101
    factories = {}  # noqa: RUF012

    def addFactory(id, runnerFactory):  # noqa: A002, N802, N803, N805, D102
        UqRunnerFactory.factories.put[id] = runnerFactory

    # A Template Method:
    def createRunner(id):  # noqa: A002, N802, N805, D102
        if id not in UqRunnerFactory.factories:
            UqRunnerFactory.factories[id] = eval(id + '.Factory()')  # noqa: S307
        return UqRunnerFactory.factories[id].create()


# Abstract base class
class UqRunner:  # noqa: D101
    pass
