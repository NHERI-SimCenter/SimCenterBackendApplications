from wntrfr.sim.results import SimulationResults  # noqa: D100


class SimulationResults(SimulationResults):
    """Water network simulation results class."""

    def __init__(self):  # noqa: ANN204, D107
        super().__init__()
        self.maximum_trial_time = []
