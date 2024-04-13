from wntrfr.sim.results import SimulationResults


class SimulationResults(SimulationResults):
    """
    Water network simulation results class.
    """

    def __init__(self):
        super().__init__()
        self.maximum_trial_time = []
