"""Created on Tue Nov  1 23:25:30 2022

@author: snaeimi
"""  # noqa: N999, D400

from .Damage_Discovery_Designer import Damage_Discovery_Designer


class Pipe_Damage_Discovery_Designer(Damage_Discovery_Designer):  # noqa: D101
    def __init__(self, pipe_damage_discovery_model):
        super().__init__(pipe_damage_discovery_model)
        self._window.setWindowTitle('Pipe Damage Discovery')
