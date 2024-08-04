"""Created on Tue Nov  1 23:25:30 2022

@author: snaeimi
"""  # noqa: CPY001, D400, N999

from .Damage_Discovery_Designer import Damage_Discovery_Designer


class Pump_Damage_Discovery_Designer(Damage_Discovery_Designer):  # noqa: D101
    def __init__(self, pump_damage_discovery_model):
        super().__init__(pump_damage_discovery_model)
        self._window.setWindowTitle('Pump Damage Discovery')
        self.leak_based_radio.setEnabled(False)
