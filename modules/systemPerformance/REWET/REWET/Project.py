"""Created on Mon Jan  9 09:03:57 2023

@author: snaeimi
"""  # noqa: CPY001, D400, N999


class Project:  # noqa: D101
    def __init__(self, project_settings, scenario_list):  # noqa: ANN001, ANN204
        self.scenario_list = scenario_list
        self.project_settings = project_settings
