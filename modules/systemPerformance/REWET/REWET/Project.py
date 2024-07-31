"""Created on Mon Jan  9 09:03:57 2023

@author: snaeimi
"""  # noqa: N999, D400, D415


class Project:  # noqa: D101
    def __init__(self, project_settings, scenario_list):  # noqa: ANN001, ANN204, D107
        self.scenario_list = scenario_list
        self.project_settings = project_settings
