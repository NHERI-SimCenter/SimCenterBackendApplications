"""Created on Mon Jan  9 09:03:57 2023

@author: snaeimi
"""  # noqa: N999, D400


class Project:  # noqa: D101
    def __init__(self, project_settings, scenario_list):
        self.scenario_list = scenario_list
        self.project_settings = project_settings
