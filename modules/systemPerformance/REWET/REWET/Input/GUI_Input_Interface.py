"""Created on Mon Oct 10 15:04:40 2022

This is the input module, an interface between all inputs from GUI, TEXT-based
inputs and the mail code.

@author: snaeimi
"""  # noqa: CPY001, D400, INP001


class input:  # noqa: A001, D101
    def __init__(self, settings, registry):  # noqa: ANN001, ANN204
        pass

    def convertShiftFromDictToPandasTable(self, dict_data):  # noqa: ANN001, ANN201, ARG002, D102, N802, PLR6301
        shift_name_list = list(shift_data)  # noqa: F821
        shift_begining_list = [shift_data[i][0] for i in shift_name_list]  # noqa: F821, F841
        shift_end_list = [shift_data[i][1] for i in shift_name_list]  # noqa: F821, F841
