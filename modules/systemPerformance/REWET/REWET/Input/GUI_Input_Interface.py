"""Created on Mon Oct 10 15:04:40 2022

This is the input module, an interface between all inputs from GUI, TEXT-based
inputs and the mail code.

@author: snaeimi
"""  # noqa: INP001, D400, D415


class input:  # noqa: A001, N801, D101
    def __init__(self, settings, registry):  # noqa: ANN001, ANN204, D107
        pass

    def convertShiftFromDictToPandasTable(self, dict_data):  # noqa: ANN001, ANN201, ARG002, N802, D102
        shift_name_list = list(shift_data)  # noqa: F821
        shift_begining_list = [shift_data[i][0] for i in shift_name_list]  # noqa: F821, F841
        shift_end_list = [shift_data[i][1] for i in shift_name_list]  # noqa: F821, F841
