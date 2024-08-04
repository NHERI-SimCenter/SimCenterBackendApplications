"""Created on Thu Nov 10 19:12:46 2022

@author: snaeimi
"""  # noqa: CPY001, D400, INP001

import pickle  # noqa: S403


def getDummyDataForQNExeedanceCurve():  # noqa: N802, D103
    with open('qn_data.pkl', 'rb') as f:  # noqa: PTH123
        dummy_data = pickle.load(f)  # noqa: S301
    return dummy_data  # noqa: RET504


"""
This section is for single scenario results.

"""


"""
This section is for multi-scenarios (probabilistic) results.

"""


def QNExceedanceCurve(pr, percentage_list, time_type, time_shift=0):  # noqa: N802
    """Gets Project Result object, and returns Exceedance probability and Quantity
    outage for the given percentages. Caution: the current version only accept
    one percentage in the percentage list.

    Parameters
    ----------
    pr : Project Result Object
        DESCRIPTION.
    percentage_List : list
        the percentage.
    time_type : 'second', 'hour' or 'day'
    time_shift : in seconds, shift time representing the time before the event

    Excdance probability and outage for each percentage
    -------
    None.

    """  # noqa: D205, D401
    data = getDummyDataForQNExeedanceCurve()

    if len(percentage_list) > 1:
        raise ValueError(  # noqa: DOC501, TRY003
            'the current version only accept one percentage in the percentage list'  # noqa: EM101
        )

    if type(time_shift) != int:  # noqa: E721
        raise ValueError(  # noqa: DOC501
            'Time shift must be integer type: ' + repr(type(time_shift)) + '.'
        )

    if time_shift < 0:
        raise ValueError('Time shift ust be bigger than or equal to zero.')  # noqa: DOC501, EM101, TRY003

    res = {}
    for percentage in percentage_list:
        temp_res = pr.PR_getBSCPercentageExcedanceCurce(data, percentage)

        if time_type.lower() == 'seconds':
            pass
        elif time_type.lower() == 'hour':
            pr.convertTimeSecondToHour(temp_res, 'restore_time', time_shift)
        elif time_type.lower() == 'day':
            pr.convertTimeSecondToDay(temp_res, 'restore_time', time_shift)
        else:
            raise ValueError('Uknown time_type: ' + repr(time_type))  # noqa: DOC501

        res[percentage] = temp_res
    return res


def DLExceedanceCurve(pr, percentage_list, time_type, time_shift=0):  # noqa: N802
    """Gets Project Result object, and returns Exceedance probability and Delivery
    outage for the given percentages. Caution: the current version only accept
    one percentage in the percentage list.

    Parameters
    ----------
    pr : Project Result Object
        DESCRIPTION.
    percentage_List : list
        the percentage.
    time_type : 'second', 'hour' or 'day'
    time_shift : in seconds, shift time representing the time before the event

    Excdance probability and outage for each percentage
    -------
    None.

    """  # noqa: D205, D401
    data = getDummyDataForQNExeedanceCurve()

    if len(percentage_list) > 1:
        raise ValueError(  # noqa: DOC501, TRY003
            'the current version only accept one percentage in the percentage list'  # noqa: EM101
        )

    if type(time_shift) != int:  # noqa: E721
        raise ValueError(  # noqa: DOC501
            'Time shift must be integer type: ' + repr(type(time_shift)) + '.'
        )

    if time_shift < 0:
        raise ValueError('Time shift ust be bigger than or equal to zero.')  # noqa: DOC501, EM101, TRY003

    res = {}
    for percentage in percentage_list:
        temp_res = pr.PR_getBSCPercentageExcedanceCurce(data, percentage)

        if time_type.lower() == 'seconds':
            pass
        elif time_type.lower() == 'hour':
            pr.convertTimeSecondToHour(temp_res, 'restore_time', time_shift)
        elif time_type.lower() == 'day':
            pr.convertTimeSecondToDay(temp_res, 'restore_time', time_shift)
        else:
            raise ValueError('Uknown time_type: ' + repr(time_type))  # noqa: DOC501

        res[percentage] = temp_res
    return res
