import typing  # noqa: CPY001, D100, INP001
from typing import Any

import numpy as np
import pydantic
from pydantic import Field
from typing_extensions import Annotated

_supported_distributions = typing.Literal[
    'Beta',
    'ChiSquared',
    'Exponential',
    'Gamma',
    'Gumbel',
    'Lognormal',
    'Normal',
    'Uniform',
    'Weibull',
]
_supported_input_types = typing.Literal['Parameters', 'Moments', 'Dataset']
_supported_variable_classes = typing.Literal['Uncertain', 'Design', 'Uniform', 'NA']


def _get_ERADistObjectName(name_from_quoFEM: str) -> str:  # noqa: N802, N803
    _ERADistNames = {}  # noqa: N806
    _ERADistNames['ChiSquared'] = 'chisquare'
    try:
        nm = _ERADistNames[name_from_quoFEM].value
    except:  # noqa: E722
        nm = name_from_quoFEM.lower()
    return nm


def _get_ERADistOpt(input_type_from_quoFEM: str) -> str:  # noqa: N802, N803
    _ERADistOpts = {}  # noqa: N806
    _ERADistOpts['Parameters'] = 'PAR'
    _ERADistOpts['Moments'] = 'MOM'
    _ERADistOpts['Dataset'] = 'DATA'
    try:
        opt = _ERADistOpts[input_type_from_quoFEM].value
    except:  # noqa: E722
        opt = 'PAR'
    return opt


class RVData(pydantic.BaseModel):  # noqa: D101
    distribution: _supported_distributions
    name: str
    inputType: _supported_input_types = 'Parameters'  # noqa: N815
    refCount: int  # noqa: N815
    value: str
    variableClass: _supported_variable_classes  # noqa: N815
    ERAName: str = ''
    ERAOpt: str = ''
    ERAVal: list = []

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAName = _get_ERADistObjectName(self.distribution)
        self.ERAOpt = _get_ERADistOpt(self.inputType)
        return super().model_post_init(__context)


############################################
class BetaUncertainData(RVData):  # noqa: D101
    lowerbound: float = 0.0
    upperbound: float = 1.0

    @pydantic.validator('upperbound')
    def upper_bound_not_bigger_than_lower_bound(v, values):  # noqa: ANN001, ANN201, N805, D102
        if 'lowerbound' in values and v <= values['lowerbound']:
            raise ValueError(f"The upper bound must be bigger than the \
                             lower bound {values['lowerbound']}. \
                             Got a value of {v}.")  # noqa: EM102, TRY003
        return v


class BetaParameters(BetaUncertainData):  # noqa: D101
    alphas: pydantic.PositiveFloat
    betas: pydantic.PositiveFloat

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [self.alphas, self.betas, self.lowerbound, self.upperbound]
        return super().model_post_init(__context)


class BetaMoments(BetaUncertainData):  # noqa: D101
    mean: float
    standardDev: pydantic.PositiveFloat  # noqa: N815

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [self.mean, self.standardDev, self.lowerbound, self.upperbound]
        return super().model_post_init(__context)


class BetaDataset(BetaUncertainData):  # noqa: D101
    dataDir: str  # noqa: N815

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [
            np.genfromtxt(self.dataDir).tolist(),
            [self.lowerbound, self.upperbound],
        ]
        return super().model_post_init(__context)


############################################
class ChiSquaredUncertainData(RVData):  # noqa: D101
    pass


class ChiSquaredParameters(ChiSquaredUncertainData):  # noqa: D101
    k: Annotated[int, Field(ge=1)]

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [self.k]
        return super().model_post_init(__context)


class ChiSquaredMoments(ChiSquaredUncertainData):  # noqa: D101
    mean: pydantic.PositiveFloat

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [self.mean]
        return super().model_post_init(__context)


class ChiSquaredDataset(ChiSquaredUncertainData):  # noqa: D101
    dataDir: str  # noqa: N815

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [np.genfromtxt(self.dataDir).tolist()]
        return super().model_post_init(__context)


############################################
class ExponentialUncertainData(RVData):  # noqa: D101
    pass


class ExponentialParameters(ExponentialUncertainData):  # noqa: D101
    lamda: pydantic.PositiveFloat = pydantic.Field(alias='lambda')

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [self.lamda]
        return super().model_post_init(__context)


class ExponentialMoments(ExponentialUncertainData):  # noqa: D101
    mean: pydantic.PositiveFloat

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [self.mean]
        return super().model_post_init(__context)


class ExponentialDataset(ExponentialUncertainData):  # noqa: D101
    dataDir: str  # noqa: N815

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [np.genfromtxt(self.dataDir).tolist()]
        return super().model_post_init(__context)


############################################
class GammaUncertainData(RVData):  # noqa: D101
    pass


class GammaParameters(GammaUncertainData):  # noqa: D101
    k: pydantic.PositiveFloat
    lamda: pydantic.PositiveFloat = pydantic.Field(alias='lambda')

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [self.lamda, self.k]
        return super().model_post_init(__context)


class GammaMoments(GammaUncertainData):  # noqa: D101
    mean: pydantic.PositiveFloat
    standardDev: pydantic.PositiveFloat  # noqa: N815

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [self.mean, self.standardDev]
        return super().model_post_init(__context)


class GammaDataset(GammaUncertainData):  # noqa: D101
    dataDir: str  # noqa: N815

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [np.genfromtxt(self.dataDir).tolist()]
        return super().model_post_init(__context)


############################################
class GumbelUncertainData(RVData):  # noqa: D101
    pass


class GumbelParameters(GumbelUncertainData):  # noqa: D101
    alphaparam: pydantic.PositiveFloat
    betaparam: float

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [self.alphaparam, self.betaparam]
        return super().model_post_init(__context)


class GumbelMoments(GumbelUncertainData):  # noqa: D101
    mean: float
    standardDev: pydantic.PositiveFloat  # noqa: N815

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [self.mean, self.standardDev]
        return super().model_post_init(__context)


class GumbelDataset(GumbelUncertainData):  # noqa: D101
    dataDir: str  # noqa: N815

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [np.genfromtxt(self.dataDir).tolist()]
        return super().model_post_init(__context)


############################################
class LognormalUncertainData(RVData):  # noqa: D101
    pass


class LognormalParameters(LognormalUncertainData):  # noqa: D101
    lamda: float = pydantic.Field(alias='lambda')
    zeta: pydantic.PositiveFloat

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [self.lamda, self.zeta]
        return super().model_post_init(__context)


class LognormalMoments(LognormalUncertainData):  # noqa: D101
    mean: pydantic.PositiveFloat
    stdDev: pydantic.PositiveFloat  # noqa: N815

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [self.mean, self.stdDev]
        return super().model_post_init(__context)


class LognormalDataset(LognormalUncertainData):  # noqa: D101
    dataDir: str  # noqa: N815

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [np.genfromtxt(self.dataDir).tolist()]
        return super().model_post_init(__context)


############################################
class NormalUncertainData(RVData):  # noqa: D101
    pass


class NormalParameters(NormalUncertainData):  # noqa: D101
    mean: float
    stdDev: pydantic.PositiveFloat  # noqa: N815

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [self.mean, self.stdDev]
        return super().model_post_init(__context)


class NormalMoments(NormalUncertainData):  # noqa: D101
    mean: float
    stdDev: pydantic.PositiveFloat  # noqa: N815

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [self.mean, self.stdDev]
        return super().model_post_init(__context)


class NormalDataset(NormalUncertainData):  # noqa: D101
    dataDir: str  # noqa: N815

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [np.genfromtxt(self.dataDir).tolist()]
        return super().model_post_init(__context)


############################################
# class TruncatedExponentialUncertainData(RVData):
#     a: float
#     b: float
#     @pydantic.validator('b')
#     def upper_bound_not_bigger_than_lower_bound(v, values):
#         if 'a' in values and v <= values['a']:
#             raise ValueError(f"The upper bound must be bigger than \
#                              the lower bound {values['a']}. \
#                              Got a value of {v}.")
#         return v

# class TruncatedExponentialParameters(TruncatedExponentialUncertainData):
#     inputType: typing.Literal["Parameters"]
#     lamda: pydantic.PositiveFloat = pydantic.Field(alias="lambda")

# class TruncatedExponentialMoments(TruncatedExponentialUncertainData):
#     inputType: typing.Literal["Moments"]
#     mean: pydantic.PositiveFloat

# class TruncatedExponentialDataset(TruncatedExponentialUncertainData):
#     inputType: typing.Literal["Dataset"]
#     dataDir: str


############################################
class UniformUncertainData(RVData):  # noqa: D101
    pass


class UniformParameters(UniformUncertainData):  # noqa: D101
    lowerbound: float = 0.0
    upperbound: float = 1.0

    @pydantic.validator('upperbound')
    def upper_bound_not_bigger_than_lower_bound(v, values):  # noqa: ANN001, ANN201, N805, D102
        if 'lowerbound' in values and v <= values['lowerbound']:
            raise ValueError(f"The upper bound must be bigger than the \
                             lower bound {values['lowerbound']}. \
                             Got a value of {v}.")  # noqa: EM102, TRY003
        return v

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [self.lowerbound, self.upperbound]
        return super().model_post_init(__context)


class UniformMoments(UniformUncertainData):  # noqa: D101
    mean: float
    standardDev: pydantic.PositiveFloat  # noqa: N815

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [self.mean, self.standardDev]
        return super().model_post_init(__context)


class UniformDataset(UniformUncertainData):  # noqa: D101
    dataDir: str  # noqa: N815

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [np.genfromtxt(self.dataDir).tolist()]
        return super().model_post_init(__context)


############################################
class WeibullUncertainData(RVData):  # noqa: D101
    pass


class WeibullParameters(WeibullUncertainData):  # noqa: D101
    scaleparam: pydantic.PositiveFloat
    shapeparam: pydantic.PositiveFloat

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [self.scaleparam, self.shapeparam]
        return super().model_post_init(__context)


class WeibullMoments(WeibullUncertainData):  # noqa: D101
    mean: pydantic.PositiveFloat
    standardDev: pydantic.PositiveFloat  # noqa: N815

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [self.mean, self.standardDev]
        return super().model_post_init(__context)


class WeibullDataset(WeibullUncertainData):  # noqa: D101
    dataDir: str  # noqa: N815

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401, D102, PYI063
        self.ERAVal = [np.genfromtxt(self.dataDir).tolist()]
        return super().model_post_init(__context)


############################################
