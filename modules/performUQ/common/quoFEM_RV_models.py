from typing import Any
import pydantic
import typing
from pydantic import Field
from typing_extensions import Annotated
import numpy as np


_supported_distributions = typing.Literal["Beta", "ChiSquared", "Exponential", "Gamma", "Gumbel", "Lognormal", "Normal", "Uniform", "Weibull"]
_supported_input_types = typing.Literal["Parameters", "Moments", "Dataset"]
_supported_variable_classes = typing.Literal["Uncertain", "Design", "Uniform", "NA"]


def _get_ERADistObjectName(name_from_quoFEM: str) -> str:
    _ERADistNames = {}
    _ERADistNames["ChiSquared"] = "chisquare"
    try:
        nm = _ERADistNames[name_from_quoFEM].value
    except:
        nm = name_from_quoFEM.lower()
    return nm


def _get_ERADistOpt(input_type_from_quoFEM: str) -> str:
    _ERADistOpts = {}
    _ERADistOpts["Parameters"] = "PAR"
    _ERADistOpts["Moments"] = "MOM"
    _ERADistOpts["Dataset"] = "DATA"
    try:
        opt = _ERADistOpts[input_type_from_quoFEM].value
    except:
        opt = "PAR"
    return opt


class RVData(pydantic.BaseModel):
    distribution: _supported_distributions
    name: str
    inputType: _supported_input_types = "Parameters"
    refCount: int
    value: str
    variableClass: _supported_variable_classes
    ERAName: str = ""
    ERAOpt: str = ""
    ERAVal: list = []

    def model_post_init(self, __context: Any) -> None:
        self.ERAName = _get_ERADistObjectName(self.distribution)
        self.ERAOpt = _get_ERADistOpt(self.inputType)
        return super().model_post_init(__context)


############################################     
class BetaUncertainData(RVData):
    lowerbound: float = 0.0
    upperbound: float = 1.0
    @pydantic.validator('upperbound')
    def upper_bound_not_bigger_than_lower_bound(v, values):
        if 'lowerbound' in values and v <= values['lowerbound']:
            raise ValueError(f"The upper bound must be bigger than the lower bound {values['lowerbound']}. Got a value of {v}.")
        return v


class BetaParameters(BetaUncertainData):
    alphas: pydantic.PositiveFloat
    betas: pydantic.PositiveFloat

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [self.alphas, self.betas, self.lowerbound, self.upperbound]
        return super().model_post_init(__context)


class BetaMoments(BetaUncertainData):
    mean: float
    standardDev: pydantic.PositiveFloat

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [self.mean, self.standardDev, self.lowerbound, self.upperbound]
        return super().model_post_init(__context)


class BetaDataset(BetaUncertainData):
    dataDir: str

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [np.genfromtxt(self.dataDir).tolist(), [self.lowerbound, self.upperbound]]
        return super().model_post_init(__context)

############################################
class ChiSquaredUncertainData(RVData):
    pass


class ChiSquaredParameters(ChiSquaredUncertainData):
    k: Annotated[int, Field(ge=1)]

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [self.k]
        return super().model_post_init(__context)


class ChiSquaredMoments(ChiSquaredUncertainData):
    mean: pydantic.PositiveFloat

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [self.mean]
        return super().model_post_init(__context)


class ChiSquaredDataset(ChiSquaredUncertainData):
    dataDir: str

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [np.genfromtxt(self.dataDir).tolist()]
        return super().model_post_init(__context)

############################################
class ExponentialUncertainData(RVData):
    pass


class ExponentialParameters(ExponentialUncertainData):
    lamda: pydantic.PositiveFloat = pydantic.Field(alias="lambda")

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [self.lamda]
        return super().model_post_init(__context)
    

class ExponentialMoments(ExponentialUncertainData):
    mean: pydantic.PositiveFloat

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [self.mean]
        return super().model_post_init(__context)


class ExponentialDataset(ExponentialUncertainData):
    dataDir: str

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [np.genfromtxt(self.dataDir).tolist()]
        return super().model_post_init(__context)
    
############################################
class GammaUncertainData(RVData):
    pass


class GammaParameters(GammaUncertainData):
    k: pydantic.PositiveFloat
    lamda: pydantic.PositiveFloat = pydantic.Field(alias="lambda")

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [self.lamda, self.k]
        return super().model_post_init(__context)
    

class GammaMoments(GammaUncertainData):
    mean: pydantic.PositiveFloat
    standardDev: pydantic.PositiveFloat

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [self.mean, self.standardDev]
        return super().model_post_init(__context)


class GammaDataset(GammaUncertainData):
    dataDir: str

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [np.genfromtxt(self.dataDir).tolist()]
        return super().model_post_init(__context)
    
############################################
class GumbelUncertainData(RVData):
    pass


class GumbelParameters(GumbelUncertainData):
    alphaparam: pydantic.PositiveFloat
    betaparam: float

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [self.alphaparam, self.betaparam]
        return super().model_post_init(__context)
    

class GumbelMoments(GumbelUncertainData):
    mean: float
    standardDev: pydantic.PositiveFloat

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [self.mean, self.standardDev]
        return super().model_post_init(__context)


class GumbelDataset(GumbelUncertainData):
    dataDir: str

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [np.genfromtxt(self.dataDir).tolist()]
        return super().model_post_init(__context)
    

############################################
class LognormalUncertainData(RVData):
    pass


class LognormalParameters(LognormalUncertainData):
    lamda: float = pydantic.Field(alias="lambda")
    zeta: pydantic.PositiveFloat

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [self.lamda, self.zeta]
        return super().model_post_init(__context)
    

class LognormalMoments(LognormalUncertainData):
    mean: pydantic.PositiveFloat
    stdDev: pydantic.PositiveFloat

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [self.mean, self.stdDev]
        return super().model_post_init(__context)


class LognormalDataset(LognormalUncertainData):
    dataDir: str

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [np.genfromtxt(self.dataDir).tolist()]
        return super().model_post_init(__context)
    

############################################
class NormalUncertainData(RVData):
    pass


class NormalParameters(NormalUncertainData):
    mean: float
    stdDev: pydantic.PositiveFloat

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [self.mean, self.stdDev]
        return super().model_post_init(__context)
    

class NormalMoments(NormalUncertainData):
    mean: float
    stdDev: pydantic.PositiveFloat

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [self.mean, self.stdDev]
        return super().model_post_init(__context)


class NormalDataset(NormalUncertainData):
    dataDir: str

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [np.genfromtxt(self.dataDir).tolist()]
        return super().model_post_init(__context)
    
############################################     
# class TruncatedExponentialUncertainData(TruncatedExponentialUncertainVariable):
#     a: float
#     b: float
#     @pydantic.validator('b')
#     def upper_bound_not_bigger_than_lower_bound(v, values):
#         if 'a' in values and v <= values['a']:
#             raise ValueError(f"The upper bound must be bigger than the lower bound {values['a']}. Got a value of {v}.")
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
class UniformUncertainData(RVData):
    pass


class UniformParameters(UniformUncertainData):
    lowerbound: float = 0.0
    upperbound: float = 1.0
    @pydantic.validator('upperbound')
    def upper_bound_not_bigger_than_lower_bound(v, values):
        if 'lowerbound' in values and v <= values['lowerbound']:
            raise ValueError(f"The upper bound must be bigger than the lower bound {values['lowerbound']}. Got a value of {v}.")
        return v
    
    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [self.lowerbound, self.upperbound]
        return super().model_post_init(__context)


class UniformMoments(UniformUncertainData):
    mean: float
    standardDev: pydantic.PositiveFloat

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [self.mean, self.standardDev]
        return super().model_post_init(__context)


class UniformDataset(UniformUncertainData):
    dataDir: str

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [np.genfromtxt(self.dataDir).tolist()]
        return super().model_post_init(__context)
    

############################################
class WeibullUncertainData(RVData):
    pass


class WeibullParameters(WeibullUncertainData):
    scaleparam: pydantic.PositiveFloat
    shapeparam: pydantic.PositiveFloat

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [self.scaleparam, self.shapeparam]
        return super().model_post_init(__context)
    

class WeibullMoments(WeibullUncertainData):
    mean: pydantic.PositiveFloat
    standardDev: pydantic.PositiveFloat

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [self.mean, self.standardDev]
        return super().model_post_init(__context)


class WeibullDataset(WeibullUncertainData):
    dataDir: str

    def model_post_init(self, __context: Any) -> None:
        self.ERAVal = [np.genfromtxt(self.dataDir).tolist()]
        return super().model_post_init(__context)
    
############################################