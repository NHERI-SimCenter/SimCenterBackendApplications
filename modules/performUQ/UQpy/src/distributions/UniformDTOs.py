from typing import Literal, Union

import numpy as np
from pydantic import BaseModel, validator, PositiveFloat, Field
from typing_extensions import Annotated


class RVCommonData(BaseModel):
    name: str
    value: str
    refCount: int


class UniformParameters(RVCommonData):
    variableClass: Literal["Uncertain"]
    distribution: Literal["Uniform"]
    inputType: Literal["Parameters"]
    lowerbound: float = 0.0
    upperbound: float = 1.0

    @validator('upperbound')
    def upper_bound_not_bigger_than_lower_bound(v, values):
        if 'lowerbound' in values and v <= values['lowerbound']:
            raise ValueError(
                f"The upper bound must be bigger than the lower bound {values['lowerbound']}. Got a value of {v}.")
        return v

    def init_to_text(self):
        from UQpy.distributions.collection.Uniform import Uniform
        c = Uniform

        class_name = c.__module__.split(".")[-1]
        import_statement = "from " + c.__module__ + " import " + class_name + "\n"
        import_statement_2 = "from UQpy.distributions import JointIndependent \n" 
        scipy_inputs = self._to_scipy()
        input_str = self.name
        initializer = f"{self.name} = {class_name}(loc={scipy_inputs['loc']}, scale={scipy_inputs['scale']})"
        prerequisite_str = import_statement + import_statement_2 + initializer
        return prerequisite_str, input_str

    def _to_scipy(self):
        loc = self.lowerbound
        scale = self.upperbound - self.lowerbound
        return {"loc": loc, "scale": scale}


class UniformMoments(RVCommonData):
    variableClass: Literal["Uncertain"]
    distribution: Literal["Uniform"]
    inputType: Literal["Moments"]
    mean: float
    standardDev: PositiveFloat

    def _to_scipy(self):
        loc = self.mean - np.sqrt(12) * self.standardDev / 2
        scale = np.sqrt(12) * self.standardDev
        return {"loc": loc, "scale": scale}


class UniformDataset(RVCommonData):
    variableClass: Literal["Uncertain"]
    distribution: Literal["Uniform"]
    inputType: Literal["Dataset"]
    dataDir: str

    def _to_scipy(self):
        data = readFile(self.dataDir)
        low = np.min(data)
        high = np.max(data)
        return {"loc": low, "scale": high - low}


def readFile(path):
    with open(path, "r") as f:
        return np.genfromtxt(f)


DistributionDTO = Annotated[Union[UniformParameters, UniformMoments, UniformDataset], Field(discriminator='inputType')]
