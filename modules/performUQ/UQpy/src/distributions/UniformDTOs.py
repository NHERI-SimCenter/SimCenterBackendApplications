from typing import Literal, Union  # noqa: CPY001, D100, INP001

import numpy as np
from pydantic import BaseModel, Field, PositiveFloat, validator
from typing_extensions import Annotated


class RVCommonData(BaseModel):  # noqa: D101
    name: str
    value: str
    refCount: int  # noqa: N815


class UniformParameters(RVCommonData):  # noqa: D101
    variableClass: Literal['Uncertain']  # noqa: N815
    distribution: Literal['Uniform']
    inputType: Literal['Parameters']  # noqa: N815
    lowerbound: float = 0.0
    upperbound: float = 1.0

    @validator('upperbound')
    def upper_bound_not_bigger_than_lower_bound(v, values):  # noqa: N805, D102
        if 'lowerbound' in values and v <= values['lowerbound']:
            raise ValueError(  # noqa: TRY003
                f"The upper bound must be bigger than the lower bound {values['lowerbound']}. Got a value of {v}."  # noqa: EM102
            )
        return v

    def init_to_text(self):  # noqa: D102
        from UQpy.distributions.collection.Uniform import Uniform  # noqa: PLC0415

        c = Uniform

        class_name = c.__module__.split('.')[-1]
        import_statement = 'from ' + c.__module__ + ' import ' + class_name + '\n'
        import_statement_2 = 'from UQpy.distributions import JointIndependent \n'
        scipy_inputs = self._to_scipy()
        input_str = self.name
        initializer = f"{self.name} = {class_name}(loc={scipy_inputs['loc']}, scale={scipy_inputs['scale']})"
        prerequisite_str = import_statement + import_statement_2 + initializer
        return prerequisite_str, input_str

    def _to_scipy(self):
        loc = self.lowerbound
        scale = self.upperbound - self.lowerbound
        return {'loc': loc, 'scale': scale}


class UniformMoments(RVCommonData):  # noqa: D101
    variableClass: Literal['Uncertain']  # noqa: N815
    distribution: Literal['Uniform']
    inputType: Literal['Moments']  # noqa: N815
    mean: float
    standardDev: PositiveFloat  # noqa: N815

    def _to_scipy(self):
        loc = self.mean - np.sqrt(12) * self.standardDev / 2
        scale = np.sqrt(12) * self.standardDev
        return {'loc': loc, 'scale': scale}


class UniformDataset(RVCommonData):  # noqa: D101
    variableClass: Literal['Uncertain']  # noqa: N815
    distribution: Literal['Uniform']
    inputType: Literal['Dataset']  # noqa: N815
    dataDir: str  # noqa: N815

    def _to_scipy(self):
        data = readFile(self.dataDir)
        low = np.min(data)
        high = np.max(data)
        return {'loc': low, 'scale': high - low}


def readFile(path):  # noqa: N802, D103
    with open(path) as f:  # noqa: PLW1514, PTH123
        return np.genfromtxt(f)


DistributionDTO = Annotated[
    Union[UniformParameters, UniformMoments, UniformDataset],
    Field(discriminator='inputType'),
]
