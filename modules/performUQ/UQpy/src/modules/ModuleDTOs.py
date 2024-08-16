from typing import Literal, Union  # noqa: INP001, D100

from pydantic import BaseModel, Field
from src.reliability.ReliabilityMethodsDTOs import ReliabilityMethod
from typing_extensions import Annotated


class ModuleBaseDTO(BaseModel):  # noqa: D101
    pass


class SamplingDTO(ModuleBaseDTO):  # noqa: D101
    uqType: Literal['Sampling'] = 'Sampling'  # noqa: N815

    def generate_code(self):  # noqa: D102
        pass


class SurrogatesDTO(ModuleBaseDTO):  # noqa: D101
    uqType: Literal['Surrogates'] = 'Surrogates'  # noqa: N815


class ReliabilityDTO(ModuleBaseDTO):  # noqa: D101
    uqType: Literal['Reliability Analysis'] = 'Reliability Analysis'  # noqa: N815
    methodData: ReliabilityMethod  # noqa: N815


ModuleDTO = Annotated[
    Union[ReliabilityDTO, SamplingDTO], Field(discriminator='uqType')
]
