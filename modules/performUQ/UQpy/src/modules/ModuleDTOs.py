from typing import Literal, Union

from pydantic import BaseModel, Field
from src.reliability.ReliabilityMethodsDTOs import ReliabilityMethod
from typing_extensions import Annotated


class ModuleBaseDTO(BaseModel):
    pass


class SamplingDTO(ModuleBaseDTO):
    uqType: Literal['Sampling'] = 'Sampling'

    def generate_code(self):
        pass


class SurrogatesDTO(ModuleBaseDTO):
    uqType: Literal['Surrogates'] = 'Surrogates'


class ReliabilityDTO(ModuleBaseDTO):
    uqType: Literal['Reliability Analysis'] = 'Reliability Analysis'
    methodData: ReliabilityMethod


ModuleDTO = Annotated[
    Union[ReliabilityDTO, SamplingDTO], Field(discriminator='uqType')
]
