import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator

sys.path.append(
    "/Users/aakash/SimCenter/SimCenterBackendApplications/modules/performUQ/common"
)


class EDPItem(BaseModel):
    name: str
    type: Literal["scalar", "field"]
    length: int = Field(gt=0)


# class EDP(BaseModel):
#     __root__: list[EDPItem]


class CorrelationValue(BaseModel):
    values: float = Field(..., ge=-1, le=1)

    # class CorrelationMatrix(BaseModel):
    #     CorrelationMatrix: list[CorrelationValue]

    # class CorrelationMatrix(BaseModel):
    #     __root__: Annotated[
    #         List[Annotated[float, Field(ge=-1, le=1)]], Field()
    #     ]  # List of floats, each between -1 and 1

    #     @model_validator(mode="after")
    #     def check_square_matrix(cls, model):
    #         matrix = model.correlationMatrix  # Access the matrix directly from the model
    #         if matrix:
    #             length = len(matrix)
    #             sqrt_len = math.isqrt(length)  # Integer square root
    #             if sqrt_len * sqrt_len != length:
    #                 raise ValueError(
    #                     "The length of the correlationMatrix must be a perfect square."
    #                 )
    #         return model

    # @classmethod
    # def from_list(cls, correlation_matrix: List[float]):
    #     """Construct the model directly from a list."""
    #     return cls(correlationMatrix=correlation_matrix)


class ApplicationData(BaseModel):
    MS_Path: Path
    mainScript: str
    postprocessScript: str


class FEM_App(BaseModel):
    Application: str
    ApplicationData: ApplicationData


class UQ_App(BaseModel):
    Application: str
    ApplicationData: dict[str, Any]


class Applications(BaseModel):
    FEM: FEM_App
    UQ: UQ_App


class GP_AB_UQData(BaseModel):
    calibration_data_file_name: str
    calibration_data_path: Path
    log_likelihood_file_name: Optional[str]
    log_likelihood_path: Optional[Path]
    sample_size: int
    seed: int


# class Model(BaseModel):
#     Applications: Applications
#     EDP: EDP
#     FEM: dict[str, Any]
#     UQ: GP_AB_UQData
#     correlationMatrix: CorrelationMatrix
#     localAppDir: str
#     randomVariables: list[dict[str, Any]]
#     remoteAppDir: str
#     runType: str
#     workingDir: str


class Model(BaseModel):
    Applications: Applications
    EDP: List[EDPItem]
    FEM: Dict[str, str]  # Empty dict
    UQ: GP_AB_UQData
    WorkflowType: str
    correlationMatrix: List[float]
    localAppDir: str
    randomVariables: list[dict[str, Any]]
    remoteAppDir: str
    resultType: str
    runDir: str
    runType: str
    summary: List[str]
    workingDir: str

    @validator("correlationMatrix", each_item=True)
    def check_correlation_matrix(cls, v):
        if not (-1 <= v <= 1):
            raise ValueError("Each correlation value must be between -1 and 1.")
        return v
