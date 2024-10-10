"""
Defines data models for the performUQ application using Pydantic.

Classes:
    EDPItem: Represents an Engineering Demand Parameter item.
    CorrelationValue: Represents a correlation value.
    ApplicationData: Represents application data.
    FEM_App: Represents a FEM application.
    UQ_App: Represents a UQ application.
    Applications: Represents a collection of applications.
    GP_AB_UQData: Represents GP_AB_UQ data.
    Model: Represents the main model.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:
    from pathlib import Path

from pydantic import BaseModel, Field, validator

sys.path.append(
    '/Users/aakash/SimCenter/SimCenterBackendApplications/modules/performUQ/common'
)


class EDPItem(BaseModel):
    """
    Represents an EDP (Engineering Demand Parameter) item.

    Attributes
    ----------
        name (str): The name of the EDP item.
        type (Literal["scalar", "field"]): The type of the EDP item, either "scalar" or "field".
        length (int): The length of the EDP item, must be greater than 0.
    """

    name: str
    type: Literal['scalar', 'field']
    length: int = Field(gt=0)


class CorrelationValue(BaseModel):
    """
    Represents a correlation value.

    Attributes
    ----------
        values (float): The correlation value, must be between -1 and 1.
    """

    values: float = Field(..., ge=-1, le=1)


class ApplicationData(BaseModel):
    """
    Represents application data.

    Attributes
    ----------
        MS_Path (Path): The path to the MS.
        mainScript (str): The main script name.
        postprocessScript (str): The post-process script name.
    """

    MS_Path: Path
    mainScript: str  # noqa: N815
    postprocessScript: str  # noqa: N815


class FEM_App(BaseModel):
    """
    Represents a FEM (Finite Element Method) application.

    Attributes
    ----------
        Application (str): The name of the application.
        ApplicationData (ApplicationData): The application data.
    """

    Application: str
    ApplicationData: ApplicationData


class UQ_App(BaseModel):
    """
    Represents a UQ (Uncertainty Quantification) application.

    Attributes
    ----------
        Application (str): The name of the application.
        ApplicationData (dict[str, Any]): The application data.
    """

    Application: str
    ApplicationData: dict[str, Any]


class Applications(BaseModel):
    """
    Represents a collection of applications.

    Attributes
    ----------
        FEM (FEM_App): The FEM application.
        UQ (UQ_App): The UQ application.
    """

    FEM: FEM_App
    UQ: UQ_App


class GP_AB_UQData(BaseModel):
    """
    Represents GP_AB_UQ data.

    Attributes
    ----------
        calibration_data_file_name (str): The name of the calibration data file.
        calibration_data_path (Path): The path to the calibration data.
        log_likelihood_file_name (Optional[str]): The name of the log likelihood file.
        log_likelihood_path (Optional[Path]): The path to the log likelihood file.
        sample_size (int): The sample size.
        seed (int): The seed value.
    """

    calibration_data_file_name: str
    calibration_data_path: Path
    log_likelihood_file_name: str | None
    log_likelihood_path: Path | None
    sample_size: int
    seed: int


class Model(BaseModel):
    """
    Represents the main model.

    Attributes
    ----------
        Applications (Applications): The applications.
        EDP (List[EDPItem]): The list of EDP items.
        FEM (Dict[str, str]): The FEM data.
        UQ (GP_AB_UQData): The UQ data.
        WorkflowType (str): The workflow type.
        correlationMatrix (List[float]): The correlation matrix.
        localAppDir (str): The local application directory.
        randomVariables (list[dict[str, Any]]): The list of random variables.
        remoteAppDir (str): The remote application directory.
        resultType (str): The result type.
        runDir (str): The run directory.
        runType (str): The run type.
        summary (List[str]): The summary.
        workingDir (str): The working directory.
    """

    Applications: Applications
    EDP: list[EDPItem]
    FEM: dict[str, str]  # Empty dict
    UQ: GP_AB_UQData
    WorkflowType: str
    correlationMatrix: list[float]  # noqa: N815
    localAppDir: str  # noqa: N815
    randomVariables: list[dict[str, Any]]  # noqa: N815
    remoteAppDir: str  # noqa: N815
    resultType: str  # noqa: N815
    runDir: str  # noqa: N815
    runType: str  # noqa: N815
    summary: list[str]
    workingDir: str  # noqa: N815

    @validator('correlationMatrix', each_item=True)
    def check_correlation_matrix(cls, v):  # noqa: N805
        """
        Validate each item in the correlation matrix.

        Args:
            v (float): The correlation value.

        Raises
        ------
            ValueError: If the correlation value is not between -1 and 1.

        Returns
        -------
            float: The validated correlation value.
        """
        if not (-1 <= v <= 1):
            error_message = 'Each correlation value must be between -1 and 1.'
            raise ValueError(error_message)
        return v
