from __future__ import annotations  # noqa: INP001, D100

from typing import Any, Dict, List

from pydantic import BaseModel

from .distributions.UniformDTOs import DistributionDTO  # noqa: TCH001
from .modules.ModuleDTOs import ModuleDTO  # noqa: TCH001
from .sampling.mcmc.StretchDto import StretchDto  # noqa: TCH001


class ApplicationData(BaseModel):  # noqa: D101
    MS_Path: str
    mainScript: str  # noqa: N815
    postprocessScript: str  # noqa: N815


class FEM(BaseModel):  # noqa: D101
    Application: str
    ApplicationData: ApplicationData


class UQ(BaseModel):  # noqa: D101
    Application: str
    ApplicationData: Dict[str, Any]  # noqa: UP006


class Applications(BaseModel):  # noqa: D101
    FEM: FEM
    UQ: UQ


class EDPItem(BaseModel):  # noqa: D101
    length: int
    name: str
    type: str


class SubsetSimulationData(BaseModel):  # noqa: D101
    conditionalProbability: float  # noqa: N815
    failureThreshold: int  # noqa: N815
    maxLevels: int  # noqa: N815
    mcmcMethodData: StretchDto  # noqa: N815


class ReliabilityMethodData(BaseModel):  # noqa: D101
    method: str
    subsetSimulationData: SubsetSimulationData  # noqa: N815


class RandomVariable(BaseModel):  # noqa: D101
    distribution: str
    inputType: str  # noqa: N815
    lowerbound: int
    name: str
    refCount: int  # noqa: N815
    upperbound: int
    value: str
    variableClass: str  # noqa: N815


class Model(BaseModel):  # noqa: D101
    Applications: Applications
    EDP: List[EDPItem]  # noqa: UP006
    FEM: Dict[str, Any]  # noqa: UP006
    UQ: ModuleDTO
    # correlationMatrix: List[int]
    localAppDir: str  # noqa: N815
    randomVariables: List[DistributionDTO]  # noqa: N815, UP006
    remoteAppDir: str  # noqa: N815
    runType: str  # noqa: N815
    workingDir: str  # noqa: N815
