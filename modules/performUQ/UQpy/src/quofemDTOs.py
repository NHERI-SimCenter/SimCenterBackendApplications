from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel

from .distributions.UniformDTOs import DistributionDTO
from .modules.ModuleDTOs import ModuleDTO
from .sampling.mcmc.StretchDto import StretchDto


class ApplicationData(BaseModel):
    MS_Path: str
    mainScript: str
    postprocessScript: str


class FEM(BaseModel):
    Application: str
    ApplicationData: ApplicationData


class UQ(BaseModel):
    Application: str
    ApplicationData: Dict[str, Any]


class Applications(BaseModel):
    FEM: FEM
    UQ: UQ


class EDPItem(BaseModel):
    length: int
    name: str
    type: str


class SubsetSimulationData(BaseModel):
    conditionalProbability: float
    failureThreshold: int
    maxLevels: int
    mcmcMethodData: StretchDto


class ReliabilityMethodData(BaseModel):
    method: str
    subsetSimulationData: SubsetSimulationData


class RandomVariable(BaseModel):
    distribution: str
    inputType: str
    lowerbound: int
    name: str
    refCount: int
    upperbound: int
    value: str
    variableClass: str


class Model(BaseModel):
    Applications: Applications
    EDP: List[EDPItem]
    FEM: Dict[str, Any]
    UQ: ModuleDTO
    # correlationMatrix: List[int]
    localAppDir: str
    randomVariables: List[DistributionDTO]
    remoteAppDir: str
    runType: str
    workingDir: str
