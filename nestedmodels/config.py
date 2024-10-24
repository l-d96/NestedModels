from __future__ import annotations
from typing import Any, Callable
from typing_extensions import Self
from tomlkit import TOMLDocument, load
from pydantic import BaseModel, field_validator, model_validator
from pathlib import Path
from .utils import load_custom_transformations
import numpy as np
import json
import pytensor.tensor as pt
import pymc as pm
import inspect
import xarray

TRANSFORMATIONS = load_custom_transformations()


class Parameter(BaseModel):
    name: str
    distribution_name: str
    hyperparameters: dict[str, float | int]

    @field_validator('distribution_name')
    @classmethod
    def _validate_distribution(cls, distribution_name: str) -> str:
        if distribution_name not in pm.distributions.__all__:
            raise ValueError(
                f"'{distribution_name}' is not a valid distribution in PyMC")

        return distribution_name


class Transformation(BaseModel):
    function_name: str
    target_name: str
    parameters: list[Parameter]

    @field_validator('function_name')
    @classmethod
    def _validate_function_name(cls, function_name: str) -> str:
        if function_name not in TRANSFORMATIONS.keys():
            raise ValueError(
                f"'{function_name}' is not a valid transformation")

        return function_name

    @model_validator(mode='after')
    def _validate_transformation_args(self) -> Self:
        function: Callable = TRANSFORMATIONS[self.function_name]
        signature = inspect.signature(function)
        optional_args = [k for k, v in
                         signature.parameters.items()
                         if v.default is not inspect.Parameter.empty]
        mandatory_args = [k for k, v in
                          signature.parameters.items()
                          if v.default is inspect.Parameter.empty]
        all_args = optional_args + mandatory_args
        params = [p.name for p in self.parameters]

        # check we have all the mandatory args
        for k in mandatory_args:
            if k == 'input_':
                continue
            if k not in params:
                raise ValueError(
                    f"mandatory '{k}' parameter for function '{self.function_name}' is missing.")

        for p in params:
            if p not in all_args:
                raise ValueError(
                    f"parameter '{p}' is not an accepted argument for function '{self.function_name}'"
                )

        return self


class Node(BaseModel):
    name: str
    targets: list[Transformation]

    def track_node(self, model: pm.Model, input: pt.TensorVariable) -> None:
        with model:
            for t in self.targets:
                name = f"{self.name}_{t.target_name}_transformation"
                func: Callable = TRANSFORMATIONS[t.function_name]
                params = {}
                for param in t.parameters:
                    pymc_name = f"{self.name}_{t.target_name}_{param.name}"
                    distribution: pm.Distribution = getattr(
                        pm, param.distribution_name)
                    param_distribution = distribution(
                        name=pymc_name, **param.hyperparameters)
                    params[param.name] = param_distribution

                pm.Deterministic(name, func(input_=input, **params))

    def posterior_transformations(
            self,
            trace: xarray.DataArray,
            input: np.ndarray, data_dict: dict[str, pt.TensorVariable]):

        for t in self.targets:
            name = f"{self.name}_{t.target_name}_transformation"
            func: Callable = TRANSFORMATIONS[t.function_name]
            params = {}
            for param in t.parameters:
                pymc_name = f"{self.name}_{t.target_name}_{param.name}"
                param_posterior = trace.posterior[pymc_name] \
                    .data.reshape(-1, 1)
                params[param.name] = param_posterior

            data_dict[name] = func(input_=input, **params)


class Config(BaseModel):
    nodes: list[Node]

    def to_toml(self) -> TOMLDocument:
        pass

    def to_json(self) -> list | dict:
        pass

    def get_pointers(self, node_name: str) -> list[Node]:
        pointers = []
        for n in self.nodes:
            if n.name == node_name:
                continue
            for t in n.targets:
                if t.target_name == node_name:
                    pointers.append(n)
                    break
        return pointers

    @staticmethod
    def from_toml(config: TOMLDocument) -> Config:
        nodes = []

        # track leaf nodes
        all_nodes_found = set()

        for name, targets in config.items():
            node_name = name
            all_nodes_found.add(node_name)
            transformations = []

            if not hasattr(targets, 'items'):
                raise ConfigurationError(
                    f"variable '{name}' has no configuration")

            for target, target_params in targets.items():
                # track leaf nodes
                all_nodes_found.add(target)
                target_name = target
                transformation_name = target_params.get('transformation')
                transformation_params = target_params.get('parameters')
                parameters = []
                for param in transformation_params:
                    param_name = param.get('name')
                    distribution_params = param.get('distribution')
                    distribution_name = ''
                    hyperparameters = {}
                    if not hasattr(distribution_params, 'items'):
                        raise ConfigurationError(
                            f"parameter '{param_name}' has no configuration")
                    for attribute, val in distribution_params.items():
                        if attribute == 'name':
                            distribution_name = val
                            continue
                        hyperparameters[attribute] = val

                    parameter = Parameter(name=param_name,
                                          distribution_name=distribution_name,
                                          hyperparameters=hyperparameters)
                    parameters.append(parameter)

                transformation = Transformation(function_name=transformation_name,
                                                target_name=target_name,
                                                parameters=parameters)
                transformations.append(transformation)

            node = Node(name=node_name,
                        targets=transformations)
            nodes.append(node)

        # add leaf nodes to the configuration model
        for node_name in all_nodes_found:
            if node_name in [n.name for n in nodes]:
                continue
            new_node = Node(name=node_name, targets=[])
            nodes.append(new_node)

        return Config(nodes=nodes)

    @staticmethod
    def from_json(config: list[Any] | dict[str, Any]) -> Config:
        pass


class ConfigurationError(Exception):
    """
        Configuration has something wrong
    """


def read_toml(file: Path | str) -> Config:
    try:
        with open(file, "r") as f:
            toml = load(f)
            config = Config.from_toml(toml)
            return config
    except FileNotFoundError as e:
        print("error:", e)
        return


def read_json(file: Path | str) -> Config:
    try:
        with open(file, "r") as f:
            json_f = json.load(f)
            config = Config.from_json(json_f)
            return config
    except FileNotFoundError as e:
        print("error:", e)
        return
