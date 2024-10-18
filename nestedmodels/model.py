from .config import Config, Node
from typing import Optional
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pathlib import Path


class NestedModel:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._model: Optional[pm.Model] = None

    def compile(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        n_obs_x = X.shape[0]
        n_obs_y = y.shape[0]
        if n_obs_x != n_obs_y:
            raise Exception(
                "input and target have different number of observations")

        self._model = pm.Model()

        # track data
        with self._model:
            for c in X.columns:
                pm.Data(name=f"{c}_X", value=X[c], coords={
                        'variables': c}, dims='obs')

            target_variables = y.columns
            observed_data = pm.Data(name='target_variables', value=y[target_variables],
                                    coords={'variables': target_variables},
                                    dims=('obs', 'variables'))

        self._compile_model(self._config.nodes, [])
        mus = self._get_mu_variables()

        mu_all = pt.stack([mus[c] for c in target_variables], axis=1)

        with self._model:
            sigma = pm.HalfCauchy('sigma', 5)
            target = pm.Normal('target', mu=mu_all, sigma=sigma,
                               observed=observed_data,
                               shape=(n_obs_y, len(target_variables)),
                               dims=('obs', 'variables'))
    def _compile_model(self, nodes: list[Node], explored: list[Node]) -> None:
        if not nodes:
            return
        for n in nodes:
            if n in explored:
                continue
            explored.append(n)
            prior_nodes = self._config.get_pointers(n.name)

            self._compile_model(prior_nodes, explored)

            if prior_nodes:
                with self._model:
                    intercept = pm.HalfNormal(f"{n.name}_intercept")
                    input = pm.Deterministic(f"{n.name}_mu", sum(
                        [intercept] + self._get_parent_variables(n.name)), dims='obs')

                n.track_node(self._model, input)
            else:
                input = self._get_data_input(n.name)
                n.track_node(self._model, input)

    def _get_data_input(self, name: str) -> pt.sharedvar.TensorSharedVariable:
        if self._model is None:
            raise Exception("model hasn't been compiled yet")

        shared_data = self._model.data_vars
        for sd in shared_data:
            if sd.name == f"{name}_X":
                return sd
        raise Exception(f"Missing input data for variable '{name}'")

    def _get_parent_variables(self, name: str) -> list[pt.TensorVariable]:
        if self._model is None:
            raise Exception("model hasn't been compiled yet")

        variables = self._model.unobserved_RVs
        parents = []
        for v in variables:
            if v.name.endswith(f"_{name}_transformation"):
                parents.append(v)

        return parents

    def _get_mu_variables(self) -> dict[str, pt.TensorVariable]:
        if self._model is None:
            raise Exception("model hasn't been compiled yet")

        variables = self._model.unobserved_RVs
        mus = {}
        for v in variables:
            if v.name.endswith('_mu'):
                name = v.name.removesuffix('_mu')
                mus[name] = v

        return mus
class CyclicalGraphError(Exception):
    """
        Cycles are not allowed in DAG
    """
