from .model import NestedModel
from sklearn.metrics import r2_score as _r2_score
from sklearn.metrics import mean_absolute_percentage_error as _mape
from sklearn.metrics import mean_squared_error as _mse
import numpy as np
import pandas as pd


def r2_score(model: NestedModel) -> pd.Series:
    if not model.is_compiled():
        raise Exception("model is not compiled.")
    y_true = model._model['target_variables'].eval()

    y_pred = np.zeros(y_true.shape)
    for index, v in enumerate(model._targets):
        pred = model.idata.posterior[f'{v}_mu'].data.reshape(
            -1, y_true.shape[0]).mean(0)
        y_pred[:, index] = pred

    rsquare = _r2_score(y_true, y_pred, multioutput='raw_values')

    return pd.Series(rsquare, index=model._targets, name='R2')


def mape(model: NestedModel):
    if not model.is_compiled():
        raise Exception("model is not compiled.")
    y_true = model._model['target_variables'].eval()

    y_pred = np.zeros(y_true.shape)
    for index, v in enumerate(model._targets):
        pred = model.idata.posterior[f'{v}_mu'].data.reshape(
            -1, y_true.shape[0]).mean(0)
        y_pred[:, index] = pred

    mapescore = _mape(y_true, y_pred, multioutput='raw_values')

    return pd.Series(mapescore, index=model._targets, name='MAPE')


def mse(model: NestedModel):
    if not model.is_compiled():
        raise Exception("model is not compiled.")
    y_true = model._model['target_variables'].eval()

    y_pred = np.zeros(y_true.shape)
    for index, v in enumerate(model._targets):
        pred = model.idata.posterior[f'{v}_mu'].data.reshape(
            -1, y_true.shape[0]).mean(0)
        y_pred[:, index] = pred

    msescore = _mse(y_true, y_pred, multioutput='raw_values')

    return pd.Series(msescore, index=model._targets, name='MAPE')


def loo(model: NestedModel):
    # if not model.is_compiled():
    #     raise Exception("model is not compiled.")
    raise NotImplementedError


def waic(model: NestedModel):
    # if not model.is_compiled():
    #     raise Exception("model is not compiled.")
    raise NotImplementedError
