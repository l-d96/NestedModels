import pytensor.tensor as pt
import pytensor
import numpy as np


def saturation_function(input_: pt.TensorVariable,
                        saturation: float) -> pt.TensorVariable:
    return 1 - pt.exp(-pt.true_div(input_, saturation))


def linear_function(input_: pt.TensorVariable,
                    coefficient: float) -> pt.TensorVariable:
    return input_ * coefficient


def hill_function(input_: pt.TensorVariable,
                  half_life: float,
                  saturation: float) -> pt.TensorVariable:
    return pt.true_div(1, (1 + pt.power(pt.true_div(input_, half_life), -1 * saturation)))


def linear_saturation(input_: pt.TensorVariable,
                      coefficient: float,
                      saturation: float) -> pt.TensorVariable:
    return linear_function(saturation_function(input_, saturation),
                           coefficient)


def linear_hill(input_: pt.TensorVariable,
                coefficient: float,
                half_life: float,
                saturation: float) -> pt.TensorVariable:
    return linear_function(hill_function(input_, half_life, saturation),
                           coefficient)


def decay_tensor(input_: pt.TensorVariable,
                 decay_param: pt.TensorVariable) -> pt.TensorVariable:
    def inner_decay(input_row, cumulative_decay, decay_value):
        return input_row + (1-decay_value) * cumulative_decay

    output, update = pytensor.scan(fn=inner_decay,
                                   outputs_info=[input_[0]],
                                   sequences=[input_[1:]],
                                   non_sequences=[decay_param])
    final_output = pt.concatenate([[input_[0]], output])

    return final_output


def decay_vector(input_: np.ndarray,
                 decay_param: np.ndarray | float) -> np.ndarray:

    def inner_decay(input_row, cumulative_decay, decay_value):
        return input_row + (1-decay_value) * cumulative_decay

    input_ = input_.reshape(-1, 1)
    if isinstance(decay_param, float):
        outputs_info = input_[0]
    else:
        outputs_info = pt.tile(input_[0], (decay_param.shape[0], 1))
    output, update = pytensor.scan(fn=inner_decay,
                                   outputs_info=[outputs_info],
                                   sequences=[input_[1:]],
                                   non_sequences=[decay_param])
    final_output = pt.concatenate([[outputs_info], output])

    return final_output[:, :, 0].T


def decay(input_: pt.TensorVariable | np.ndarray,
          decay_param: pt.TensorVariable | float | np.ndarray) -> pt.TensorVariable | np.ndarray:

    if isinstance(input_, pt.TensorVariable) or isinstance(decay_param, pt.TensorVariable):
        return decay_tensor(input_, decay_param)

    return decay_vector(input_, decay_param)


__all__ = [
    'saturation_function',
    'linear_function',
    'linear_saturation',
    'hill_function',
    'linear_hill',
    'decay'
]
