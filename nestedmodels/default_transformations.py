import pytensor.tensor as pt
import pytensor


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


def decay(input_: pt.TensorVariable,
          decay_param: float) -> pt.TensorVariable:

    def inner_decay(input_row, cumulative_decay, decay_value):
        return input_row + (1-decay_value) * cumulative_decay

    output, update = pytensor.scan(fn=inner_decay,
                                   outputs_info=[input_[0, ...]],
                                   sequences=[input_[1:, ...]],
                                   non_sequences=[decay_param])
    final_output = pt.concatenate([[input_[0, ...]], output])

    return final_output


__all__ = [
    'saturation_function',
    'linear_function',
    'linear_saturation',
    'hill_function',
    'linear_hill',
    'decay'
]
