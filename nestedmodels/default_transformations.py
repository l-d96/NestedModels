import pytensor.tensor as pt


def saturation_function(input_: pt.TensorVariable,
                        saturation: float) -> pt.TensorVariable:
    return 1 - pt.exp(-pt.true_div(input_, saturation))


def linear_function(input_: pt.TensorVariable,
                    coefficient: float) -> pt.TensorVariable:
    return pt.dot(input_, coefficient)


def linear_saturation(input_: pt.TensorVariable,
                      coefficient: float,
                      saturation: float) -> pt.TensorVariable:
    return linear_function(saturation_function(input_, saturation),
                           coefficient)


__all__ = [
    'saturation_function',
    'linear_function',
    'linear_saturation',
]
