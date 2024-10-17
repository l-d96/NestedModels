from nestedmodels.config import (Config,
                                 Node,
                                 Transformation,
                                 Parameter)
from nestedmodels import read_toml
from tomlkit import load
import math


def saturation_function(x, saturation):
    return 1-math.exp(-x/saturation)


configuration = read_toml("./tests/data/example.toml")
print(configuration.get_pointers('Preference'))

# with open("../tests/data/example.toml") as f:
#     configuration_toml = load(f)

# # for name, targets in configuration.items():
# targets = configuration_toml.get('Spend')
# params = targets.get('Awareness')
# # # for target, params in targets.items():
# parameters = params.get('parameters')
# for el in parameters:
#     print(el)
# print(target)
# print(targets)

# example = [
#     Node(name="Spend",
#          targets=Transformation(
#              function=saturation_function,
#              target_name="Awareness",
#              parameters=
#          ))
# ]
# print(Config.from_toml(configuration))
