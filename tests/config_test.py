from nestedmodels import read_toml, NestedModel
import pytensor.tensor as pt
import pandas as pd
import numpy as np


X = pd.DataFrame({'Spend': np.random.normal(0, 1, (100,))})
y = pd.DataFrame({
                 'Awareness': np.random.normal(0, 1, (10,)),
                 'Consideration': np.random.normal(0, 1, (10,)),
                 'Preference': np.random.normal(0, 1, (10,)),
                 })

configuration = read_toml("./tests/data/example.toml")
model = NestedModel(configuration)
model.fit(X, y)
mus = model.predict(X)
# v: pt.variable.TensorVariable
# for v in model._model.unobserved_RVs:
#     if v.name == "Preference_mu":
#         v.dprint()
# print(model._model.unobserved_RVs)
model.to_graphviz('/home/ld/Desktop/Projects/NestedModels/tests/test.png')
# print(model.idata)

# print(configuration)
# print([n.name for n in configuration.nodes])

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
