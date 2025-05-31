from pysr import PySRRegressor
import numpy as np
import sys; sys.path += [".."]
from symbolic.helper.io import csv_to_dict

X = 2 * np.random.randn(100, 5)
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 2


default_pysr_params = dict(
    populations=30,
    model_selection="best",
)

model = PySRRegressor(
    maxsize=20,
    niterations=40,  # < Increase me for better results
    binary_operators=["+", "*"],
    unary_operators=[
        "cos",
        "exp",
        "sin",
        "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
)

print(type(X))
print(X.shape)
print(y.shape)

model.fit(X, y)
print(model)
