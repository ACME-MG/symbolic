"""
 Title:         Basic model
 Description:   Performs the symbolic regression
 Author:        Janzen Choi

"""

# Libraries
from symbolic.models.__model__ import __Model__, replace_variables, equate_to
from pysr import PySRRegressor
# from sympy import Piecewise

# Model class
class Model(__Model__):

    def initialise(self, **settings):
        """
        Initialises the model
        """
        self.regressor = PySRRegressor(
            populations          = 32,
            population_size      = 32,
            maxsize              = 32,
            niterations          = 32,
            binary_operators     = ["+", "*", "^", "/"],
            constraints          = {"^": (-1, 1)},
            unary_operators      = ["log", "exp"],
            # unary_operators      = ["log", "st(x) = x >= 1 ? one(x) : zero(x)"],
            # unary_operators      = ["cos", "sin", "exp", "log", "ninv(x) = 1/(1-x)"],
            # extra_sympy_mappings = {
                # "ninv": lambda x : 1/(1-x),
                # "st":   lambda x: Piecewise((1, x >= 1), (0, True)),
            # },
            elementwise_loss     = "loss(prediction, target, weight) = weight*(prediction - target)^2",
            output_directory     = self.output_path,
        )
        self.set_input_fields(["time", "stress", "temperature"])
        self.set_output_fields(["strain"])

    def fit(self, data_list:list) -> None:
        """
        Performs the fitting

        Parameters:
        * `data_list`: List of dictionaries containing data
        """
        input_data = self.get_input(data_list)
        output_data = self.get_output(data_list)
        weights = self.get_fit_weights(data_list)
        self.regressor.fit(input_data, output_data, weights=weights)

    def predict(self, data_list:list) -> list:
        """
        Predicts data using fit

        Parameters:
        * `data_list`: List of dictionaries containing data

        Returns the list of predicted data
        """
        prd_dict_list = []
        for data in data_list:
            input_data = self.get_input([data])
            output_data = self.regressor.predict(input_data)
            prd_dict = {
                "time": [0] + [d[0] for d in input_data.tolist()],
                "strain": [0] + output_data.tolist()
            }
            prd_dict_list.append(prd_dict)
        return prd_dict_list

    def get_latex(self) -> str:
        """
        Returns the LaTeX equation of the final fit; must be overridden
        """
        latex_string = self.regressor.latex()
        latex_string = replace_variables(latex_string, [r't', r'\sigma', r'T'])
        latex_string = equate_to(r'\epsilon', latex_string)
        return [latex_string]
