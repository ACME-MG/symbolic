"""
 Title:         Basic model
 Description:   Performs the symbolic regression
 Author:        Janzen Choi

"""

# Libraries
from symbolic.models.__model__ import __Model__, convert_data, replace_variables, equate_to, sparsen_data
import numpy as np
from pysr import PySRRegressor, TemplateExpressionSpec
from copy import deepcopy

# Model class
class Model(__Model__):

    def initialise(self, **settings):
        """
        Initialises the model
        """

        # Define regressor for time-to-failure
        self.ttf_reg = PySRRegressor(
            populations      = 32,
            population_size  = 32,
            maxsize          = 32,
            niterations      = 64,
            binary_operators = ["*", "^", "/"],
            constraints      = {"^": (-1, 1)},
            elementwise_loss = "loss(prediction, target) = (prediction - target)^2",
            output_directory = self.output_path,
        )

        # Define regressor for strain
        self.strain_reg = PySRRegressor(
            populations      = 32,
            population_size  = 32,
            maxsize          = 32,
            niterations      = 256,
            binary_operators = ["+", "*", "^", "/"],
            constraints      = {"^": (-1, 1)},
            unary_operators  = ["log", "exp"],
            elementwise_loss = "loss(prediction, target, weight) = weight*(prediction - target)^2",
            output_directory = self.output_path,
        )
        self.set_fields(["time", "stress", "temperature", "strain"])

    def fit(self, data_list:list) -> None:
        """
        Performs the fitting

        Parameters:
        * `data_list`: List of dictionaries containing data
        """

        # Sparsen data
        data_list = [sparsen_data(data, 100) for data in data_list]

        # Fit the time-to-failure regression model
        input_data = convert_data(data_list, ["stress", "temperature"])
        ttf_list = [max(data.get_data("time")) for data in data_list]
        output_data = np.array([np.array([ttf]) for ttf in ttf_list])
        self.ttf_reg.fit(input_data, output_data)

        # Fit the strain regression model
        input_data = convert_data(data_list, ["time", "stress", "temperature"])
        output_data = convert_data(data_list, ["strain"])
        weights = self.get_fit_weights(data_list)
        self.strain_reg.fit(input_data, output_data, weights=weights)

    def predict(self, data_list:list) -> list:
        """
        Predicts data using fit

        Parameters:
        * `data_list`: List of dictionaries containing data

        Returns the list of predicted data
        """

        # Sparsen data
        data_list = [sparsen_data(data, 100) for data in data_list]
        
        # Iterate through data
        prd_dict_list = []
        for data in data_list:

            # Predict time-to-failure
            input_data = convert_data([data], ["stress", "temperature"])
            time_failure = self.ttf_reg.predict(input_data)[0]
            
            # Predict creep strain curve
            time_list = np.linspace(0, time_failure, len(data.get_data("time"))).tolist()
            ttf_data = deepcopy(data)
            ttf_data.set_data("time", time_list)
            input_ttf_data = convert_data([ttf_data], ["time", "stress", "temperature"])
            output_ttf_data = self.strain_reg.predict(input_ttf_data)

            # Combine and append
            prd_dict = {
                "time":   [0] + [d[0] for d in input_ttf_data.tolist()],
                "strain": [0] + output_ttf_data.tolist()
            }
            prd_dict_list.append(prd_dict)
        
        # Return predictions
        return prd_dict_list

    def get_latex(self) -> str:
        """
        Returns the LaTeX equation of the final fit; must be overridden
        """

        # Get time-to-failure latex equation
        ttf_reg_ls = self.ttf_reg.latex()
        ttf_reg_ls = replace_variables(ttf_reg_ls, [r'\sigma', r'T'])
        ttf_reg_ls = equate_to(r't_f', ttf_reg_ls)

        # Get strain latex equation
        strain_reg_ls = self.strain_reg.latex()
        strain_reg_ls = replace_variables(strain_reg_ls, [r't', r'\sigma', r'T'])
        strain_reg_ls = equate_to(r'\epsilon', strain_reg_ls)
        
        # Combine and return
        return [ttf_reg_ls, strain_reg_ls]
