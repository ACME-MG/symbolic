"""
 Title:         Basic model
 Description:   Performs the symbolic regression
 Author:        Janzen Choi

"""

# Libraries
import numpy as np
from symbolic.models.__model__ import __Model__
from pysr import PySRRegressor

# Model class
class Model(__Model__):

    def initialise(self, **settings):
        """
        Initialises the model
        """
        self.model = PySRRegressor(**settings, output_directory=self.output_path)

    def fit(self, input_data:np.array, output_data:np.array) -> None:
        """
        Performs the fittingg

        Parameters:
        * `input_data`:  Input data
        * `output_data`: Output data
        """
        self.model.fit(input_data, output_data)

    def predict(self, input_data:np.array) -> np.array:
        """
        Predicts data using fit

        Parameters:
        * `input_data`: Input data

        Returns the predicted data
        """
        self.model.predict(input_data)
