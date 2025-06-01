"""
 Title:         Basic model
 Description:   Performs the symbolic regression
 Author:        Janzen Choi

"""

# Libraries
from symbolic.models.__model__ import __Model__
from pysr import PySRRegressor

# Model class
class Model(__Model__):

    def initialise(self, **settings):
        """
        Initialises the model
        """
        self.regressor = PySRRegressor(
            maxsize              = 10,
            niterations          = 20,
            binary_operators     = ["+", "*"],
            unary_operators      = ["cos", "exp", "sin", "inv(x) = 1/x"],
            extra_sympy_mappings = {"inv": lambda x: 1 / x},
            elementwise_loss     = "loss(prediction, target) = (prediction - target)^2",
            output_directory     = self.output_path,
        )
        self.set_input_fields(["strain"])
        self.set_output_fields(["stress"])

    def fit(self, data_list:list) -> None:
        """
        Performs the fitting

        Parameters:
        * `data_list`: List of dictionaries containing data
        """
        input_data, output_data = self.data_list_to_arrays(data_list)
        self.regressor.fit(input_data, output_data)

    def predict(self, data_list:list) -> list:
        """
        Predicts data using fit

        Parameters:
        * `data_list`: List of dictionaries containing data

        Returns the list of predicted data
        """
        prd_dict_list = []
        for data in data_list:
            input_data, _ = self.data_list_to_arrays([data])
            output_data = self.regressor.predict(input_data)
            prd_dict = {
                "strain": [0] + [d[0] for d in input_data.tolist()],
                "stress": [0] + output_data.tolist()
            }
            prd_dict_list.append(prd_dict)
        return prd_dict_list

    def get_latex(self) -> str:
        """
        Returns the LaTeX equation of the final fit; must be overridden
        """
        latex_string = self.regressor.latex()
        latex_string = self.replace_variables(latex_string, [r'\epsilon'])
        return [latex_string]
