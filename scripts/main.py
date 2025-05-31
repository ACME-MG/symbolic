"""
 Title:         Main
 Description:   Script to run symbolic regression
 Author:        Janzen Choi

"""

# Libraries
import sys; sys.path += [".."]
from symbolic.interface import Interface

def main():
    """
    Main function
    """

    # Initialise
    itf = Interface()
    itf.define_problem(
        input_fields  = ["strain"],
        output_fields = ["stress"]
    )
    
    # Add training data
    csv_path = "tensile/inl/AirBase_20_D5.csv"
    itf.add_data(csv_path)

    # Fit the data
    itf.fit_model(
        model_name           = "basic",
        maxsize              = 20,
        niterations          = 40,
        binary_operators     = ["+", "*"],
        unary_operators      = ["cos", "exp", "sin", "inv(x) = 1/x"],
        extra_sympy_mappings = {"inv": lambda x: 1 / x},
        elementwise_loss     = "loss(prediction, target) = (prediction - target)^2",
    )

    # Plots the results
    itf.plot_fit(
        x_field = "strain",
        y_field = "stress",
        x_units = "mm/mm",
        y_units = "MPa",
    )

# Calls the main function
if __name__ == "__main__":
    main()
