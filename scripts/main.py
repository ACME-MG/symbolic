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
    itf.define_model("basic")

    # Add training data
    itf.add_data("tensile/inl/AirBase_20_D5.csv")
    itf.sparsen_data(100)

    # Fit the data
    itf.fit_model()

    # Save the results
    itf.plot_fit(
        x_field = "strain",
        y_field = "stress",
        x_units = "mm/mm",
        y_units = "MPa",
    )
    itf.plot_equation()

# Calls the main function
if __name__ == "__main__":
    main()
