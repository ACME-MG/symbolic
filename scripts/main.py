"""
 Title:         Main
 Description:   Script to run symbolic reggression
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

# Calls the main function
if __name__ == "__main__":
    main()
