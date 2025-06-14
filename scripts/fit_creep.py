"""
 Title:         Main
 Description:   Script to run symbolic regression
 Author:        Janzen Choi

"""

# Libraries
import sys; sys.path += [".."]
from symbolic.interface import Interface

# Data list
CAL_LIST = [
    # "AirBase_800_60_G32.csv", #
    # "AirBase_800_60_G47.csv",
    # "AirBase_800_65_G33.csv", #
    # "AirBase_800_65_G43.csv",
    # "AirBase_800_70_G24.csv",
    # "AirBase_800_70_G44.csv", # #
    # "AirBase_800_80_G25.csv", # #
    # "AirBase_800_80_G34.csv",
    # "AirBase_900_26_G42.csv",
    # "AirBase_900_26_G59.csv", #
    # "AirBase_900_28_G40.csv",
    # "AirBase_900_28_G45.csv", #
    # "AirBase_900_31_G21.csv",
    # "AirBase_900_31_G50.csv", # #
    # "AirBase_900_36_G19.csv",
    # "AirBase_900_36_G22.csv", # #
    # "AirBase_900_36_G63.csv",
    # "AirBase_1000_11_G39.csv", #
    # "AirBase_1000_12_G48.csv", #
    # "AirBase_1000_12_G52.csv",
    "AirBase_1000_13_G30.csv", # #
    # "AirBase_1000_13_G51.csv",
    "AirBase_1000_16_G18.csv", # #
    # "AirBase_1000_16_G38.csv",
]
VAL_LIST = [
    # "AirBase_800_60_G32.csv", #
    # "AirBase_800_65_G33.csv", #
    # "AirBase_900_26_G59.csv", #
    # "AirBase_900_28_G45.csv", #
    "AirBase_1000_11_G39.csv", #
    "AirBase_1000_12_G48.csv", #
]

def main():
    """
    Main function
    """

    # Initialise
    itf = Interface("creep_1000C")
    itf.define_model("creep")

    # Add fitting data
    for file in CAL_LIST:
        itf.add_data(f"creep/inl_1/{file}", fitting=True)
        itf.sparsen_data(100)
        itf.set_weights([2, 1, 0.5, 0.5, 1, 2])

    # Add validation data
    for file in VAL_LIST:
        itf.add_data(f"creep/inl_1/{file}", fitting=False)
        itf.sparsen_data(100)

    # Fit the data
    itf.fit_model()

    # Save the results
    for temperature in [800, 900, 1000]:
        itf.plot_fit(
            x_field    = "time",
            y_field    = "strain",
            x_scale    = 1/3600,
            x_units    = "h",
            y_units    = "mm/mm",
            file_name  = f"plot_fit_{temperature}",
            conditions = {"temperature": temperature},
        )
    itf.plot_equation()

    # Analyse results
    strains = lambda data_dict : [max(data_dict["strain"])*(i+1)/10 for i in range(10)]
    itf.plot_1to1(strains, r"$\epsilon$", "mm/mm")
    time_failure = lambda data_dict : [max(data_dict["time"])]
    itf.plot_1to1(time_failure, r"$t_f$", "mm/mm")

# Calls the main function
if __name__ == "__main__":
    main()
