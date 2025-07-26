"""
 Title:         Main
 Description:   Script to run symbolic regression
 Author:        Janzen Choi

"""

# Libraries
import sys; sys.path += [".."]
from symbolic.interface import Interface
from symbolic.helper.derivative import differentiate_curve

# Constants
WEIGHTS = [2, 1, 0.5, 0.5, 1, 2]

def main():
    """
    Main function
    """

    # Initialise
    itf = Interface("creep")
    itf.define_model("creep")

    # Add fitting data
    itf.add_data("creep/inl_1/AirBase_800_70_G44.csv",  fitting=True, weights=WEIGHTS)
    itf.add_data("creep/inl_1/AirBase_800_80_G25.csv",  fitting=True, weights=WEIGHTS)
    # itf.add_data("creep/inl_1/AirBase_900_31_G50.csv",  fitting=True, weights=WEIGHTS)
    # itf.add_data("creep/inl_1/AirBase_900_36_G22.csv",  fitting=True, weights=WEIGHTS)
    # itf.add_data("creep/inl_1/AirBase_1000_13_G30.csv", fitting=True, weights=WEIGHTS)
    # itf.remove_oxidation(0.1, 0.7)
    # itf.add_data("creep/inl_1/AirBase_1000_16_G18.csv", fitting=True, weights=WEIGHTS)
    # itf.remove_oxidation()

    # Add validation data
    itf.add_data("creep/inl_1/AirBase_800_60_G32.csv",  fitting=False, weights=WEIGHTS)
    itf.add_data("creep/inl_1/AirBase_800_65_G33.csv",  fitting=False, weights=WEIGHTS)
    # itf.add_data("creep/inl_1/AirBase_900_26_G59.csv",  fitting=False, weights=WEIGHTS)
    # itf.remove_oxidation()
    # itf.add_data("creep/inl_1/AirBase_900_28_G45.csv",  fitting=False, weights=WEIGHTS)
    # itf.add_data("creep/inl_1/AirBase_1000_11_G39.csv", fitting=False, weights=WEIGHTS)
    # itf.remove_oxidation(0.1, 0.7)
    # itf.add_data("creep/inl_1/AirBase_1000_12_G48.csv", fitting=False, weights=WEIGHTS)
    # itf.remove_oxidation()

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
    ttf = lambda data_dict : [max(data_dict["time"])]
    itf.plot_1to1(ttf, r"${t}_{f}$", "mm/mm")
    mcr = lambda data_dict : [min(differentiate_curve(data_dict, "time", "strain")["strain"])]
    itf.plot_1to1(mcr, r"$\dot{\epsilon}_{min}$", "mm/mm")

# Calls the main function
if __name__ == "__main__":
    main()
