"""
 Title:         Main
 Description:   Script to run symbolic regression
 Author:        Janzen Choi

"""

# Libraries
import sys; sys.path += [".."]
from symbolic.interface import Interface
from symbolic.helper.derivative import differentiate_curve
from symbolic.helper.interpolator import intervaluate

# Constants
ALL_CAL = False

def main():
    """
    Main function
    """

    # Initialise
    itf = Interface("custom")
    itf.define_model("custom")

    # 800C
    itf.add_data("creep/inl_1/AirBase_800_70_G44.csv", fitting=True)
    # itf.add_data("creep/inl_1/AirBase_800_80_G25.csv", fitting=True)
    # itf.add_data("creep/inl_1/AirBase_800_60_G32.csv", fitting=ALL_CAL)
    # itf.add_data("creep/inl_1/AirBase_800_65_G33.csv", fitting=ALL_CAL)

    # # 900C
    # itf.add_data("creep/inl_1/AirBase_900_31_G50.csv", fitting=True)
    # itf.add_data("creep/inl_1/AirBase_900_36_G22.csv", fitting=True)
    # itf.add_data("creep/inl_1/AirBase_900_26_G59.csv", fitting=ALL_CAL)
    # itf.remove_oxidation()
    # itf.add_data("creep/inl_1/AirBase_900_28_G45.csv", fitting=ALL_CAL)
    
    # # !000C
    # itf.add_data("creep/inl_1/AirBase_1000_13_G30.csv", fitting=True)
    # itf.remove_oxidation(0.1, 0.7)
    # itf.add_data("creep/inl_1/AirBase_1000_16_G18.csv", fitting=True)
    # itf.remove_oxidation()
    # itf.add_data("creep/inl_1/AirBase_1000_11_G39.csv", fitting=ALL_CAL)
    # itf.remove_oxidation(0.1, 0.7)
    # itf.add_data("creep/inl_1/AirBase_1000_12_G48.csv", fitting=ALL_CAL)
    # itf.remove_oxidation()

    # Prepare the data
    # time_h = lambda dd : dd.update({"time": [round(t/3600, 1) for t in dd["time"]]}) or dd
    # itf.change_field(time_h)
    # time_n = lambda dd : dd.update({"time": [round(t/10000/3600, 1) for t in dd["time"]]}) or dd
    # itf.change_field(time_n)
    # stress_n = lambda dd : dd.update({"stress": dd["stress"]/80}) or dd
    # itf.change_field(stress_n)
    # strain_p = lambda dd : dd.update({"strain": [s*100 for s in dd["strain"]]}) or dd
    # itf.change_field(strain_p)

    # Fit the model
    itf.fit_model()

    # Run model-specific functions
    # itf.run_model_function("set_julia", julia="f0 = 0; f1 = 0")
    # itf.run_model_function("set_julia", julia="f0 = log(#2) * (((((#1 + #3) / ((#3 * #2) * #2)) ^ 1.4776893550074655) * (-68.05796486172522 + #2)) / #2); f1 = (#1 * (#2 / ((#1 / #2) + -0.07971549275158935))) / 0.7217399144268667")
    itf.run_model_function("export_errors", data_list=itf.__controller__.get_data_list())

    # Save the results
    for temperature in [800, 900, 1000]:
        itf.plot_fit(
            x_field    = "time",
            y_field    = "strain",
            x_scale    = 1/3600,
            x_units    = "h",
            y_units    = "mm/mm",
            # y_units    = "%",
            y_limits   = (0, 1.0),
            file_name  = f"plot_fit_{temperature}",
            conditions = {"temperature": temperature},
        )
    itf.plot_equation()

    # Analyse strain predictions
    def strains(fd:dict, sd:dict) -> tuple:
        cttf = min([max(fd["time"]), max(sd["time"])])
        t_list = [(i+1)*cttf/10 for i in range(10)]
        fd_strains = intervaluate(fd["time"], fd["strain"], t_list)
        sd_strains = intervaluate(sd["time"], sd["strain"], t_list)
        return fd_strains, sd_strains
    itf.plot_1to1(strains, r"$\epsilon$", "mm/mm", file_name="1to1_strains")
    
    # Analyse time-to-failure predictions
    ttf = lambda fd, sd : ([max(fd["time"])], [max(sd["time"])])
    itf.plot_1to1(ttf, r"${t}_{f}$", "mm/mm", file_name="1to1_ttf")
    
    # Analyse strain-to-failure predictions
    stf = lambda fd, sd : ([max(fd["strain"])], [max(sd["strain"])])
    itf.plot_1to1(stf, r"${\epsilon}_{f}$", "mm/mm", file_name="1to1_stf")
    
    # Analyse minimum creep rate predictions
    get_mcr = lambda dd : min(differentiate_curve(dd, "time", "strain")["strain"])
    mcr = lambda fd, sd : ([get_mcr(fd)], [get_mcr(sd)])
    itf.plot_1to1(mcr, r"$\dot{\epsilon}_{min}$", "1/s", file_name="1to1_mcr")

# Calls the main function
if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for i in range(iters):
        main()
