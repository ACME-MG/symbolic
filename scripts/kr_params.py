

# Libraries
import sys; sys.path += [".."]
from symbolic.helper.plotter import save_plot
from symbolic.helper.general import round_sf, transpose
from symbolic.io.files import csv_to_dict
import matplotlib.pyplot as plt
import numpy as np

# Constants
TICK_SIZE = 12
WIDTH     = 80
DATA_PATH = "data/kr_params.csv"

def main():
    """
    Main function
    """
    
    # Get parameters
    kr_dict = csv_to_dict(DATA_PATH)
    for key in ["A", "M"]:
        kr_dict[key] = [np.log10(float(p)) for p in kr_dict[key]]
    for key in ["n", "phi", "chi", "temp"]:
        kr_dict[key] = [float(p) for p in kr_dict[key]]

    # Reorganise parameters
    param_names = ["A", "n", "M", "phi", "chi"]
    param_values = [kr_dict[pn] for pn in param_names]
    param_dict = dict(zip(param_names, param_values))

    # # Fit parameters
    # fit_index = 0
    # fit_index_list = []
    # for temp in [800, 900, 1000]:
    #     index_list = [i for i in range(len(kr_dict["temp"])) if kr_dict["temp"][i] == temp]
    #     fit_index_list.append(index_list[fit_index])
    fit_param_values_list = [
        [np.log10(1.2912e-15), 6.9643, np.log10(1.0351e-16), 14.752, 6.1453],
        [np.log10(2.9854e-11), 5.6306, np.log10(2.8379e-13), 37.86, 5.0152],
        [np.log10(2.8203e-08), 4.451,  np.log10(1.6669e-09), 7.9495, 3.7298],
    ]
    fit_param_values_list = transpose(fit_param_values_list)

    # Plot parameter distributions
    for i, param in enumerate(["A", "n", "M", "phi", "chi"]):

        # Plot boxplot
        x_list = [800, 900, 1000]
        y_list_list = [[] for _ in range(len(x_list))]
        for j, temp in enumerate(kr_dict["temp"]):
            y_index = x_list.index(temp)
            y_list_list[y_index].append(kr_dict[param][j])
        plot_boxplots(x_list, y_list_list, "tab:cyan")
        
        # Fit the parameters
        fit_param_values = fit_param_values_list[i]
        plt.scatter(x_list, fit_param_values, marker="o", color="tab:red", s=8**2)
        polynomial = np.polyfit(x_list, fit_param_values, 2)
        x_fit = list(np.linspace(800, 1000, 100))
        y_fit = list(np.polyval(polynomial, x_fit))
        plt.plot(x_fit, y_fit, color="tab:green", linestyle="--", linewidth=2)
        pr = round_sf(list(polynomial), 5)
        print(f"{pr[0]}*x2^2 + {pr[1]}*x2 + {pr[2]}".replace("+ -", "- "))

        # Save
        save_plot(f"param_{param}.png")

def plot_boxplots(x_list:list, y_list_list:list, colour:str) -> None:
    """
    Plots several boxplots together

    Parameters:
    * `x_list`:      List of x labels
    * `y_list_list`: List of data lists
    * `colour`:      Boxplot colour
    """

    # Format plot
    plt.figure(figsize=(5,5), dpi=200)
    plt.gca().set_position([0.17, 0.12, 0.75, 0.75])
    plt.gca().grid(which="major", axis="both", color="SlateGray", linewidth=1, linestyle=":", alpha=0.5)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1)

    # Plot boxplots
    boxplots = plt.boxplot(y_list_list, positions=x_list, showfliers=False, patch_artist=True,
                           vert=True, widths=WIDTH, whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1))
    
    # Apply additional formatting to the boxplots
    for i in range(len(y_list_list)):
        patch = boxplots["boxes"][i]
        patch.set_facecolor(colour)
        patch.set_edgecolor("black")
        patch.set_linewidth(1)
        median = boxplots["medians"][i]
        median.set(color="black", linewidth=1)

# Calls the main function
if __name__ == "__main__":
    main()
