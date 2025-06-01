"""
 Title:         Controller
 Description:   Directs the symbolic regression procedure
 Author:        Janzen Choi

"""

# Libraries
import matplotlib.pyplot as plt
from symbolic.reader.dataset import Dataset
from symbolic.helper.general import get_thinned_list, flatten
from symbolic.helper.plotter import prep_plot, set_limits, add_legend, save_plot, save_latex, plot_1to1
from symbolic.helper.plotter import EXP_COLOUR, CAL_COLOUR, VAL_COLOUR
from symbolic.models.__model__ import get_model

# Controller class
class Controller:

    def __init__(self, output_path:str):
        """
        Constructor for the controller class

        Parameters:
        * `output_path`:   Path to the output directory
        """
        self.output_path = output_path
        self.model = None
        self.data_list = []
    
    def define_model(self, model_name:str, **kwargs) -> None:
        """
        Defines the model

        Parameters:
        * `model_name`: Name of the model
        """
        self.model = get_model(model_name, self.output_path, **kwargs)
        self.input_fields = self.model.get_input_fields()
        self.output_fields = self.model.get_output_fields()
        self.all_fields = self.input_fields + self.output_fields

    def add_data(self, csv_path:str, fitting:bool=True) -> None:
        """
        Adds fitting data

        Parameters:
        * `csv_path`: Path to the csv file containing the data
        * `fitting`:  Whether the data will be used for fitting
        """
        data = Dataset(csv_path, self.all_fields, fitting)
        self.data_list.append(data)

    def sparsen_data(self, new_size:int=100) -> None:
        """
        Sparsen the most recently added dataset

        Parameters:
        * `new_size`: New size to sparsen the data to
        """
        last_data = self.get_last_data()
        last_data_dict = last_data.get_data_dict()
        for field in last_data_dict.keys():
            if isinstance(last_data_dict[field], list):
                last_data_dict[field] = get_thinned_list(last_data_dict[field], new_size)
        last_data.set_data_dict(last_data_dict)
        self.set_last_data(last_data)

    def fit_model(self):
        """
        Fits the symbolic regression model
        """
        fit_data_list = self.get_fit_data_list()
        print()
        self.model.fit(fit_data_list)
        print()

    def plot_fit(self, plot_path:str, x_field:str, y_field:str, x_units:str="",
                 y_units:str="", x_limits:tuple=None, y_limits:tuple=None, **settings) -> None:
        """
        Plots the results

        Parameters:
        * `plot_path`: Path to save the plot
        * `x_field`:   Field to use for the x-axis
        * `y_field`:   Field to use for the y-axis
        * `x_units`:   Units for the x-axis
        * `y_units`:   Units for the y-axis
        * `x_limits`:  Limits to apply on the x-axis
        * `y_limits`:  Limits to apply on the y-axis
        """

        # Check if model has been defined
        if self.model == None:
            raise ValueError("The model has not been defined!")

        # Initialise
        prep_plot(
            x_label = x_field,
            y_label = y_field,
            x_units = x_units,
            y_units = y_units,
        )

        # Plot experimental data
        data_dict_list = self.get_data_dict_list()
        for data_dict in data_dict_list:
            plt.scatter(data_dict[x_field], data_dict[y_field], color=EXP_COLOUR, s=8**2)

        # Plot fitted data
        fit_data_list = self.get_fit_data_list()
        fit_dict_list = self.model.predict(fit_data_list)
        for fit_dict in fit_dict_list:
            plt.plot(fit_dict[x_field], fit_dict[y_field], color=CAL_COLOUR, linewidth=3)

        # Plot predicted data
        prd_data_list = self.get_prd_data_list()
        prd_dict_list = self.model.predict(prd_data_list)
        for prd_dict in prd_dict_list:
            plt.plot(prd_dict[x_field], prd_dict[y_field], color=VAL_COLOUR, linewidth=3)

        # Format and save
        set_limits(x_limits, y_limits)
        add_legend(
            calibration = len(fit_data_list) > 0,
            validation  = len(prd_data_list) > 0,
        )
        save_plot(plot_path, **settings)

    def plot_1to1(self, plot_path:str, handle, label:str="", units:str="", limits:tuple=None) -> None:
        """
        Plots 1:1 comparison plots based on a function handle;
        the function must take a dictionary as an argument and
        return a list of values

        Parameters:
        * `plot_path`: Path to save the plot
        * `handle`:    The function handle
        * `label`:     Label to represent values
        * `units`:     Units to place beside label
        * `limits`:    Limits of the plot
        """

        # Check if model has been defined
        if self.model == None:
            raise ValueError("The model has not been defined!")
        
        # Get all data
        fit_data_list = self.get_fit_data_list()
        fit_dict_list = self.model.predict(fit_data_list)
        prd_data_list = self.get_prd_data_list()
        prd_dict_list = self.model.predict(prd_data_list)

        # Apply function handle
        exp_cal_list = flatten([handle(fd.get_data_dict()) for fd in fit_data_list])
        exp_val_list = flatten([handle(pd.get_data_dict()) for pd in prd_data_list])
        sim_cal_list = flatten([handle(fd) for fd in fit_dict_list])
        sim_val_list = flatten([handle(pd) for pd in prd_dict_list])

        # Plot and save
        plot_1to1(exp_cal_list, exp_val_list, sim_cal_list, sim_val_list, label, units, limits)
        save_plot(plot_path)

    def plot_equation(self, equation_path:str) -> None:
        """
        Saves an image of the equation
        """
        latex_equations = self.model.get_latex()
        if not isinstance(latex_equations, list):
            latex_equations = [latex_equations]
        save_latex(equation_path, latex_equations)

    def get_last_data(self) -> dict:
        """
        Returns the previously added dataset object
        """
        if len(self.data_list) == 0:
            raise ValueError("No datasets have been added!")
        return self.data_list[-1]

    def set_last_data(self, last_data) -> None:
        """
        Redefines the previously added dataset object

        Parameters:
        * `last_data`: The new last dataset object
        """
        if len(self.data_list) == 0:
            raise ValueError("No datasets have been added!")
        self.data_list[-1] = last_data

    def get_data_dict_list(self) -> list:
        """
        Gets the data dictionaries only
        """
        return [data.get_data_dict() for data in self.data_list]

    def get_num_data(self) -> int:
        """
        Returns the number of datasets added
        """
        return len(self.data_list)

    def get_fit_data_list(self) -> list:
        """
        Returns the list of fitting datasets
        """
        return [data for data in self.data_list if data.is_fitting()]

    def get_prd_data_list(self) -> list:
        """
        Returns the list of prediction datasets
        """
        return [data for data in self.data_list if not data.is_fitting()]

    def get_num_fit_data(self) -> int:
        """
        Returns the number of datasets added
        """
        return len([data for data in self.data_list if data.is_fitting()])
