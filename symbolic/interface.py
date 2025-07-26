"""
 Title:         Simulation Interface
 Description:   Interface for running MOOSE simulations
 Author:        Janzen Choi

"""

# Libraries
import inspect, re, time
from symbolic.io.files import safe_mkdir, get_file_path_exists
from symbolic.helper.derivative import remove_after_sp
from symbolic.regression.controller import Controller

# Interface Class
class Interface:

    def __init__(self, title:str="", input_path:str="./data", output_path:str="./results",
                 verbose:bool=True, output_here:bool=False):
        """
        Class to interact with the optimisation code
        
        Parameters:
        * `title`:       Title of the output folder
        * `input_path`:  Path to the input folder
        * `output_path`: Path to the output folder
        * `verbose`:     If true, outputs messages for each function call
        * `output_here`: If true, just dumps the output in ths executing directory
        """
        
        # Initialise internal variables
        self.__print_index__ = 0
        self.__verbose__     = verbose
        self.__controller__  = None

        # Starting code
        time_str = time.strftime("%A, %D, %H:%M:%S", time.localtime())
        self.__print__(f"\n  Starting on {time_str}\n", add_index=False)
        self.__start_time__ = time.time()
        time_stamp = time.strftime("%y%m%d%H%M%S", time.localtime(self.__start_time__))
        
        # Define input and output
        self.__input_path__ = input_path
        file_path = inspect.currentframe().f_back.f_code.co_filename
        file_name = file_path.split("/")[-1].replace(".py","")
        title = f"_{file_name}" if title == "" else f"_{title}"
        title = re.sub(r"[^a-zA-Z0-9_]", "", title.replace(" ", "_"))
        self.__output_dir__ = "." if output_here else time_stamp
        self.__output_path__ = "." if output_here else f"{output_path}/{self.__output_dir__}{title}"
        
        # Define input / output functions
        self.__get_input__  = lambda x : f"{self.__input_path__}/{x}"
        self.__get_output__ = lambda x : f"{self.__output_path__}/{x}"
        
        # Initialise controller
        self.__controller__ = Controller(self.__output_path__)
        
        # Create directories
        if not output_here:
            safe_mkdir(output_path)
            safe_mkdir(self.__output_path__)

    def __print__(self, message:str, add_index:bool=True, sub_index:bool=False) -> None:
        """
        Displays a message before running the command (for internal use only)
        
        Parameters:
        * `message`:   the message to be displayed
        * `add_index`: if true, adds a number at the start of the message
        * `sub_index`: if true, adds a number as a decimal
        """
        if not add_index:
            print(f"\t  {message}")
        if not self.__verbose__ or not add_index:
            return
        if sub_index:
            self.__print_subindex__ += 1
            print_index = f"     {self.__print_index__}.{self.__print_subindex__}"
        else:
            self.__print_index__ += 1
            self.__print_subindex__ = 0
            print_index = f"{self.__print_index__}"
        print(f"   {print_index})\t{message} ...")

    def __del__(self):
        """
        Prints out the final message (for internal use only)
        """
        time_str = time.strftime("%A, %D, %H:%M:%S", time.localtime())
        duration = round(time.time() - self.__start_time__)
        duration_h = duration // 3600
        duration_m = (duration - duration_h * 3600) // 60
        duration_s = duration - duration_h * 3600 - duration_m * 60
        duration_str_list = [
            f"{duration_h} hours" if duration_h > 0 else "",
            f"{duration_m} mins" if duration_m > 0 else "",
            f"{duration_s} seconds" if duration_s > 0 else ""
        ]
        duration_str = ", ".join([d for d in duration_str_list if d != ""])
        duration_str = f"in {duration_str}" if duration_str != "" else "in less than 1 second"
        self.__print__(f"\n  Finished on {time_str} {duration_str}\n", add_index=False)

    def define_model(self, model_name:str, **kwargs) -> None:
        """
        Defines the model

        Parameters:
        * `model_name`: Name of the model
        """
        self.__print__(f"Initialising the '{model_name}' model")
        self.__controller__.define_model(model_name, **kwargs)

    def add_data(self, csv_path:str, fitting:bool=True, weights:list=None) -> None:
        """
        Adds fitting data

        Parameters:
        * `csv_path`: Path to the csv file containing the data
        * `fitting`:  Whether the data will be used for fitting
        * `weights`:  List of weights; does not apply weights if undefined
        """
        csv_path = self.__get_input__(csv_path)
        fit_str = "calibration" if fitting else "validation"
        self.__print__(f"Reading {fit_str} data from '{csv_path}'")
        self.__controller__.add_data(csv_path, fitting)
        if weights != None:
            self.__controller__.set_weights(weights)
            
    def get_data_dict(self) -> dict:
        """
        Retrieves last added data
        """
        self.__print__("Retrieving data", sub_index=True)
        return self.__controller__.get_last_data().get_data_dict()

    def get_data(self, field:str) -> list:
        """
        Retrieves last added data under a field

        Parameters:
        * `field`: The field from which data is retrieved
        
        Returns the list of data
        """
        self.__print__(f"Retrieving data under '{field}'", sub_index=True)
        data_dict = self.__controller__.get_last_data().get_data_dict()
        if not field in data_dict.keys():
            raise ValueError(f"The '{field}' field does not exist!")
        return data_dict[field]

    def sparsen_data(self, new_size:int=100) -> None:
        """
        Sparsen the most recently added dataset

        Parameters:
        * `new_size`: New size to sparsen the data
        """
        old_size = self.__controller__.get_last_data().get_size()
        self.__print__(f"Sparsening data ({old_size} -> {new_size})", sub_index=True)
        self.__controller__.sparsen_data(new_size)

    def remove_oxidation(self, window:int=0.1, acceptance:float=0.9) -> None:
        """
        Removes the data after the tertiary creep for the most recently added
        
        Parameters:
        * `window`:     The window ratio to identify the stationary points of the derivative; the actual
                        window size is the product of `window` and the number of data points (1000)
        * `acceptance`: The acceptance value for identifying the nature of stationary points; should
                        have a value between 0.5 and 1.0
        """
        self.__print__(f"Removing the creep oxidisation effects", sub_index=True)
        data = self.__controller__.get_last_data()
        data_dict = data.get_data_dict()
        data_dict = remove_after_sp(data_dict, "max", "time", "strain", window, acceptance, 0)
        data.set_data_dict(data_dict)
        self.__controller__.set_last_data(data)

    def fit_model(self) -> None:
        """
        Performs the fitting

        Parameters:
        * `model_name`: Name of the model
        """
        num_data = self.__controller__.get_num_data()
        num_fit_data = self.__controller__.get_num_fit_data()
        total_size = sum([data.get_size() for data in self.__controller__.get_data_list()])
        self.__print__(f"Fitting the model against {num_fit_data}/{num_data} datasets ({total_size} points)")
        self.__controller__.fit_model()

    def plot_fit(self, x_field:str, y_field:str, x_units:str="", x_scale:float=1.0,
                 y_scale:float=1.0, y_units:str="", x_limits:tuple=None,
                 y_limits:tuple=None, file_name:str="", conditions:dict={}) -> None:
        """
        Plots the fitting results

        Parameters:
        * `x_field`:    Field to use for the x-axis
        * `y_field`:    Field to use for the y-axis
        * `x_scale`:    Factor to apply to x values
        * `y_scale`:    Factor to apply to y values
        * `x_limits`:   Limits to apply on the x-axis
        * `y_limits`:   Limits to apply on the y-axis
        * `file_name`:  Custom name for the plot file
        * `conditions`: Conditions to constrain plotting
        """
        self.__print__(f"Plotting the fit for the {y_field}-{x_field} curve")
        file_name = file_name if file_name != "" else "plot_fit"
        plot_path = get_file_path_exists(self.__get_output__(file_name), "png")
        self.__controller__.plot_fit(plot_path, x_field, y_field, x_scale, y_scale,
                                     x_units, y_units, x_limits, y_limits, conditions)

    def plot_1to1(self, handle, label:str="", units:str="", limits:tuple=None,
                  file_name:str="", conditions:dict={}) -> None:
        """
        Plots 1:1 comparison plots based on a function handle;
        the function must take a dictionary as an argument and
        return a list of values

        Parameters:
        * `handle`:    The function handle
        * `label`:     Label to represent values
        * `units`:     Units to place beside label
        * `limits`:    Limits of the plot
        * `file_name`: Custom name for the plot file
        * `conditions`: Conditions to constrain plotting
        """
        self.__print__("Plotting a 1:1 comparison")
        file_name = file_name if file_name != "" else "plot_1to1"
        plot_path = get_file_path_exists(self.__get_output__(file_name), "png")
        self.__controller__.plot_1to1(plot_path, handle, label, units, limits, conditions)

    def plot_equation(self, file_name:str="") -> None:
        """
        Saves an image of the equation

        Parameters:
        * `file_name`: Custom name for the plot file
        """
        self.__print__(f"Plotting the equation for the best fit")
        file_name = file_name if file_name != "" else "equation"
        equation_path = get_file_path_exists(self.__get_output__(file_name), "png")
        self.__controller__.plot_equation(equation_path)
