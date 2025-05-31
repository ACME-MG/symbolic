"""
 Title:         Simulation Interface
 Description:   Interface for running MOOSE simulations
 Author:        Janzen Choi

"""

# Libraries
import inspect, re, time
from symbolic.helper.io import safe_mkdir, get_file_path_exists
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
        
        # Create directories
        if not output_here:
            safe_mkdir(output_path)
            safe_mkdir(self.__output_path__)

    def __print__(self, message:str, add_index:bool=True) -> None:
        """
        Displays a message before running the command (for internal use only)
        
        Parameters:
        * `message`:   the message to be displayed
        * `add_index`: if true, adds a number at the start of the message
        """
        if not add_index:
            print(f"\t  {message}")
        if not self.__verbose__ or not add_index:
            return
        self.__print_index__ += 1
        print(f"   {self.__print_index__})\t{message} ...")
    
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

    def define_problem(self, input_fields:list, output_fields:list):
        """
        Defines the input and output fields

        Parameters:
        * `input_fields`:  List of fields to use as inputs
        * `output_fields`: List of fields to use as outputs
        """
        self.__print__(f"Defining the regression problem")
        
        # Check inputs
        if len(input_fields) == 0:
            raise ValueError("No inputs have been defined!")
        if len(output_fields) == 0:
            raise ValueError("No outputs have been defined!")
        
        # Print summary of inputs
        self.__print__("", False)
        self.__print__(f"Inputs:  {input_fields}", False)
        self.__print__(f"Outputs: {output_fields}", False)
        self.__print__("", False)

        # Define controller
        self.__controller__ = Controller(self.__output_path__, input_fields, output_fields)

    def add_data(self, csv_path:str, training:bool=True) -> None:
        """
        Adds fitting data

        Parameters:
        * `csv_path`: Path to the csv file containing the data
        * `training`: Whether the data will be used for fitting
        """
        csv_path = self.__get_input__(csv_path)
        self.__print__(f"Reading data from '{csv_path}'")
        self.__controller__.add_csv_data(csv_path, training)

    def fit_model(self, model_name:str, **settings) -> None:
        """
        Performs the fitting

        Parameters:
        * `model_name`: Name of the model
        """
        num_data = self.__controller__.get_num_data()
        self.__print__(f"Fitting the model against {num_data} sets of data")
        self.__controller__.fit_model(model_name, **settings)

    def plot_fit(self, x_field:str, y_field:str, x_units:str="",
                 y_units:str="", x_limits:tuple=None, y_limits:tuple=None) -> None:
        """
        Plots the fitting results

        Parameters:
        * `x_field`:  Field to use for the x-axis
        * `y_field`:  Field to use for the y-axis
        * `x_limits`: Limits to apply on the x-axis
        * `y_limits`: Limits to apply on the y-axis
        """
        self.__print__(f"Plotting the fit for the {y_field}-{x_field} curve")
        plot_path = get_file_path_exists(self.__get_output__("plot"), "png")
        self.__controller__.plot_fit(plot_path, x_field, y_field, x_units, y_units, x_limits, y_limits)
