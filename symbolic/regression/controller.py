"""
 Title:         Controller
 Description:   Directs the symbolic regression procedure
 Author:        Janzen Choi

"""

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from symbolic.helper.io import csv_to_dict
from symbolic.helper.plotter import prep_plot, set_limits, add_legend, save_plot
from symbolic.helper.plotter import EXP_COLOUR, CAL_COLOUR, VAL_COLOUR
from symbolic.models.__model__ import get_model

# Controller class
class Controller:

    def __init__(self, output_path:str, input_fields:list, output_fields:list):
        """
        Constructor for the controller class

        Parameters:
        * `output_path`:   Path to the output directory
        * `input_fields`:  List of fields to use as inputs
        * `output_fields`: List of fields to use as outputs
        """
        
        # Initialise arguments
        self.output_path = output_path
        self.input_fields = input_fields
        self.output_fields = output_fields
        
        # Initialise internal variables
        self.all_fields = self.input_fields + self.output_fields
        self.model = None
        self.data_dict_list = []
    
    def add_csv_data(self, csv_path:str, training:bool=True) -> None:
        """
        Adds fitting data

        Parameters:
        * `csv_path`: Path to the csv file containing the data
        * `training`: Whether the data will be used for fitting
        """

        # Read and check the data
        csv_dict = csv_to_dict(csv_path)
        for field in self.all_fields:
            if not field in csv_dict.keys():
                raise ValueError(f"The '{field}' field does not exist in '{csv_path}'!")
        
        # Process the data
        data_dict = {"path": csv_path, "training": training}
        max_length = max([len(csv_dict[field]) for field in self.all_fields])
        for field in self.all_fields:
            if isinstance(csv_dict[field], float):
                data_dict[field] = [csv_dict[field]]*max_length
            elif isinstance(csv_dict[field], list):
                if len(csv_dict[field]) != max_length:
                    raise ValueError(f"The data under the '{field}' field may not be the same length!")
                data_dict[field] = csv_dict[field]

        # Append data
        self.data_dict_list.append(data_dict)

    def fit_model(self, model_name, **settings):
        """
        Fits the symbolic regression model

        Parameters:
        * `model_name`: Name of the model
        """

        # Prepare fitting data
        input_data = []
        output_data = []

        # Iterate through the training data
        for data_dict in self.data_dict_list:
            
            # Ignore non-training data
            if not data_dict["training"]:
                continue

            # Synthesise data
            data_length = len(data_dict[list(data_dict.keys())[0]])
            for i in range(data_length):
                input_list = [data_dict[field][i] for field in self.input_fields]
                output_list = [data_dict[field][i] for field in self.output_fields]
                input_data.append(input_list)
                output_data.append(output_list)

        # Prepare training
        input_data = np.array(input_data)
        output_data = np.array(output_data)
        print()
        
        # Perform training
        self.model = get_model(model_name, self.output_path, **settings)
        self.model.fit(input_data, output_data)

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
        for data_dict in self.data_dict_list:
            plt.scatter(data_dict[x_field], data_dict[y_field], color=EXP_COLOUR, s=8**2)

        # Format and save
        set_limits(x_limits, y_limits)
        add_legend()
        save_plot(plot_path, **settings)

    def get_num_data(self) -> int:
        """
        Returns the number of datasets added
        """
        return len(self.data_dict_list)

    def get_num_training_data(self) -> int:
        """
        Returns the number of datasets added
        """
        return len([dd for dd in self.data_dict_list if dd["training"]])
