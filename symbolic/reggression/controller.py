"""
 Title:         Controller
 Description:   Directs the symbolic reggression procedure
 Author:        Janzen Choi

"""

# Libraries
from symbolic.helper.io import csv_to_dict

# Controller class
class Controller:

    def __init__(self, input_fields:list, output_fields:list):
        """
        Constructor for the controller class

        Parameters:
        * `input_fields`:  List of fields to use as inputs
        * `output_fields`: List of fields to use as outputs
        """
        self.input_fields = input_fields
        self.output_fields = output_fields
        self.all_fields = self.input_fields + self.output_fields
        self.data_dict_list = []
    
    def add_csv_data(self, csv_path:str) -> None:
        """
        Adds fitting data

        Parameters:
        * `csv_path`: Path to the csv file containing the data
        """

        # Read and check the data
        csv_dict = csv_to_dict(csv_path)
        for field in self.all_fields:
            if not field in csv_dict.keys():
                raise ValueError(f"The '{field}' field does not exist in '{csv_path}'!")
        
        # Process the data
        data_dict = {"path": csv_path}
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

    def train_model(self):
        """
        Trains the symbolic reggression model
        """

        training_data_dict = dict(zip(self.all_fields, [[] for _ in range(len(self.all_fields))]))