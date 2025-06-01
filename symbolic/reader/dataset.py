"""
 Title:         Datset
 Description:   Class for storing data
 Author:        Janzen Choi

"""

# Libraries
from symbolic.helper.general import get_spread_list
from symbolic.helper.io import csv_to_dict
from scipy.interpolate import interp1d

# Dataset class
class Dataset:

    def __init__(self, csv_path:str, fields:list, fitting:bool):
        """
        Constructor for the dataset class
        
        Parameters:
        * `csv_path`: Path to the CSV file
        * `fields`:   Fields to be included in the dataset
        * `fitting`:  Whether the dataset is for fitting
        """
        
        # Initialise arguments
        self.path = csv_path
        self.fields = fields
        self.fitting = fitting

        # Read the data
        csv_dict = csv_to_dict(csv_path)
        max_length = max([len(csv_dict[field]) for field in self.fields if isinstance(csv_dict[field], list)])
        self.data_dict = {}

        # Check and save the data
        for field in self.fields:
            if not field in csv_dict.keys():
                raise ValueError(f"The '{field}' field does not exist in '{csv_path}'!")
            if isinstance(csv_dict[field], list) and len(csv_dict[field]) != max_length:
                raise ValueError(f"The '{field}' field in '{csv_path}' does not have the same entries as the other fields!")
            self.data_dict[field] = csv_dict[field]

        # Initialise weigghts
        self.weights = [1, 1] # uniform weighting

    def get_path(self) -> str:
        """
        Returns the path to the CSV files
        """
        return self.path

    def get_fields(self) -> list:
        """
        Returns the list of fields to be included in the dataset
        """
        return self.fields

    def is_fitting(self) -> bool:
        """
        Returns whether the dataset is intended for fitting
        """
        return self.fitting

    def set_data_dict(self, data_dict:dict) -> None:
        """
        Resets the data dictionary

        Parameters:
        * `data_dict`: The new data dictionary
        """
        self.data_dict = data_dict

    def get_data_dict(self) -> dict:
        """
        Returns the actual dataset
        """
        return self.data_dict
    
    def get_data(self, field:str) -> list:
        """
        Gets the data under a field
        
        Parameters:
        * `field`: The field under which the data is stored
        
        Returns the data under the defined field
        """
        if not field in self.fields:
            raise ValueError(f"The '{field}' field has not been defined!")
        return self.data_dict[field]

    def get_size(self) -> int:
        """
        Returns the size of the lists in the dataset;
        assumes all lists are of the same size
        """
        for field in self.data_dict.keys():
            if isinstance(self.data_dict[field], list):
                return len(self.data_dict[field]) 
        return 0

    def set_weights(self, weights:list) -> None:
        """
        Sets the weights in the data set

        Parameters:
        * `weights`: Weights to apply
        """
        self.weights = weights

    def get_weights(self) -> list:
        """
        Spline interpolates the weights based on the number of data points;
        assumes relatively uniform spreading of values
        """
        size = self.get_size()
        index_list = get_spread_list(len(self.weights), size)
        interp = interp1d(index_list, self.weights)
        weight_list = interp(range(size)).tolist()
        return weight_list
