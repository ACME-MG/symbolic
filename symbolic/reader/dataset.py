"""
 Title:         Datset
 Description:   Class for storing data
 Author:        Janzen Choi

"""

# Libraries
from symbolic.helper.io import csv_to_dict

# Dataset class
class Dataset:

    def __init__(self, csv_path:str, fields:list, fitting:bool):
        """
        Constructor for the dataset class
        
        Parameters:
        * `csv_path`: Path to the CSV file
        * `fields`:   Fields to be included in the dataset
        * `fitting`:  Whether the dataset is for fittingg
        """
        
        # Initialise arguments
        self.path = csv_path
        self.fields = fields
        self.fitting = fitting

        # Read the data
        csv_dict = csv_to_dict(csv_path)
        max_length = max([len(csv_dict[field]) for field in self.fields])
        self.data_dict = {}

        # Check and save the data
        for field in self.fields:
            
            # Check field existence
            if not field in csv_dict.keys():
                raise ValueError(f"The '{field}' field does not exist in '{csv_path}'!")

            # Check list lengths
            if isinstance(csv_dict[field], list) and len(csv_dict[field]) != max_length:
                raise ValueError(f"The '{field}' field in '{csv_path}' does not have the same entries as the other fields!")
        
            # Save the data           
            self.data_dict[field] = csv_dict[field]

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
