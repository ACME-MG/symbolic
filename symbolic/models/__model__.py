"""
 Title:         Model
 Description:   For creating model files
 Author:        Janzen Choi

"""

# Libraries
import importlib, os, pathlib, re, sys, numpy as np
from symbolic.helper.general import transpose, flatten, get_thinned_list
from symbolic.reader.dataset import Dataset

# Model Class
class __Model__:

    def __init__(self, name:str, output_path:str):
        """
        Template class for model objects
        
        Parameters:
        * `name`:        The name of the model
        * `output_path`: Output path
        """
        self.name = name
        self.output_path = output_path
        self.fields = []

    def set_fields(self, fields:list) -> None:
        """
        Sets the required fields
        
        Parameters:
        * `fields`: List of fields to use as inputs or outputs
        """
        self.fields = fields

    def get_fields(self) -> list:
        """
        Returns the list of fields to use as inputs or outputs
        """
        return self.fields

    def get_name(self) -> str:
        """
        Gets the name of the model
        """
        return self.name

    def get_fit_weights(self, data_list:list) -> np.array:
        """
        Gets an array of weights corresponding to the fitting data

        Parameters:
        * `data_list`: List of data objects
        
        Returns the new array of weights
        """
        weights_list = [data.get_weights() for data in data_list]
        weights_list = flatten(weights_list)
        weights_array = np.array(weights_list)
        return weights_array

    def initialise(self, **kwargs) -> None:
        """
        Initialises the model; must be overridden
        """
        raise NotImplementedError(f"The 'initialise' function has not been defined for the '{self.name}' model!")

    def fit(self, data_list:list) -> None:
        """
        Performs the fitting; must be overridden

        Parameters:
        * `data_list`: List of datasets
        """
        raise NotImplementedError(f"The 'fit' function has not been defined for the '{self.name}' model!")
    
    def predict(self, data_list:list) -> list:
        """
        Predicts data using fit; must be overridden

        Parameters:
        * `data_list`: List of datasets

        Returns the predicted data
        """
        raise NotImplementedError(f"The 'predict' function has not been implemented for the '{self.name}' model!")
    
    def get_latex(self) -> str:
        """
        Returns the LaTeX equation of the final fit; must be overridden
        """
        raise NotImplementedError(f"The 'get_latex' function has not been implemented for the '{self.name}' model!")

def sparsen_data(data:Dataset, new_size:int) -> Dataset:
    """
    Sparsens a datset

    Parameters:
    * `data`:     Data object
    * `new_size`: New size for the data

    Returns the sparsened data
    """
    data_dict = data.get_data_dict()
    for field in data_dict.keys():
        if isinstance(data_dict[field], list):
            data_dict[field] = get_thinned_list(data_dict[field], new_size)
    data.set_data_dict(data_dict)
    return data
        
def convert_data(data_list:list, field_list:list) -> np.array:
    """
    Converts a list of data objects into a numpy array

    Parameters:
    * `data_list`:  List of data objects
    * `field_list`: List of fields

    Returns the data as a numpy array
    """

    # Prepare data list
    field_data_list = []
    
    # Synthesise the data
    for data in data_list:

        # Get data
        data_dict = data.get_data_dict()
        has_list = True in [isinstance(data_dict[field], list) for field in field_list]
        if has_list:
            max_length = max([len(data_dict[field]) for field in field_list if isinstance(data_dict[field], list)])
        else:
            max_length = 1

        # Add data
        field_data_sublist = [data_dict[field] if isinstance(data_dict[field], list) else [data_dict[field]]*max_length for field in field_list]
        field_data_sublist = transpose(field_data_sublist)
        field_data_list += field_data_sublist

    # Convert and return
    field_data_array = np.array(field_data_list)
    return field_data_array

def replace_variables(latex_string:str, new_variables:list) -> str:
    """
    Replaces variable names inside a latex string

    Parameters:
    * `latex_string`:  The latex string
    * `new_variables`: List of new variables in regex

    Returns the replaced latex string
    """
    def replacer(match):
        index = int(match.group(1))
        if 0 <= index < len(new_variables):
            return new_variables[index]
        return match.group(0)
    return re.sub(r'x_\{(\d+)\}', replacer, latex_string)

def equate_to(prepend:str, latex_string:str) -> str:
    """
    Performs a simple prepending

    Parameters:
    * `prepend`:      The string to prepend the larger string
    * `latex_string`: The larger LaTeX string

    Returns `{prepend} + " = " + {latex_string}`
    """
    return f"{prepend} = {latex_string}"

def get_model(model_path:str, output_path:str, **kwargs) -> str:
    """
    Gets the model file's content
    
    Parameters:
    * `model_path`:  The path to the the model
    * `output_path`: The path to the output file
    """

    # Separate model file and path
    model_file = model_path.split("/")[-1]
    model_path = "/".join(model_path.split("/")[:-1])
    models_dir = pathlib.Path(__file__).parent.resolve()
    models_dir = f"{models_dir}/{model_path}"

    # Get available models in current folder
    files = os.listdir(models_dir)
    files = [file.replace(".py", "") for file in files]
    files = [file for file in files if not file in ["__model__", "__pycache__"]]
    
    # Raise error if model name not in available models
    if not model_file in files:
        raise NotImplementedError(f"The model '{model_file}' has not been implemented")

    # Import and prepare model
    module_path = f"{models_dir}/{model_file}.py"
    spec = importlib.util.spec_from_file_location("model_file", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    
    # Initialise and return the model
    from model_file import Model
    model = Model(model_file, output_path)
    model.initialise(**kwargs)
    return model
