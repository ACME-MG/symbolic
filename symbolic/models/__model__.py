"""
 Title:         Model
 Description:   For creating model files
 Author:        Janzen Choi

"""

# Libraries
import importlib, os, pathlib, sys, numpy as np

# Model Class
class __Model__:

    def __init__(self, name:str, output_path:str, input_fields:str, output_fields:list):
        """
        Template class for model objects
        
        Parameters:
        * `name`:          The name of the model
        * `output_path`:   Output path
        * `input_fields`:  List of fields to use as inputs
        * `output_fields`: List of fields to use as outputs
        """
        self.name = name
        self.output_path = output_path
        self.input_fields = input_fields
        self.output_fields = output_fields

    def get_name(self) -> str:
        """
        Gets the name of the model
        """
        return self.name
    
    def fit(self, input_data:np.array, output_data:np.array) -> None:
        """
        Performs the fitting; must be overridden

        Parameters:
        * `input_data`:  Input data
        * `output_data`: Output data
        """
        raise NotImplementedError(f"The 'fit' function has not been defined for the '{self.name}' model!")
    
    def predict(self, input_data:np.array) -> np.array:
        """
        Predicts data using fit; must be overridden

        Parameters:
        * `input_data`: Input data

        Returns the predicted data
        """
        raise NotImplementedError(f"The 'predict' function has not been implemented for the '{self.name}' model!")

def get_model(model_path:str, output_path:str, **settings) -> str:
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
    model.initialise(**settings)
    return model
