"""
 Title:         Submodel
 Description:   Functions for incorporating submodels
 Author:        Janzen Choi

"""

# Libraries
from symbolic.io.expression import get_params
from pysr import TemplateExpressionSpec

def create_tes(num_inputs:int, submodels:list) -> TemplateExpressionSpec:
    """
    Creates a template expression specification

    Parameters:
    * `num_inputs`: The number of inputs
    * `submodels`:  List of strings representing submodels;
                    input variables should be named 'x1', 'x2', etc. 

    Returns the template expression specification
    """
    
    # Define input variables
    inputs = [f"x{i+1}" for i in range(num_inputs)]

    # Extract information from each submodel
    submodel_defs = ""
    num_params_list = []
    for i, submodel in enumerate(submodels):

        # Extract the parameters
        parameters = get_params(submodel)
        num_params_list.append(len(parameters))

        # Construct the combine string
        parameter_def = ", ".join(parameters) + " = " + ", ".join([f"p{i+1}[{j+1}]" for j in range(len(parameters))])
        expression_def = f"y{i+1} = {submodel}"
        submodel_def = f"{parameter_def}\n{expression_def}\n"
        submodel_defs += submodel_def

    # Define the parameters dict
    parameters_dict = {}
    for i, num_params in enumerate(num_params_list):
        parameters_dict[f"p{i+1}"] = num_params

    # Create the combined string 
    combined_str = submodel_defs + "f(" + ", ".join(inputs) + ", "
    combined_str += ", ".join([f"y{j+1}" for j in range(len(submodels))]) + ")"

    # Define the expression specification and return
    expression_spec = TemplateExpressionSpec(
        expressions    = ["f"],
        variable_names = inputs,
        parameters     = parameters_dict,
        combine        = combined_str
    )
    return expression_spec

