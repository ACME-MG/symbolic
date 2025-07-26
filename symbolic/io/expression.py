"""
 Title:         Expression
 Description:   Contains expression related functions
 Author:        Janzen Choi

"""

# Libraries
from symbolic.helper.general import round_sf
from sympy import sympify
from sympy.parsing.sympy_parser import parse_expr
from sympy import latex

def get_params(expression:str) -> list:
    """
    Gets the parameters from an expression

    Parameters:
    * `expression`: The expression string

    Returns the list of parameters as strings
    """
    expression_str = expression.replace("^", "**")
    expression = sympify(expression_str)
    variables = expression.free_symbols
    parameters = [str(variable) for variable in variables if not str(variable).startswith("x")]
    return parameters

def julia_to_latex(julia:str, num_inputs:int, submodels:list) -> str:
    """
    Converts an expression to a latex string

    Parameters:
    * `julia`:      The julia expression string
    * `num_inputs`: The number of inputs
    * `submodels`:  List of strings representing submodels;
                    input variables should be named 'x1', 'x2', etc. 

    Returns the latex string
    """

    # Clean and extract parameter information
    julia_clean = julia.replace(" ", "")
    julia_list = julia_clean.split(";")
    param_info_list = julia_list[1:]

    # Construct expressions for each submodel
    expression_list = []
    for param_info, submodel in zip(param_info_list, submodels):
        
        # Get parameter names
        param_names = get_params(submodel)

        # Get parameter values; "p1 = [1,2]"
        param_str_list = param_info.split("=")
        param_values_str = param_str_list[1][1:-1] # remove braces
        param_values = [float(pv) for pv in param_values_str.strip("[]").split(",")]
        param_values = round_sf(param_values, 5)

        # Convert submodel to expression; "A*x1^n" -> "1*x1^2"
        expression = str(submodel).replace("^","**").replace(" ","")
        for param_name, param_value in zip(param_names, param_values):
            expression = expression.replace(str(param_name), str(param_value))
        expression_list.append(f"({expression})")

    # Replace placeholders (e.g., #1) inputs and submodel expressions
    inputs = [f"x{i+1}" for i in range(num_inputs)]
    placeholdees = inputs + expression_list
    placeholders = [f"#{i+1}" for i in range(len(placeholdees))]
    julia_expression = julia_list[0][2:]
    for placeholder, placeholdee in zip(placeholders, placeholdees):
        julia_expression = julia_expression.replace(placeholder, placeholdee)

    # Return latex string
    latex_str = latex(parse_expr(julia_expression, evaluate=False), mul_symbol=' ')
    return f"{latex_str}"
