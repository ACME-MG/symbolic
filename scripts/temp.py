import numpy as np
from pysr import PySRRegressor, TemplateExpressionSpec
from sympy import sympify
 
rng = np.random.RandomState(0)
X = np.linspace(1, 50, 100).reshape(-1, 1)
# y = 2.5 * X[:, 0] ** 1.4 - 9*X[:,0] + 50
y = 2.5 * X[:, 0]

def round_sf(value:float, sf:int) -> float:
    """
    Rounds a float to a number of significant figures

    Parameters:
    * `value`: The value to be rounded; accounts for lists
    * `sf`:    The number of significant figures

    Returns the rounded number
    """
    if isinstance(value, list):
        return [round_sf(v, sf) for v in value]
    format_str = "{:." + str(sf) + "g}"
    rounded_value = float(format_str.format(value))
    return rounded_value

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

expression_spec = create_tes(
    num_inputs = 1,
    submodels = [
        "A * x1 ^ n",
    ]
)

# 3. Set up and run the symbolic regressor
model = PySRRegressor(
    expression_spec  = expression_spec,
    populations      = 16,
    population_size  = 16,
    maxsize          = 16,
    niterations      = 32,
    binary_operators = ["+", "-", "*", "/"],
    elementwise_loss = "loss(prediction, target) = (prediction - target)^2",
)

model.fit(X, y)
print(model.get_best()["julia_expression"])

# f = (#3 * 1.804934) * (#1 + (((#3 + (#2 + #2)) + #2) + -30.525042)); p1 = [39.870716, -0.6011687]; p2 = [0.27247825, 0.6305302]

def template_to_latex(model) -> str:
    """
    Given a fitted symbolic regression model defined with a template expression,
    outputs the latex string representing the best solution

    Parameters:
    * `model`: The fitted model object

    Returns the latex string
    """