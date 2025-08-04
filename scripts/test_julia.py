import sys; sys.path += [".."]
import numpy as np
from symbolic.regression.expression import save_latex, replace_variables, julia_to_expression, expression_to_latex
from symbolic.regression.expression import round_expression, set_parameters, evaluate_expression

julia = "f1 = #2 * 12376.228496393594; p = [12464.824043183033, 1.9615939337418535, -12464.824101263024, 0.8152816667838239, -3.3670639500886685e-8]"
combine_str = """
    A = 10^fA(x1,x2); n = abs(fn(x1,x2)); M = 10^fM(x1,x2); phi = abs(fphi(x1,x2)); chi = abs(fchi(x1,x2));
    z0 = A*x1^n * ((1-(phi+1)*M*x1^chi*x0)^((phi+1-n)/(phi+1))-1) / (M*x1^chi*(n-phi-1)) + f0(x0,x1,x2);
    z1 = 1/((phi+1)*M*x1^chi) + f1(x1,x2);
    z2 = A*x1^n / (M*x1^chi*(phi+1-n)) + f0(x0,x1,x2);
    e0 = ((y0-z0)/y0)*100;
    e1 = ((y1-z1)/y1)*100;
    e2 = ((y2-z2)/y2)*100;
    e0^2 + e1^2 + e2^2
"""

# parameter_values = [6.37573E-17, 7.643720763, 2.32596E-16, 17.01786425, 5.9171705]
# parameter_map = dict(zip([f"p[{i+1}]" for i in range(len(parameter_values))], parameter_values))
# combine_str = set_parameters(combine_str, parameter_map)
expression_dict = julia_to_expression(julia, combine_str)

# expression_dict = round_expression(expression_dict, 5)
latex_dict = expression_to_latex(expression_dict)
variable_map = {
    "x0": r't',
    "x1": r'\sigma',
    "x2": r'T',
    "z0": r'\varepsilon',
    "z1": r't_{f}',
    "z2": r'\varepsilon_{f}',
}
latex_dict = replace_variables(latex_dict, variable_map)
save_latex("plot.png", [latex_dict[p] for p in ["z0", "z1", "t0", "t1", "t2", "t3", "t4"]])
# save_latex("plot.png", [latex_dict[p] for p in ["z0", "z1", "A", "n", "M", "phi", "chi"]])

print(evaluate_expression(expression_dict, "z0", {"x0": [0.1], "x1": [80], "x2": [800]}))
