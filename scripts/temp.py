from pysr import PySRRegressor, TemplateExpressionSpec
import numpy as np

# combine_str = """
#     A = 6.3757e-19; n = 7.643720763; M = 6.461e-20; phi = 17.01786425; chi = 5.9171705;
#     z0 = A*x1^n * ((1-(phi+1)*M*x1^chi*x0)^((phi+1-n)/(phi+1))-1) / (M*x1^chi*(n-phi-1)) + f0(x0,x1,x2);
#     z1 = 1/((phi+1)*M*x1^chi) + f1(x1,x2);
#     (y0-z0)^2+(y1-z1)^2;
# """
# """
#     ((; f0, f1), (x0, x1, x2, y0, y1)) -> begin
#         A = 6.3757e-19
#         n = 7.643720763
#         M = 6.461e-20
#         phi = 17.01786425
#         chi = 5.9171705
#         z0 = A*x1^n * ((1-(phi+1)*M*x1^chi*x0)^((phi+1-n)/(phi+1))-1) / (M*x1^chi*(n-phi-1)) + f0(x0, x1, x2)
#         z1 = 1/((phi+1)*M*x1^chi) + f1(x1, x2)
#         e0 = (y0-z0)^2+(y1-z1)^2
#         e0_valid = [e0x.valid ? e0x : 0 for e0x in e0.x]
#         ValidVector(e0_valid, e0.valid)
#     end
# """

expression_spec = TemplateExpressionSpec(
    function_symbols = ["f0", "f1"],
    # variable_names = ["x0", "x1", "x2", "y0", "y1"],
    combine        = """
        ((; f0, f1), (x0, x1, x2, y0, y1)) -> begin
            A = 6.3757e-19; n = 7.643720763; M = 6.461e-20; phi = 17.01786425; chi = 5.9171705;
            z0 = A*x1^n * ((1-(phi+1)*M*x1^chi*x0)^((phi+1-n)/(phi+1))-1) / (M*x1^chi*(n-phi-1)) + f0(x0, x1, x2);
            z1 = 1/((phi+1)*M*x1^chi) + f1(x1, x2);
            e0 = (y0-z0)^2+(y1-z1)^2;
            e0_valid = [x0x < z1.x[1] ? e0x : zero(e0x) for (x0x, e0x) in zip(x0.x, e0.x)];
            ValidVector(e0_valid, e0.valid)
        end
    """
)

regressor = PySRRegressor(
    expression_spec  = expression_spec,
    populations      = 32,
    population_size  = 32,
    maxsize          = 32,
    niterations      = 64,
    precision        = 64,
    binary_operators = ["+", "*", "/"],
    elementwise_loss = "take_first(p, t) = p"
)

x_data = np.array([
    [2.9520000e+03, 7.0000000e+01, 8.0000000e+02, 4.1000000e-05, 9.9026640e+06],
    [1.2022200e+06, 7.0000000e+01, 8.0000000e+02, 4.8268000e-02, 9.9026640e+06],
    [2.5709760e+06, 7.0000000e+01, 8.0000000e+02, 8.7537000e-02, 9.9026640e+06],
    [3.9732840e+06, 7.0000000e+01, 8.0000000e+02, 1.2346300e-01, 9.9026640e+06],
    [5.3164080e+06, 7.0000000e+01, 8.0000000e+02, 1.5799800e-01, 9.9026640e+06],
    [6.6475800e+06, 7.0000000e+01, 8.0000000e+02, 1.9575300e-01, 9.9026640e+06],
    [7.9175880e+06, 7.0000000e+01, 8.0000000e+02, 2.4064200e-01, 9.9026640e+06],
    [8.9542080e+06, 7.0000000e+01, 8.0000000e+02, 2.9397400e-01, 9.9026640e+06],
    [9.5383080e+06, 7.0000000e+01, 8.0000000e+02, 3.4689400e-01, 9.9026640e+06],
    [9.9026640e+06, 7.0000000e+01, 8.0000000e+02, 4.6810600e-01, 9.9026640e+06],
], dtype=np.float32)
y_data = np.zeros(x_data.shape[0], dtype=np.float32)
regressor.fit(x_data, y_data)
