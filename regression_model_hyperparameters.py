from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LassoLars, BayesianRidge, SGDRegressor, HuberRegressor, PoissonRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel
from xgboost import XGBRegressor
import numpy as np

regressors = {
    "KNeighborsRegressor": {
        "model": KNeighborsRegressor(),
        "params": {
            "n_neighbors": list(range(1, 51)),
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "leaf_size": list(range(10, 51)),
            "p": [1, 2]
        }
    },
    "DecisionTreeRegressor": {
        "model": DecisionTreeRegressor(),
        "params": {
            "splitter": ["best", "random"],
            "max_depth": [None] + list(range(2, 31)),
            "min_samples_split": list(range(2, 21)),
            "min_samples_leaf": list(range(1, 21)),
            "max_features": [None, "sqrt", "log2"],
        }
    },
    "RandomForestRegressor": {
        "model": RandomForestRegressor(),
        "params": {
            "n_estimators": list(range(5, 501, 5)),
            "max_depth": [None] + list(range(1, 31)),
            "min_samples_split": list(range(1, 31)),
            "min_samples_leaf": list(range(1, 31)),
            "max_features": [None, "auto", "sqrt", "log2"],
        }
    },
    "XGBoostRegressor": {
        "model": XGBRegressor(),
        "params": {
            "n_estimators": list(range(50, 501, 5)),
            "learning_rate": np.logspace(-4, 0, 5),
            "max_depth": [None] + list(range(2, 21)),
            "subsample": np.linspace(0.1, 1, 10),
            "colsample_bytree": np.linspace(0.1, 1, 10),
            "gamma": np.linspace(0, 1, 11)
        }
    },
    "BayesianRidge": {
        "model": BayesianRidge(),
        "params": {
            "n_iter": [100, 300, 500, 1000],
            "tol": np.logspace(-6, -3, 4),
            "alpha_1": np.logspace(-6, 2, 9),
            "alpha_2": np.logspace(-6, 2, 9),
            "lambda_1": np.logspace(-6, 2, 9),
            "lambda_2": np.logspace(-6, 2, 9)
        }
    },
    "SGDRegressor": {
        "model": SGDRegressor(),
        "params": {
            "loss": ["squared_loss", "huber", "epsilon_insensitive"],
            "penalty": ["none", "l2", "l1", "elasticnet"],
            "alpha": np.logspace(-6, 6, 13),
            "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
            "max_iter": [1000, 5000, 10000],
            "tol": np.logspace(-5, -3, 3)
        }
    },
    "AdaBoostRegressor": {
        "model": AdaBoostRegressor(),
        "params": {
            "n_estimators": list(range(50, 501, 5)),
            "learning_rate": np.logspace(-4, 0, 5),
            "loss": ["linear", "square", "exponential"]
        }
    },
    "MLPRegressor": {
        "model": MLPRegressor(),
        "params": {
            # Size of the hidden layers
            "hidden_layer_sizes": [(100,), (50, 50), (100, 100)],
            # Activation function for the hidden layer
            "activation": ["relu", "tanh"],
            "solver": ["adam", "sgd"],  # Solver for weight optimization
            "alpha": [0.0001, 0.001, 0.01],  # L2 penalty (regularization term)
            "learning_rate_init": [0.001, 0.01],  # Initial learning rate
            "max_iter": [200, 500],  # Maximum number of iterations
        }
    },
}
