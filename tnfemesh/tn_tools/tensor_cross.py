from typing import List, Callable, Optional
import numpy as np
import teneva


class TTCrossConfig:
    """
    Configuration class for the tensor train cross approximation algorithm.

    Args:
        cache (dict, optional): Cache for storing requested function values.
        info (dict, optional): Stores TTCross run information.
        num_sweeps (int, optional): Number of sweeps for DMRG. Defaults to 10.
        rel_stagnation_tol (float, optional): Relative stagnation tolerance. Defaults to 1e-4.
        max_func_calls (Optional[int], optional): Maximum number of function calls. Defaults to None.
        cache_calls_factor (int, optional): If the number of calls to cache is this factor
            times larger than number of function calls, TTCross stops. Defaults to 20.
        num_anova_init (int, optional): Number of training indices for ANOVA initializations.
            Defaults to 1000.
        anova_order (int, optional): Order of the ANOVA decomposition. Defaults to 2.
        verbose (bool, optional): Verbose output. Defaults to False.
    """

    def __init__(self,
                 cache: Optional[dict] = None,
                 info: Optional[dict] = None,
                 num_sweeps: int = 10,
                 rel_stagnation_tol: float = 1e-4,
                 max_func_calls: Optional[int] = None,
                 cache_calls_factor: int = 5,
                 num_anova_init: int = 1000,
                 anova_order: int = 2,
                 verbose: bool = False):

        self.cache = cache
        self.info = info
        self.num_sweeps = num_sweeps
        self.rel_stagnation_tol = rel_stagnation_tol
        self.max_func_calls = max_func_calls
        self.cache_calls_factor = cache_calls_factor
        self.num_anova_init = num_anova_init
        self.anova_order = anova_order
        self.verbose = verbose

    def to_dict(self):
        """
        Convert all attributes of the configuration to a dictionary that is passed to
        teneva

        Returns:
            dict: A dictionary representation of the configuration.
        """
        kwargs = {"cache": self.cache,
                  "info": self.info,
                  "nswp": self.num_sweeps,
                  "e": self.rel_stagnation_tol,
                  "log": self.verbose,
                  "m": self.max_func_calls,
                  "m_cache_scale": self.cache_calls_factor,
                  "num_anova_init": self.num_anova_init,
                  "anova_order": self.anova_order}

        return kwargs


def gen_teneva_indices(num_indices: int, tensor_shape: List[int]) -> np.ndarray:
    """
    Generate random indices for a tensor of shape tensor_shape.

    Args:
        num_indices (int): Number of indices to generate.
        tensor_shape (List[int]): Shape of the tensor.

    Returns:
        np.ndarray: Random indices of shape (num_indices, len(tensor_shape)).
    """
    idxs = np.vstack([np.random.choice(k, num_indices) for k in tensor_shape]).T
    return idxs

def anova_init_tensor_train(oracle: Callable[[np.ndarray], np.ndarray],
                            train_indices: np.ndarray,
                            order: int = 2) -> List[np.ndarray]:
    """
    Initialize the tensor train with the ANOVA decomposition of the training data.

    Args:
        oracle (Callable[[np.ndarray], np.ndarray]): Oracle function.
        train_indices (np.ndarray): Training indices.
        order (int, optional): Order of the ANOVA decomposition. Defaults to 2.

    Returns:
        List[np.ndarray]: List of TT-cores for the ANOVA decomposition of the training data.
    """
    ytrain = oracle(train_indices)
    yanova = teneva.anova(train_indices, ytrain, order=order)
    return yanova

def tensor_train_cross_approximation(oracle: Callable[[np.ndarray], np.ndarray],
                                     tt_init: List[np.ndarray],
                                     **kwargs) -> List[np.ndarray]:
    """
    Approximate the tensor train with the cross approximation algorithm.

    Args:
        oracle (Callable[[np.ndarray], np.ndarray]): Oracle function.
        tt_init (List[np.ndarray]): Initial tensor train.
        **kwargs: Additional keyword arguments for the cross approximation algorithm.

    Returns:
        List[np.ndarray]: List of TT-cores for the approximated tensor train.
    """

    return teneva.cross(oracle, tt_init, **kwargs)

def test_accuracy(oracle: Callable[[np.ndarray], np.ndarray],
                  approx_tt: List[np.ndarray],
                  test_indices: np.ndarray) -> float:
    """
    Test the accuracy of the approximated tensor train.

    Args:
        oracle (Callable[[np.ndarray], np.ndarray]): Oracle function.
        approx_tt (List[np.ndarray]): Approximated tensor train cores.
        test_indices (np.ndarray): Test indices.

    Returns:
        float: Relative error of the approximated tensor train.
    """
    ytest = oracle(test_indices)
    error = teneva.accuracy_on_data(approx_tt, test_indices, ytest)
    return error

def test_accuracy_random(oracle: Callable[[np.ndarray], np.ndarray],
                         approx_tt: List[np.ndarray],
                         num_test_indices: int,
                         tensor_shape: List[int]) -> float:
    """
    Test the accuracy of the approximated tensor train with random test indices.

    Args:
        oracle (Callable[[np.ndarray], np.ndarray]): Oracle function.
        approx_tt (List[np.ndarray]): Approximated tensor train cores.
        num_test_indices (int): Number of test indices.
        tensor_shape (List[int]): Shape of the tensor.

    Returns:
        float: Relative error of the approximated tensor train.
    """
    test_indices = gen_teneva_indices(num_test_indices, tensor_shape)
    return test_accuracy(oracle, approx_tt, test_indices)
