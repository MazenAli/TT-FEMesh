import numpy as np
import pytest
import teneva

from ttfemesh.tt_tools.tensor_cross import (
    TTCrossConfig,
    anova_init_tensor_train,
    error_on_indices,
    error_on_random_indices,
    gen_teneva_indices,
    tensor_train_cross_approximation,
)


class TestTTCrossConfig:
    def test_default_config(self):
        config = TTCrossConfig()
        assert config.num_sweeps == 10
        assert config.rel_stagnation_tol == 1e-4
        assert config.max_func_calls is None
        assert config.cache_calls_factor == 5
        assert config.num_anova_init == 1000
        assert config.anova_order == 2
        assert config.verbose is False

    def test_custom_config(self):
        config = TTCrossConfig(
            num_sweeps=5,
            rel_stagnation_tol=1e-6,
            max_func_calls=1000,
            cache_calls_factor=10,
            num_anova_init=500,
            anova_order=3,
            verbose=True,
        )
        assert config.num_sweeps == 5
        assert config.rel_stagnation_tol == 1e-6
        assert config.max_func_calls == 1000
        assert config.cache_calls_factor == 10
        assert config.num_anova_init == 500
        assert config.anova_order == 3
        assert config.verbose is True

    def test_to_dict(self):
        config = TTCrossConfig()
        config_dict = config.to_dict()
        assert config_dict["nswp"] == 10
        assert config_dict["e"] == 1e-4
        assert config_dict["m"] is None
        assert config_dict["m_cache_scale"] == 5
        assert config_dict["num_anova_init"] == 1000
        assert config_dict["anova_order"] == 2
        assert config_dict["log"] is False


class TestGenTenevaIndices:
    def test_basic_generation(self):
        num_indices = 10
        tensor_shape = [2, 3, 4]
        indices = gen_teneva_indices(num_indices, tensor_shape)

        assert isinstance(indices, np.ndarray)
        assert indices.shape == (num_indices, len(tensor_shape))
        assert np.all(indices >= 0)
        assert np.all(indices < np.array(tensor_shape))

    def test_single_index(self):
        num_indices = 1
        tensor_shape = [2, 2, 2]
        indices = gen_teneva_indices(num_indices, tensor_shape)

        assert indices.shape == (1, 3)
        assert np.all(indices >= 0)
        assert np.all(indices < 2)


class TestAnovaInitTensorTrain:
    def test_basic_anova_init(self):
        def oracle(x):
            return np.sum(x, axis=1)

        train_indices = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        order = 2

        tt_cores = anova_init_tensor_train(oracle, train_indices, order)

        assert isinstance(tt_cores, list)
        assert len(tt_cores) == 2
        for core in tt_cores:
            assert isinstance(core, np.ndarray)

    def test_invalid_order(self):
        def oracle(x):
            return np.sum(x, axis=1)

        train_indices = np.array([[0, 0], [1, 1]])

        with pytest.raises(ValueError):
            anova_init_tensor_train(oracle, train_indices, order=3)


class TestTensorTrainCrossApproximation:
    def test_basic_approximation(self):
        random_array = np.random.randn(2, 2, 2)

        def oracle(indices):
            collect = []
            for idx in indices:
                collect.append(random_array[tuple(idx)])
            return np.array(collect)

        tt_init = teneva.rand([2, 2, 2], 2)

        tt_approx = tensor_train_cross_approximation(oracle, tt_init, nswp=2)

        assert isinstance(tt_approx, list)
        assert len(tt_approx) == 3
        for core in tt_approx:
            assert isinstance(core, np.ndarray)


class TestTestAccuracy:
    def test_basic_accuracy(self):
        def oracle(indices):
            return np.ones(indices.shape[0])

        tt_approx = teneva.rand([2, 2, 2], 2)

        test_indices = np.array([[0, 0], [1, 1]])
        error = error_on_indices(oracle, tt_approx, test_indices)

        assert isinstance(error, float)
        assert error >= 0

    def test_zero_error(self):
        def oracle(indices):
            return np.ones(indices.shape[0])

        tt_approx = [np.ones((1, 2, 1)) for _ in range(2)]

        test_indices = np.array([[0, 0], [1, 1]])
        error = error_on_indices(oracle, tt_approx, test_indices)

        assert error == 0.0


class TestTestAccuracyRandom:
    def test_basic_random_accuracy(self):
        def oracle(x):
            return np.sum(x, axis=1)

        tt_approx = [np.ones((1, 2, 1)) for _ in range(2)]

        num_test_indices = 10
        tensor_shape = [2, 2]
        error = error_on_random_indices(oracle, tt_approx, num_test_indices, tensor_shape)

        assert isinstance(error, float)
        assert error >= 0

    def test_zero_random_error(self):
        def oracle(indices):
            return np.ones(indices.shape[0])

        tt_approx = [np.ones((1, 2, 1)) for _ in range(2)]

        num_test_indices = 10
        tensor_shape = [2, 2]
        error = error_on_random_indices(oracle, tt_approx, num_test_indices, tensor_shape)

        assert error == 0.0
