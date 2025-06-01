import numpy as np
import pytest

from ttfemesh.utils.array import ensure_1d, ensure_2d


class TestEnsure1D:
    def test_scalar_input(self):
        result = ensure_1d(5)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result[0] == 5

        result = ensure_1d(3.14)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result[0] == 3.14

    def test_1d_array_input(self):
        input_array = np.array([1, 2, 3])
        result = ensure_1d(input_array)
        assert result is input_array
        assert result.shape == (3,)

        input_list = [1, 2, 3]
        result = ensure_1d(input_list)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.array_equal(result, np.array([1, 2, 3]))

    def test_invalid_input(self):
        with pytest.raises(ValueError, match="Input must be a scalar or a 1D array"):
            ensure_1d(np.array([[1, 2], [3, 4]]))

        with pytest.raises(ValueError, match="Input must be a scalar or a 1D array"):
            ensure_1d(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))


class TestEnsure2D:
    def test_1d_array_input(self):
        input_array = np.array([1, 2, 3])
        result = ensure_2d(input_array)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)
        assert np.array_equal(result, np.array([[1, 2, 3]]))

        input_list = [1, 2, 3]
        result = ensure_2d(input_list)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)
        assert np.array_equal(result, np.array([[1, 2, 3]]))

    def test_2d_array_input(self):
        input_array = np.array([[1, 2, 3], [4, 5, 6]])
        result = ensure_2d(input_array)
        assert result is input_array
        assert result.shape == (2, 3)

        input_list = [[1, 2, 3], [4, 5, 6]]
        result = ensure_2d(input_list)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        assert np.array_equal(result, np.array([[1, 2, 3], [4, 5, 6]]))

    def test_invalid_input(self):
        with pytest.raises(ValueError, match="Input array must have 1 or 2 dimensions"):
            ensure_2d(np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))

        with pytest.raises(ValueError, match="Input array must have 1 or 2 dimensions"):
            ensure_2d(5)
