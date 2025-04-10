import pytest
import torch
import numpy as np
from ttfemesh.tt_tools.meshgrid import map2canonical2d
from ttfemesh.tt_tools.operations import zorder_kron, zorder_linfunc2d
from ttfemesh.types import TensorTrain
import torchtt as tt


@pytest.fixture
def create_tt_matrix():
    core = torch.tensor([[1., 0.], [0., 1.]]).reshape(1, 2, 2, 1)
    cores = [core]*3
    return TensorTrain(cores)


@pytest.fixture
def create_tt_vector():
    core = torch.tensor([[1., 2.]]).reshape(1, 2, 1)
    cores = [core]*3
    return TensorTrain(cores)


class TestZorderKron:
    def test_basic_kron(self, create_tt_matrix):
        left = create_tt_matrix
        right = create_tt_matrix
        
        result = zorder_kron(left, right)
        assert isinstance(result, TensorTrain)
        assert len(result.cores) == 3

    def test_invalid_length(self):
        left_cores = [torch.tensor([[[1.0]]])]
        right_cores = [torch.tensor([[[2.0]]]), torch.tensor([[[3.0]]])]
        left = TensorTrain(left_cores)
        right = TensorTrain(right_cores)
        
        with pytest.raises(ValueError):
            zorder_kron(left, right)

    def test_kron_random(self):
        left_tt = tt.randn([2, 2, 2], [1, 2, 2, 1])
        right_tt = tt.randn([2, 2, 2], [1, 2, 2, 1])

        result = zorder_kron(left_tt, right_tt)
        assert isinstance(result, tt.TT)
        assert result.R == [1, 4, 4, 1]
        assert result.shape == [4, 4, 4]

        left_tt_full = np.array(left_tt.full()).flatten("F")
        right_tt_full = np.array(right_tt.full()).flatten("F")

        expected = np.kron(right_tt_full, left_tt_full)
        zmap = map2canonical2d(3)
        result_full = np.empty_like(expected)
        result_full[zmap] = np.array(result.full()).flatten("F")

        assert np.allclose(result_full, expected)


class TestZorderLinfunc2d:
    def test_basic_linfunc(self, create_tt_vector):
        X = create_tt_vector
        Y = create_tt_vector
        
        c = 1.0
        cx = 2.0
        cy = 3.0
        
        result = zorder_linfunc2d(c, cx, X, cy, Y) # does not work for rank 1?
        assert isinstance(result, TensorTrain)
        assert len(result.cores) == 3

    def test_multi_core_linfunc(self):
        # Create TT-tensors with multiple cores
        X_cores = [
            torch.tensor([[[1.0], [2.0]]]),
            torch.tensor([[[3.0]], [[4.0]]])
        ]
        Y_cores = [
            torch.tensor([[[5.0], [6.0]]]),
            torch.tensor([[[7.0]], [[8.0]]])
        ]
        X = TensorTrain(X_cores)
        Y = TensorTrain(Y_cores)
        
        c = 1.0
        cx = 2.0
        cy = 3.0
        
        result = zorder_linfunc2d(c, cx, X, cy, Y)
        assert len(result.cores) == 2
        
        # Check first core
        expected_core1 = torch.tensor([[[2.0 + 15.0], [4.0 + 18.0]]])
        assert torch.allclose(result.cores[0], expected_core1)
        
        # Check second core
        expected_core2 = torch.tensor([[[6.0 + 21.0]], [[8.0 + 24.0 + 1.0]]])
        assert torch.allclose(result.cores[1], expected_core2)

    def test_zero_coefficients(self):
        X_cores = [torch.tensor([[[1.0]], [[2.0]]])]
        Y_cores = [torch.tensor([[[3.0]], [[4.0]]])]
        X = TensorTrain(X_cores)
        Y = TensorTrain(Y_cores)
        
        # Test with zero coefficients
        result = zorder_linfunc2d(1.0, 0.0, X, 0.0, Y)
        expected = torch.tensor([[[0.0]], [[1.0]]])
        assert torch.allclose(result.cores[0], expected) 