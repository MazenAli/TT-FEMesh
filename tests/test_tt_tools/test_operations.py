import pytest
import torch
import numpy as np
from ttfemesh.tt_tools.meshgrid import map2canonical2d, range_meshgrid2d
from ttfemesh.tt_tools.operations import zorder_kron, zorder_linfunc2d
from ttfemesh.types import TensorTrain
import torchtt as tt


class TestZorderKron:
    def test_basic_kron(self):
        core = torch.tensor([[1., 0.], [0., 1.]]).reshape(1, 2, 2, 1)
        cores = [core]*3
        left = TensorTrain(cores)
        right = TensorTrain(cores)
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
    def test_invalid_rank_X(self):
        core = torch.tensor([[1., 2.]]).reshape(1, 2, 1)
        cores = [core]*3
        X = TensorTrain(cores)
        Y = TensorTrain(cores)
        
        c = 1.0
        cx = 2.0
        cy = 3.0
        
        with pytest.raises(ValueError):
            zorder_linfunc2d(c, cx, X, cy, Y)

    def test_invalid_rank_Y(self):
        core = torch.tensor([[1., 2.]]).reshape(1, 2, 1)
        cores = [core]*3
        X, _ = range_meshgrid2d(3)
        Y = TensorTrain(cores)
        
        c = 1.0
        cx = 2.0
        cy = 3.0
        
        with pytest.raises(ValueError):
            zorder_linfunc2d(c, cx, X, cy, Y)
        

    def test_invalid_length(self):
        core = torch.tensor([[1., 2.]]).reshape(1, 2, 1)
        cores = [core]*3
        X = TensorTrain(cores)

        cores = [core]*2
        Y = TensorTrain(cores)
        
        with pytest.raises(ValueError):
            zorder_linfunc2d(1.0, 0.0, X, 0.0, Y)


    def test_meshgrid_example(self):
        XX, YY = range_meshgrid2d(3)
        result = zorder_linfunc2d(1.0, 1.0, XX, 1.0, YY)
        assert isinstance(result, TensorTrain)
        assert len(result.cores) == 3
        assert result.cores[0].shape == (1, 4, 2)
        assert result.cores[1].shape == (2, 4, 2)
        assert result.cores[2].shape == (2, 4, 1)

        expected = np.empty((2**3, 2**3))
        for i in range(2**3):
            for j in range(2**3):
                expected[i, j] = 1.0 + 1.0 * i + 1.0 * j

        zmap = map2canonical2d(3)
        result_full = np.empty(4**3)
        result_full[zmap] = np.array(result.full()).flatten("F")
        result_full = result_full.reshape((2**3, 2**3), order="F")

        assert np.allclose(result_full, expected)
