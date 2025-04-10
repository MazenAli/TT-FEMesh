import pytest
import numpy as np
import torchtt as tntt
from ttfemesh.tt_tools.meshgrid import zmeshgrid2d, range_meshgrid2d, map2canonical2d
from ttfemesh.types import TensorTrain


class TestZMeshGrid2D:
    def test_constant_tensors(self):
        d = 3
        X = tntt.ones([2] * d)
        Y = tntt.ones([2] * d)
        
        XX, YY = zmeshgrid2d(X, Y)
        
        assert isinstance(XX, TensorTrain)
        assert isinstance(YY, TensorTrain)
        assert np.allclose(XX.full(), 1.0)
        assert np.allclose(YY.full(), 1.0)

    def test_range_tensors(self):
        d = 3
        X = tntt._extras.xfun([2] * d)
        Y = tntt._extras.xfun([2] * d)
        
        XX, YY = zmeshgrid2d(X, Y)
        
        assert isinstance(XX, TensorTrain)
        assert isinstance(YY, TensorTrain)
        
        zmap = map2canonical2d(d)
        full_XX = np.array(XX.full()).flatten("F")
        full_YY = np.array(YY.full()).flatten("F")
        
        reordered_XX = np.empty_like(full_XX)
        reordered_YY = np.empty_like(full_YY)
        reordered_XX[zmap] = full_XX
        reordered_YY[zmap] = full_YY
        
        final_XX = reordered_XX.reshape((2**d, 2**d), order="F")
        final_YY = reordered_YY.reshape((2**d, 2**d), order="F")
        
        for j in range(2**d):
            assert np.allclose(final_XX[:, j], np.arange(2**d))
        for i in range(2**d):
            assert np.allclose(final_YY[i, :], np.arange(2**d))


class TestRangeMeshGrid2D:
    def test_small_exponent(self):
        """Test range meshgrid with small exponent."""
        d = 2
        XX, YY = range_meshgrid2d(d)
        
        assert isinstance(XX, TensorTrain)
        assert isinstance(YY, TensorTrain)
        
        zmap = map2canonical2d(d)
        full_XX = np.array(XX.full()).flatten("F")
        full_YY = np.array(YY.full()).flatten("F")
        
        reordered_XX = np.empty_like(full_XX)
        reordered_YY = np.empty_like(full_YY)
        reordered_XX[zmap] = full_XX
        reordered_YY[zmap] = full_YY
        
        final_XX = reordered_XX.reshape((2**d, 2**d), order="F")
        final_YY = reordered_YY.reshape((2**d, 2**d), order="F")
        
        for j in range(2**d):
            assert np.allclose(final_XX[:, j], np.arange(2**d))
        for i in range(2**d):
            assert np.allclose(final_YY[i, :], np.arange(2**d))

    def test_larger_exponent(self):
        d = 4
        XX, YY = range_meshgrid2d(d)
        
        assert isinstance(XX, TensorTrain)
        assert isinstance(YY, TensorTrain)
        
        zmap = map2canonical2d(d)
        full_XX = np.array(XX.full()).flatten("F")
        full_YY = np.array(YY.full()).flatten("F")
        
        reordered_XX = np.empty_like(full_XX)
        reordered_YY = np.empty_like(full_YY)
        reordered_XX[zmap] = full_XX
        reordered_YY[zmap] = full_YY
        
        final_XX = reordered_XX.reshape((2**d, 2**d), order="F")
        final_YY = reordered_YY.reshape((2**d, 2**d), order="F")
        
        for j in range(2**d):
            assert np.allclose(final_XX[:, j], np.arange(2**d))
        for i in range(2**d):
            assert np.allclose(final_YY[i, :], np.arange(2**d))


class TestMap2Canonical2D:
    def test_small_exponent(self):
        d = 2
        zmap = map2canonical2d(d)
        
        assert isinstance(zmap, np.ndarray)
        assert zmap.shape == (2**(2*d),)
        
        assert len(np.unique(zmap)) == 2**(2*d)
        assert np.min(zmap) == 0
        assert np.max(zmap) == 2**(2*d) - 1
        
        expected = np.array([0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15])
        assert np.allclose(zmap, expected)

    def test_larger_exponent(self):
        d = 3
        zmap = map2canonical2d(d)
        
        assert isinstance(zmap, np.ndarray)
        assert zmap.shape == (2**(2*d),)
        
        assert len(np.unique(zmap)) == 2**(2*d)
        assert np.min(zmap) == 0
        assert np.max(zmap) == 2**(2*d) - 1
        
        for i in range(0, 2**(2*d), 4):
            assert zmap[i] < zmap[i+1]
            assert zmap[i] < zmap[i+2]
            assert zmap[i+1] < zmap[i+3]
            assert zmap[i+2] < zmap[i+3] 