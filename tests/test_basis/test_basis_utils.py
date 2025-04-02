import numpy as np
import pytest

from ttfemesh.basis.basis_utils import (
    left_corner2index_map_ttcores,
    left_corner2index_ttmap,
    right_corner2index_map_ttcores,
    right_corner2index_ttmap,
)


class TestLeftCorner2IndexMapTTCores:
    def test_cores_have_correct_shapes(self):
        firstcore, middlecore, lastcore = left_corner2index_map_ttcores()
        assert firstcore.shape == (1, 2, 2, 2)
        assert middlecore.shape == (2, 2, 2, 2)
        assert lastcore.shape == (2, 2, 2, 1)

    def test_firstcore_has_correct_values(self):
        firstcore, _, _ = left_corner2index_map_ttcores()
        expected_firstcore = np.zeros((1, 2, 2, 2))
        expected_firstcore[0, :, :, 0] = np.array([[1.0, 0.0], [0.0, 1.0]])
        expected_firstcore[0, :, :, 1] = np.array([[1.0, 0.0], [0.0, 0.0]])
        assert np.array_equal(firstcore, expected_firstcore)

    def test_middlecore_has_correct_values(self):
        _, middlecore, _ = left_corner2index_map_ttcores()
        expected_middlecore = np.zeros((2, 2, 2, 2))
        expected_middlecore[0, :, :, 0] = np.array([[1.0, 0.0], [0.0, 1.0]])
        expected_middlecore[0, :, :, 1] = np.array([[1.0, 0.0], [0.0, 0.0]])
        expected_middlecore[1, :, :, 0] = np.array([[0.0, 0.0], [0.0, 0.0]])
        expected_middlecore[1, :, :, 1] = np.array([[0.0, 0.0], [0.0, 1.0]])
        assert np.array_equal(middlecore, expected_middlecore)

    def test_lastcore_has_correct_values(self):
        _, _, lastcore = left_corner2index_map_ttcores()
        expected_lastcore = np.zeros((2, 2, 2, 1))
        expected_lastcore[0, :, :, 0] = np.array([[1.0, 0.0], [0.0, 0.0]])
        expected_lastcore[1, :, :, 0] = np.array([[0.0, 0.0], [0.0, 1.0]])
        assert np.array_equal(lastcore, expected_lastcore)


class TestRightCorner2IndexMapTTCores:
    def test_cores_have_correct_shapes(self):
        firstcore, middlecore, lastcore = right_corner2index_map_ttcores()
        assert firstcore.shape == (1, 2, 2, 2)
        assert middlecore.shape == (2, 2, 2, 2)
        assert lastcore.shape == (2, 2, 2, 1)

    def test_firstcore_has_correct_values(self):
        firstcore, _, _ = right_corner2index_map_ttcores()
        expected_firstcore = np.zeros((1, 2, 2, 2))
        expected_firstcore[0, :, :, 0] = np.array([[0.0, 1.0], [0.0, 0.0]])
        expected_firstcore[0, :, :, 1] = np.array([[0.0, 0.0], [1.0, 0.0]])
        assert np.array_equal(firstcore, expected_firstcore)

    def test_middlecore_has_correct_values(self):
        _, middlecore, _ = right_corner2index_map_ttcores()
        expected_middlecore = np.zeros((2, 2, 2, 2))
        expected_middlecore[0, :, :, 0] = np.array([[0.0, 1.0], [0.0, 0.0]])
        expected_middlecore[0, :, :, 1] = np.array([[0.0, 0.0], [1.0, 0.0]])
        expected_middlecore[1, :, :, 0] = np.array([[0.0, 0.0], [0.0, 0.0]])
        expected_middlecore[1, :, :, 1] = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert np.array_equal(middlecore, expected_middlecore)

    def test_lastcore_has_correct_values(self):
        _, _, lastcore = right_corner2index_map_ttcores()
        expected_lastcore = np.zeros((2, 2, 2, 1))
        expected_lastcore[0, :, :, 0] = np.array([[0.0, 0.0], [1.0, 0.0]])
        expected_lastcore[1, :, :, 0] = np.array([[1.0, 0.0], [0.0, 1.0]])
        assert np.array_equal(lastcore, expected_lastcore)


class TestLeftCorner2IndexTTMap:
    def test_ttmap_has_correct_shape_for_mesh_size_exponent_2(self):
        ttmap = left_corner2index_ttmap(2)
        assert ttmap.shape == [(2, 2), (2, 2)]

    def test_ttmap_has_correct_shape_for_mesh_size_exponent_3(self):
        ttmap = left_corner2index_ttmap(3)
        assert ttmap.shape == [(2, 2), (2, 2), (2, 2)]

    def test_ttmap_has_correct_values_for_mesh_size_exponent_2(self):
        ttmap = left_corner2index_ttmap(2)
        size = 4
        W0_reshaped = np.reshape(ttmap.full(), (-1, size), order="F")
        expected_W0 = np.eye(size, dtype=float)
        expected_W0[-1, -1] = 0
        assert np.array_equal(W0_reshaped, expected_W0)


class TestRightCorner2IndexTTMap:
    def test_ttmap_has_correct_shape_for_mesh_size_exponent_2(self):
        ttmap = right_corner2index_ttmap(2)
        assert ttmap.shape == [(2, 2), (2, 2)]

    def test_ttmap_has_correct_shape_for_mesh_size_exponent_3(self):
        ttmap = right_corner2index_ttmap(3)
        assert ttmap.shape == [(2, 2), (2, 2), (2, 2)]

    def test_ttmap_has_correct_values_for_mesh_size_exponent_2(self):
        ttmap = right_corner2index_ttmap(2)
        size = 4
        W1_reshaped = np.reshape(ttmap.full(), (-1, size), order="F")
        expected_W1 = np.zeros((size, size), dtype=float)
        np.fill_diagonal(expected_W1[1:], 1)
        expected_W1[-1, -1] = 0
        assert np.array_equal(W1_reshaped, expected_W1)
