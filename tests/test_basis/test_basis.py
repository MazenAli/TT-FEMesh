import numpy as np
import pytest

from ttfemesh.basis.basis import BilinearBasis, LinearBasis


class TestLinearBasis:
    @pytest.fixture(autouse=True)
    def setup_basis(self):
        self.basis = LinearBasis()

    def test_dimension_is_correct(self):
        assert self.basis.dimension == 1

    def test_evaluates_correctly(self):
        assert self.basis.evaluate(0, -1) == pytest.approx(1.0)
        assert self.basis.evaluate(0, 0) == pytest.approx(0.5)
        assert self.basis.evaluate(0, 1) == pytest.approx(0.0)
        assert self.basis.evaluate(1, -1) == pytest.approx(0.0)
        assert self.basis.evaluate(1, 0) == pytest.approx(0.5)
        assert self.basis.evaluate(1, 1) == pytest.approx(1.0)

    def test_raises_error_for_invalid_index(self):
        with pytest.raises(ValueError):
            self.basis.evaluate(2, 0)

    def test_derivative_is_correct(self):
        assert self.basis.derivative(0) == pytest.approx(-0.5)
        assert self.basis.derivative(1) == pytest.approx(0.5)

    def test_index_range_is_correct(self):
        assert list(self.basis.index_range) == [0, 1]

    def test_left_element2global_ttmap_is_correct(self):
        ttmap = self.basis.get_element2global_ttmap(0, 3)
        size = 8
        W0_reshaped = np.reshape(ttmap.full(), (-1, size), order="F")
        expected_W0 = np.eye(size, dtype=float)
        expected_W0[-1, -1] = 0
        assert np.array_equal(W0_reshaped, expected_W0)
        assert ttmap.shape == [(2, 2), (2, 2), (2, 2)]

    def test_right_element2global_ttmap_is_correct(self):
        ttmap = self.basis.get_element2global_ttmap(1, 3)
        size = 8
        W1_reshaped = np.reshape(ttmap.full(), (-1, size), order="F")
        expected_W1 = np.zeros((size, size), dtype=float)
        np.fill_diagonal(expected_W1[1:], 1)
        expected_W1[-1, -1] = 0
        assert np.array_equal(W1_reshaped, expected_W1)
        assert ttmap.shape == [(2, 2), (2, 2), (2, 2)]

    def test_all_element2global_ttmaps_are_correct(self):
        ttmaps = self.basis.get_all_element2global_ttmaps(3)

        size = 8
        expected_W0 = np.eye(size, dtype=float)
        expected_W0[-1, -1] = 0

        expected_W1 = np.zeros((size, size), dtype=float)
        np.fill_diagonal(expected_W1[1:], 1)
        expected_W1[-1, -1] = 0

        assert len(ttmaps) == 2
        assert ttmaps[0].shape == [(2, 2), (2, 2), (2, 2)]
        assert ttmaps[1].shape == [(2, 2), (2, 2), (2, 2)]

        W0_reshaped = np.reshape(ttmaps[0].full(), (-1, size), order="F")
        W1_reshaped = np.reshape(ttmaps[1].full(), (-1, size), order="F")
        assert np.array_equal(W0_reshaped, expected_W0)
        assert np.array_equal(W1_reshaped, expected_W1)

    def test_dirichlet_mask_left_is_correct(self):
        mask = self.basis.get_dirichlet_mask_left(3)
        vector = np.array(mask.full()).flatten("F")
        assert mask.shape == [2, 2, 2]
        assert vector[0] == 0.
        assert np.all(vector[1:] == 1.)

    def test_dirichlet_mask_right_is_correct(self):
        mask = self.basis.get_dirichlet_mask_right(3)
        vector = np.array(mask.full()).flatten("F")
        assert mask.shape == [2, 2, 2]
        assert vector[-1] == 0.
        assert np.all(vector[:-1] == 1.)

    def test_dirichlet_mask_left_right_is_correct(self):
        mask = self.basis.get_dirichlet_mask_left_right(3)
        vector = np.array(mask.full()).flatten("F")
        assert mask.shape == [2, 2, 2]
        assert vector[0] == 0.
        assert vector[-1] == 0.
        assert np.all(vector[1:-1] == 1.)


class TestBilinearBasis:
    @pytest.fixture(autouse=True)
    def setup_basis(self):
        self.basis = BilinearBasis()

    def test_dimension_is_correct(self):
        assert self.basis.dimension == 2

    def test_raises_error_for_invalid_index(self):
        with pytest.raises(ValueError):
            self.basis.evaluate((2, 0), (0, 0))

    def test_evaluates_correctly(self):
        assert self.basis.evaluate((0, 0), (-1, -1)) == pytest.approx(1.0)
        assert self.basis.evaluate((0, 1), (-1, 1)) == pytest.approx(1.0)
        assert self.basis.evaluate((1, 0), (1, -1)) == pytest.approx(1.0)
        assert self.basis.evaluate((1, 1), (1, 1)) == pytest.approx(1.0)
        assert self.basis.evaluate((0, 0), (0, 0)) == pytest.approx(0.25)
        assert self.basis.evaluate((0, 1), (0, 0)) == pytest.approx(0.25)
        assert self.basis.evaluate((1, 0), (0, 0)) == pytest.approx(0.25)
        assert self.basis.evaluate((1, 1), (0, 0)) == pytest.approx(0.25)
        assert self.basis.evaluate((0, 1), (0, 1)) == pytest.approx(0.5)
        assert self.basis.evaluate((1, 0), (1, 0)) == pytest.approx(0.5)

    def test_derivative_is_correct(self):
        assert self.basis.derivative((0, 0), (-1, -1), 0) == pytest.approx(-0.5)
        assert self.basis.derivative((0, 0), (-1, -1), 1) == pytest.approx(-0.5)
        assert self.basis.derivative((0, 1), (-1, -1), 0) == pytest.approx(0.)
        assert self.basis.derivative((0, 1), (-1, 1), 1) == pytest.approx(0.5)
        assert self.basis.derivative((1, 0), (0, 0), 0) == pytest.approx(0.25)
        assert self.basis.derivative((1, 0), (0, 0), 1) == pytest.approx(-0.25)

    def test_raises_error_for_invalid_index(self):
        with pytest.raises(ValueError):
            self.basis.evaluate((2, 0), (0, 0))

    def test_index_range_is_correct(self):
        assert [list(r) for r in self.basis.index_range] == [[0, 1], [0, 1]]

    # TODO: index map tests
    def test_left_element2global_ttmap_is_correct(self):
        ttmap = self.basis.get_element2global_ttmap((0, 0), 3)
        size = 8
        W00_reshaped = np.reshape(ttmap.full(), (-1, size), order="F")
        expected_W00 = np.eye(size, dtype=float)
        expected_W00[-1, -1] = 0
        assert np.array_equal(W00_reshaped, expected_W00)
        assert ttmap.shape == [(2, 2), (2, 2), (2, 2)]

    def test_dirichlet_mask_is_correct(self):
        mask = self.basis.get_dirichlet_mask(2, 0, 1, 2, 3)
        assert mask.shape == [4, 4]
