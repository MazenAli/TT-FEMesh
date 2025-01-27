import numpy as np
import pytest

from ttfemesh.basis.basis import BilinearBasis, LinearBasis


class TestLinearBasis:
    @pytest.fixture(autouse=True)
    def setup_basis(self):
        self.basis = LinearBasis()

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

    def test_element2global_ttmap_is_correct(self):
        ttmap = self.basis.get_element2global_ttmap(0, 3)
        assert ttmap.shape == (8, 8)

    def test_all_element2global_ttmaps_are_correct(self):
        ttmaps = self.basis.get_all_element2global_ttmaps(3)
        assert len(ttmaps) == 2
        assert ttmaps[0].shape == (8, 8)
        assert ttmaps[1].shape == (8, 8)

    def test_dirichlet_mask_left_is_correct(self):
        mask = self.basis.get_dirichlet_mask_left(3)
        assert mask.shape == (8, 8)
        assert np.all(mask[:, 0] == 0)

    def test_dirichlet_mask_right_is_correct(self):
        mask = self.basis.get_dirichlet_mask_right(3)
        assert mask.shape == (8, 8)
        assert np.all(mask[:, -1] == 0)

    def test_dirichlet_mask_left_right_is_correct(self):
        mask = self.basis.get_dirichlet_mask_left_right(3)
        assert mask.shape == (8, 8)
        assert np.all(mask[:, 0] == 0)
        assert np.all(mask[:, -1] == 0)


class TestBilinearBasis:
    @pytest.fixture(autouse=True)
    def setup_basis(self):
        self.basis = BilinearBasis()

    def test_evaluates_correctly(self):
        assert self.basis.evaluate((0, 0), (-1, -1)) == pytest.approx(1.0)
        assert self.basis.evaluate((0, 1), (-1, 1)) == pytest.approx(1.0)
        assert self.basis.evaluate((1, 0), (1, -1)) == pytest.approx(1.0)
        assert self.basis.evaluate((1, 1), (1, 1)) == pytest.approx(1.0)

    def test_raises_error_for_invalid_index(self):
        with pytest.raises(ValueError):
            self.basis.evaluate((2, 0), (0, 0))

    def test_index_range_is_correct(self):
        assert [list(r) for r in self.basis.index_range] == [[0, 1], [0, 1]]

    def test_dirichlet_mask_is_correct(self):
        mask = self.basis.get_dirichlet_mask(2, 0, 1, 2, 3)
        assert mask.shape == (4, 4)
        assert np.all(mask == 0)
