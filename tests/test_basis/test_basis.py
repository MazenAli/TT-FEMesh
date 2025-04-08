import numpy as np
import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

# Set the matplotlib backend to Agg for testing
import matplotlib
matplotlib.use('Agg')

# Configure matplotlib for consistent test output
matplotlib.rcParams.update({
    'figure.dpi': 100,
    'figure.figsize': (6, 6),
    'font.family': 'sans-serif',
    'font.size': 10,
    'text.kerning_factor': 0,
    'image.cmap': 'viridis',
    'lines.linestyle': '-',
    'lines.linewidth': 1.0,
    'axes.grid': False,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

from ttfemesh.basis.basis import BilinearBasis, LinearBasis
from ttfemesh.tn_tools.meshgrid import map2canonical2d
from ttfemesh.types import BoundarySide2D


class TestLinearBasis:
    @pytest.fixture(autouse=True)  # noqa
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
        assert vector[0] == 0.0
        assert np.all(vector[1:] == 1.0)

    def test_dirichlet_mask_right_is_correct(self):
        mask = self.basis.get_dirichlet_mask_right(3)
        vector = np.array(mask.full()).flatten("F")
        assert mask.shape == [2, 2, 2]
        assert vector[-1] == 0.0
        assert np.all(vector[:-1] == 1.0)

    def test_dirichlet_mask_left_right_is_correct(self):
        mask = self.basis.get_dirichlet_mask_left_right(3)
        vector = np.array(mask.full()).flatten("F")
        assert mask.shape == [2, 2, 2]
        assert vector[0] == 0.0
        assert vector[-1] == 0.0
        assert np.all(vector[1:-1] == 1.0)

    def test_linear_plot_creates_correct_figure(self):
        plt.clf() 
        fig = self.basis.plot(0) 
        plt.title("Linear Basis Function")
        plt.xlabel("x")
        plt.ylabel("y")
        
        assert fig is not None
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_title() == "Linear Basis Function"
        assert ax.get_xlabel() == "x"
        assert ax.get_ylabel() == "y"
        assert len(ax.lines) == 1
        
        plt.close(fig)

    def test_plot_raises_error_for_invalid_index(self):
        with pytest.raises(ValueError):
            self.basis.plot(2)

    def test_repr_returns_correct_string(self):
        assert repr(self.basis) == "LinearBasis"


class TestBilinearBasis:
    @pytest.fixture(autouse=True)  # noqa
    def setup_basis(self):
        self.basis = BilinearBasis()

    def test_dimension_is_correct(self):
        assert self.basis.dimension == 2

    def test_raises_error_for_invalid_index(self):
        with pytest.raises(ValueError):
            self.basis.evaluate((2, 0), (0, 0))

    def test_raises_error_for_invalid_index_length(self):
        with pytest.raises(ValueError):
            self.basis.evaluate((2, 0, 0), (0, 0))

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
        assert self.basis.derivative((0, 1), (-1, -1), 0) == pytest.approx(0.0)
        assert self.basis.derivative((0, 1), (-1, 1), 1) == pytest.approx(0.5)
        assert self.basis.derivative((1, 0), (0, 0), 0) == pytest.approx(0.25)
        assert self.basis.derivative((1, 0), (0, 0), 1) == pytest.approx(-0.25)

    def test_derivative_raises_error_for_invalid_dimension(self):
        with pytest.raises(ValueError, match="Invalid dimension index: 2, expected 0 <= dim < 2"):
            self.basis.derivative((0, 0), (-1, -1), 2)
        with pytest.raises(ValueError, match="Invalid dimension index: -1, expected 0 <= dim < 2"):
            self.basis.derivative((0, 0), (-1, -1), -1)

    def test_index_range_is_correct(self):
        assert [list(r) for r in self.basis.index_range] == [[0, 1], [0, 1]]

    def test_element2global_ttmap_is_correct(self):
        mesh_size_exponent = 2
        zmap = map2canonical2d(mesh_size_exponent)

        size = 4**mesh_size_exponent
        new_size = 2**mesh_size_exponent

        def reshape_ttmap(ttmap):
            ttmap_reshape = np.array(ttmap.full()).reshape((size, size), order="F")
            ttmap_reordered = np.empty_like(ttmap_reshape)
            ttmap_reordered[np.ix_(zmap, zmap)] = ttmap_reshape
            ttmap_reordered_reshaped = ttmap_reordered.reshape(
                (new_size, new_size, new_size, new_size), order="F"
            )
            return ttmap_reordered_reshaped

        W00 = self.basis.get_element2global_ttmap((0, 0), mesh_size_exponent)
        W10 = self.basis.get_element2global_ttmap((1, 0), mesh_size_exponent)
        W01 = self.basis.get_element2global_ttmap((0, 1), mesh_size_exponent)
        W11 = self.basis.get_element2global_ttmap((1, 1), mesh_size_exponent)

        W00_reordered = reshape_ttmap(W00)
        W10_reordered = reshape_ttmap(W10)
        W01_reordered = reshape_ttmap(W01)
        W11_reordered = reshape_ttmap(W11)

        expected_W00 = np.zeros((new_size, new_size, new_size, new_size), dtype=float)
        expected_W00[0, 0, 0, 0] = 1.0
        expected_W00[0, 1, 0, 1] = 1.0
        expected_W00[0, 2, 0, 2] = 1.0
        expected_W00[1, 0, 1, 0] = 1.0
        expected_W00[1, 1, 1, 1] = 1.0
        expected_W00[1, 2, 1, 2] = 1.0
        expected_W00[2, 0, 2, 0] = 1.0
        expected_W00[2, 1, 2, 1] = 1.0
        expected_W00[2, 2, 2, 2] = 1.0

        expected_W10 = np.zeros((new_size, new_size, new_size, new_size), dtype=float)
        expected_W10[1, 0, 0, 0] = 1.0
        expected_W10[1, 1, 0, 1] = 1.0
        expected_W10[1, 2, 0, 2] = 1.0
        expected_W10[2, 0, 1, 0] = 1.0
        expected_W10[2, 1, 1, 1] = 1.0
        expected_W10[2, 2, 1, 2] = 1.0
        expected_W10[3, 0, 2, 0] = 1.0
        expected_W10[3, 1, 2, 1] = 1.0
        expected_W10[3, 2, 2, 2] = 1.0

        expected_W01 = np.zeros((new_size, new_size, new_size, new_size), dtype=float)
        expected_W01[0, 1, 0, 0] = 1.0
        expected_W01[0, 2, 0, 1] = 1.0
        expected_W01[0, 3, 0, 2] = 1.0
        expected_W01[1, 1, 1, 0] = 1.0
        expected_W01[1, 2, 1, 1] = 1.0
        expected_W01[1, 3, 1, 2] = 1.0
        expected_W01[2, 1, 2, 0] = 1.0
        expected_W01[2, 2, 2, 1] = 1.0
        expected_W01[2, 3, 2, 2] = 1.0

        expected_W11 = np.zeros((new_size, new_size, new_size, new_size), dtype=float)
        expected_W11[1, 1, 0, 0] = 1.0
        expected_W11[1, 2, 0, 1] = 1.0
        expected_W11[1, 3, 0, 2] = 1.0
        expected_W11[2, 1, 1, 0] = 1.0
        expected_W11[2, 2, 1, 1] = 1.0
        expected_W11[2, 3, 1, 2] = 1.0
        expected_W11[3, 1, 2, 0] = 1.0
        expected_W11[3, 2, 2, 1] = 1.0
        expected_W11[3, 3, 2, 2] = 1.0

        assert W00.shape == [(4, 4), (4, 4)]
        assert W10.shape == [(4, 4), (4, 4)]
        assert W01.shape == [(4, 4), (4, 4)]
        assert W11.shape == [(4, 4), (4, 4)]

        assert np.array_equal(W00_reordered, expected_W00)
        assert np.array_equal(W10_reordered, expected_W10)
        assert np.array_equal(W01_reordered, expected_W01)
        assert np.array_equal(W11_reordered, expected_W11)

    def test_all_element2global_ttmaps_are_correct(self):
        mesh_size_exponent = 2
        ttmaps = self.basis.get_all_element2global_ttmaps(mesh_size_exponent)
        W00_expected = self.basis.get_element2global_ttmap((0, 0), mesh_size_exponent)
        W10_expected = self.basis.get_element2global_ttmap((1, 0), mesh_size_exponent)
        W01_expected = self.basis.get_element2global_ttmap((0, 1), mesh_size_exponent)
        W11_expected = self.basis.get_element2global_ttmap((1, 1), mesh_size_exponent)

        assert np.array_equal(np.array(ttmaps[0, 0].full()), np.array(W00_expected.full()))
        assert np.array_equal(np.array(ttmaps[1, 0].full()), np.array(W10_expected.full()))
        assert np.array_equal(np.array(ttmaps[0, 1].full()), np.array(W01_expected.full()))
        assert np.array_equal(np.array(ttmaps[1, 1].full()), np.array(W11_expected.full()))

    def test_dirichlet_mask_raises_error(self):
        with pytest.raises(ValueError):
            self.basis.get_dirichlet_mask(3)

    def test_dirichlet_mask_is_correct(self):
        mesh_size_exponent = 3
        mask_left = self.basis.get_dirichlet_mask(mesh_size_exponent, BoundarySide2D.LEFT)
        mask_bottom = self.basis.get_dirichlet_mask(mesh_size_exponent, BoundarySide2D.BOTTOM)
        mask_right_top = self.basis.get_dirichlet_mask(
            mesh_size_exponent, BoundarySide2D.RIGHT, BoundarySide2D.TOP
        )
        mask_left_right = self.basis.get_dirichlet_mask(
            mesh_size_exponent, BoundarySide2D.LEFT, BoundarySide2D.RIGHT
        )
        mask_bottom_top = self.basis.get_dirichlet_mask(
            mesh_size_exponent, BoundarySide2D.BOTTOM, BoundarySide2D.TOP
        )
        

        zmap = map2canonical2d(mesh_size_exponent)

        def reshape_ttvec(ttvec):
            size = 4**mesh_size_exponent
            reshaped = np.array(ttvec.full()).reshape((size), order="F")
            reordered = np.empty_like(reshaped)
            reordered[zmap] = reshaped
            W = reordered.reshape((2**mesh_size_exponent, 2**mesh_size_exponent), order="F")

            return W

        mask_left_full = reshape_ttvec(mask_left)
        mask_bottom_full = reshape_ttvec(mask_bottom)
        mask_right_top_full = reshape_ttvec(mask_right_top)
        mask_left_right_full = reshape_ttvec(mask_left_right)
        mask_bottom_top_full = reshape_ttvec(mask_bottom_top)

        expected_mask_left = np.ones((2**mesh_size_exponent, 2**mesh_size_exponent), dtype=float)
        expected_mask_left[0, :] = 0.0

        expected_mask_bottom = np.ones((2**mesh_size_exponent, 2**mesh_size_exponent), dtype=float)
        expected_mask_bottom[:, 0] = 0.0

        expected_mask_right_top = np.ones(
            (2**mesh_size_exponent, 2**mesh_size_exponent), dtype=float
        )
        expected_mask_right_top[-1, :] = 0.0
        expected_mask_right_top[:, -1] = 0.0

        expected_mask_left_right = np.ones(
            (2**mesh_size_exponent, 2**mesh_size_exponent), dtype=float
        )
        expected_mask_left_right[0, :] = 0.0
        expected_mask_left_right[-1, :] = 0.0

        expected_mask_bottom_top = np.ones(
            (2**mesh_size_exponent, 2**mesh_size_exponent), dtype=float
        )
        expected_mask_bottom_top[:, 0] = 0.0
        expected_mask_bottom_top[:, -1] = 0.0

        assert mask_left.shape == [4] * mesh_size_exponent
        assert mask_bottom.shape == [4] * mesh_size_exponent
        assert mask_right_top.shape == [4] * mesh_size_exponent
        assert mask_left_right.shape == [4] * mesh_size_exponent
        assert mask_bottom_top.shape == [4] * mesh_size_exponent
        assert np.array_equal(mask_left_full, expected_mask_left)
        assert np.array_equal(mask_bottom_full, expected_mask_bottom)
        assert np.array_equal(mask_right_top_full, expected_mask_right_top)
        assert np.array_equal(mask_left_right_full, expected_mask_left_right)
        assert np.array_equal(mask_bottom_top_full, expected_mask_bottom_top)

    def test_repr_returns_correct_string(self):
        assert repr(self.basis) == "TensorProductBasis(dim=2)::BilinearBasis"

    def test_bilinear_plot_creates_correct_figure(self):
        plt.clf()
        fig = self.basis.plot((0, 0))
        
        assert fig is not None
        assert len(fig.axes) == 2
        ax = fig.axes[0]
        assert ax.get_title() == "2D Tensor Product Basis Function"
        assert ax.get_xlabel() == "x"
        assert ax.get_ylabel() == "y"
        assert len(ax.collections) > 0
        assert ax._colorbars is not None
        
        plt.close(fig)

    def test_plot_raises_error_for_invalid_dimension(self):
        self.basis._dimension = 1
        with pytest.raises(NotImplementedError):
            self.basis.plot((0,))
