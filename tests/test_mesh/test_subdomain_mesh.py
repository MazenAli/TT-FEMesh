import pytest
import numpy as np
import warnings
import matplotlib.pyplot as plt
from itertools import product
from unittest.mock import MagicMock, patch

from ttfemesh.mesh.subdomain_mesh import SubdomainMesh, SubdomainMesh2D, QuadMesh
from ttfemesh.domain import Subdomain, Subdomain2D, Quad
from ttfemesh.domain.subdomain_factory import QuadFactory
from ttfemesh.quadrature.quadrature import QuadratureRule, QuadratureRule2D, GaussLegendre2D
from ttfemesh.mesh import qindex2dtuple
from ttfemesh.tt_tools.tensor_cross import TTCrossConfig
from ttfemesh.types import TensorTrain


class TestSubdomainMesh:
    def test_abstract_base_class_cannot_be_instantiated(self):
        mock_subdomain = MagicMock(spec=Subdomain)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        
        with pytest.raises(TypeError):
            SubdomainMesh(
                subdomain=mock_subdomain,
                quadrature_rule=mock_quadrature_rule,
                mesh_size_exponent=2
            )

    def test_abstract_methods_must_be_implemented(self):
        class IncompleteSubdomainMesh(SubdomainMesh):
            pass
        
        mock_subdomain = MagicMock(spec=Subdomain)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        
        with pytest.raises(TypeError):
            IncompleteSubdomainMesh(
                subdomain=mock_subdomain,
                quadrature_rule=mock_quadrature_rule,
                mesh_size_exponent=2
            )

    def test_initialization_with_valid_parameters(self):
        class TestSubdomainMesh(SubdomainMesh):
            def ref2domain_map(self, _):
                return np.array([[0, 0]])
                
            def ref2element_map(self, index, xi):
                return np.array([[0, 0]])
                
            def ref2domain_jacobian(self, xi):
                return np.array([[[1, 0], [0, 1]]])
                
            def get_jacobian_tensor_trains(self):
                return np.array([])
                
            def plot(self):
                pass
                
            def _validate_idxs(self, *indices):
                pass
                
            def _validate_ref_coords(self, *coords, **kwargs):
                pass
                
            @property
            def dimension(self):
                return 2
        
        mock_subdomain = MagicMock(spec=Subdomain)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        
        subdomain_mesh = TestSubdomainMesh(
            subdomain=mock_subdomain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2
        )
        
        assert subdomain_mesh.subdomain == mock_subdomain
        assert subdomain_mesh.quadrature_rule == mock_quadrature_rule
        assert subdomain_mesh.mesh_size_exponent == 2
        assert subdomain_mesh._tt_cross_config is not None
        assert isinstance(subdomain_mesh._tt_cross_config, TTCrossConfig)

    def test_initialization_with_custom_tt_cross_config(self):
        class TestSubdomainMesh(SubdomainMesh):
            def ref2domain_map(self, xi):
                return np.array([[0, 0]])
                
            def ref2element_map(self, index, xi):
                return np.array([[0, 0]])
                
            def ref2domain_jacobian(self, xi):
                return np.array([[[1, 0], [0, 1]]])
                
            def get_jacobian_tensor_trains(self):
                return np.array([])
                
            def plot(self):
                pass
                
            def _validate_idxs(self, *indices):
                pass
                
            def _validate_ref_coords(self, *coords, **kwargs):
                pass
                
            @property
            def dimension(self):
                return 2
        
        mock_subdomain = MagicMock(spec=Subdomain)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        mock_tt_cross_config = MagicMock(spec=TTCrossConfig)
        
        subdomain_mesh = TestSubdomainMesh(
            subdomain=mock_subdomain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
            tt_cross_config=mock_tt_cross_config
        )
        
        assert subdomain_mesh._tt_cross_config == mock_tt_cross_config


@pytest.fixture
def subdomain_mesh():
    p1 = (0, 0)
    p2 = (2, 0)
    p3 = (3, 1)
    p4 = (-0.5, 3)

    quad = QuadFactory.create(p1, p2, p3, p4)
    qrule = GaussLegendre2D(2)
    return SubdomainMesh2D(
        subdomain=quad,
        quadrature_rule=qrule,
        mesh_size_exponent=3
    )

@pytest.fixture
def quad_mesh():
    p1 = (0, 0)
    p2 = (2, 0)
    p3 = (3, 1)
    p4 = (-0.5, 3)

    quad = QuadFactory.create(p1, p2, p3, p4)
    qrule = GaussLegendre2D(2)
    return QuadMesh(
        quad=quad,
        quadrature_rule=qrule,
        mesh_size_exponent=3
    )


def numerical_jacobian(func, xi_eta: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    num_samples = xi_eta.shape[0]
    numerical_jacobians = np.zeros((num_samples, 2, 2))

    for i, (xi, eta) in enumerate(xi_eta):
        xi_eta_0 = np.array([[xi, eta]])
        xi_eta_dxi = np.array([[xi + epsilon, eta]])
        xi_eta_deta = np.array([[xi, eta + epsilon]])

        xy_base = func(xi_eta_0).flatten()
        x_dxi = func(xi_eta_dxi).flatten()[0]
        y_dxi = func(xi_eta_dxi).flatten()[1]
        x_deta = func(xi_eta_deta).flatten()[0]
        y_deta = func(xi_eta_deta).flatten()[1]

        dx_dxi = (x_dxi - xy_base[0]) / epsilon
        dy_dxi = (y_dxi - xy_base[1]) / epsilon
        dx_deta = (x_deta - xy_base[0]) / epsilon
        dy_deta = (y_deta - xy_base[1]) / epsilon

        numerical_jacobians[i] = np.array([[dx_dxi, dx_deta], [dy_dxi, dy_deta]])

    return numerical_jacobians


class TestSubdomainMesh2D:
    def test_initialization(self):
        mock_subdomain = MagicMock(spec=Subdomain2D)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule2D)
        
        subdomain_mesh = SubdomainMesh2D(
            subdomain=mock_subdomain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2
        )
        
        assert subdomain_mesh.subdomain == mock_subdomain
        assert subdomain_mesh.quadrature_rule == mock_quadrature_rule
        assert subdomain_mesh.mesh_size_exponent == 2
        assert subdomain_mesh._tt_cross_config is not None
        assert subdomain_mesh.dimension == 2
        assert subdomain_mesh.num_points1d == 4 
        assert subdomain_mesh.num_points == 16 
        assert subdomain_mesh.num_elements1d == 3 
        assert subdomain_mesh.num_elements == 9 
        assert subdomain_mesh.index_map is not None
        assert subdomain_mesh._tca_strategy == subdomain_mesh._SubdomainMesh2D__tca_default

    def test_tt_cross_config_property(self):
        mock_subdomain = MagicMock(spec=Subdomain2D)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule2D)
        mock_tt_cross_config = MagicMock(spec=TTCrossConfig)
        
        subdomain_mesh = SubdomainMesh2D(
            subdomain=mock_subdomain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
            tt_cross_config=mock_tt_cross_config
        )
        
        assert subdomain_mesh.tt_cross_config == mock_tt_cross_config
        
        new_tt_cross_config = MagicMock(spec=TTCrossConfig)
        subdomain_mesh.tt_cross_config = new_tt_cross_config
        
        assert subdomain_mesh.tt_cross_config == new_tt_cross_config

    def test_ref2domain_map(self):
        mock_subdomain = MagicMock(spec=Subdomain2D)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule2D)
        
        mock_curve0 = MagicMock()
        mock_curve1 = MagicMock()
        mock_curve2 = MagicMock()
        mock_curve3 = MagicMock()
        
        mock_curve0.return_value = np.array([[1, 0]])
        mock_curve1.return_value = np.array([[0, 1]])
        mock_curve2.return_value = np.array([[-1, 0]])
        mock_curve3.return_value = np.array([[0, -1]])
        
        mock_curve0.get_start.return_value = np.array([0, 0])
        mock_curve1.get_start.return_value = np.array([0, 0])
        mock_curve2.get_start.return_value = np.array([0, 0])
        mock_curve3.get_start.return_value = np.array([0, 0])
        
        mock_subdomain.curves = [mock_curve0, mock_curve1, mock_curve2, mock_curve3]
        
        subdomain_mesh = SubdomainMesh2D(
            subdomain=mock_subdomain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2
        )
        
        def mock_ref2domain_map(xi_eta):
            return np.array([[0, 0]])
        
        subdomain_mesh.ref2domain_map = mock_ref2domain_map
        
        result = subdomain_mesh.ref2domain_map(np.array([[0, 0]]))
        
        assert np.array_equal(result, np.array([[0, 0]]))

    def test_ref2element_map(self):
        mock_subdomain = MagicMock(spec=Subdomain2D)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule2D)
        
        subdomain_mesh = SubdomainMesh2D(
            subdomain=mock_subdomain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2
        )
        
        def mock_ref2element_map(index, xi_eta):
            return np.array([[0, 0]])
        
        subdomain_mesh.ref2element_map = mock_ref2element_map
        
        result = subdomain_mesh.ref2element_map((0, 0), np.array([[0, 0]]))
        
        assert np.array_equal(result, np.array([[0, 0]]))

    def test_validate_idxs(self):
        mock_subdomain = MagicMock(spec=Subdomain2D)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule2D)
        
        subdomain_mesh = SubdomainMesh2D(
            subdomain=mock_subdomain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2
        )
        
        with warnings.catch_warnings(record=True) as w:
            subdomain_mesh._validate_idxs(0, 0)
            assert len(w) == 0
        
        with warnings.catch_warnings(record=True) as w:
            subdomain_mesh._validate_idxs(-1, 0)
            print(w)
            assert len(w) == 1
            assert "Index x=-1 is out of bounds" in str(w[0].message)
        
        with warnings.catch_warnings(record=True) as w:
            subdomain_mesh._validate_idxs(0, 3)
            assert len(w) == 1
            assert "Index y=3 is out of bounds" in str(w[0].message)

    def test_validate_ref_coords(self):
        mock_subdomain = MagicMock(spec=Subdomain2D)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule2D)
        
        subdomain_mesh = SubdomainMesh2D(
            subdomain=mock_subdomain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2
        )
        
        with warnings.catch_warnings(record=True) as w:
            subdomain_mesh._validate_ref_coords(np.array([[0, 0]]))
            assert len(w) == 0
        
        with pytest.raises(ValueError, match="Reference coordinates must have shape"):
            subdomain_mesh._validate_ref_coords(np.array([[0, 0, 0]]))
        
        with warnings.catch_warnings(record=True) as w:
            subdomain_mesh._validate_ref_coords(np.array([[2, 2]]))
            assert len(w) == 1
            assert "Reference coordinates are not in the range" in str(w[0].message)

    def test_repr(self):
        mock_subdomain = MagicMock(spec=Subdomain2D)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule2D)
        
        subdomain_mesh = SubdomainMesh2D(
            subdomain=mock_subdomain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2
        )
        
        repr_str = repr(subdomain_mesh)
        
        assert "SubdomainMesh2D(subdomain=" in repr_str
        assert "mesh_size_exponent=2" in repr_str
        assert "quadrature_rule=" in repr_str
        assert "num_points=16" in repr_str
        assert "num_elements=9" in repr_str

    def test_ref2domain_map_quad(self, subdomain_mesh):
        corner_points = np.array([
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1]
        ])
        expected_corners = np.array([
            [0, 0],
            [2, 0],
            [3, 1],
            [-0.5, 3]
        ])
        mapped_corners = subdomain_mesh.ref2domain_map(corner_points)
        np.testing.assert_allclose(mapped_corners, expected_corners, rtol=1e-10)

        midpoints = np.array([
            [0, -1],
            [1, 0],
            [0, 1],
            [-1, 0]
        ])
        expected_midpoints = np.array([
            [1, 0],
            [2.5, 0.5],
            [1.25, 2],
            [-0.25, 1.5]
        ])
        mapped_midpoints = subdomain_mesh.ref2domain_map(midpoints)
        np.testing.assert_allclose(mapped_midpoints, expected_midpoints, rtol=1e-10)

        center = np.array([[0, 0]])
        expected_center = np.array([[1.125, 1.]])
        mapped_center = subdomain_mesh.ref2domain_map(center)
        np.testing.assert_allclose(mapped_center, expected_center, rtol=1e-10)

    def test_ref2element_map_quad(self, subdomain_mesh):
        index = (0, 0)
        corner_points = np.array([
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1]
        ])
        expected_corners = np.array([
            [0, 0],
            [2./7., 0],
            [-0.5/7.+1./7.*(2.+1./7.+0.5/7.), 1./7.*(3.-2./7.)],
            [-0.5/7., 3./7.]
        ])

        mapped_corners = subdomain_mesh.ref2element_map(index, corner_points)
        np.testing.assert_allclose(mapped_corners, expected_corners, rtol=1e-10)

    def test_ref2domain_jacobian_quad(self, subdomain_mesh):
        center = np.array([[0, 0]])
        jacobian = subdomain_mesh.ref2domain_jacobian(center)
        approx_jacobian = numerical_jacobian(subdomain_mesh.ref2domain_map, center)
        np.testing.assert_allclose(jacobian, approx_jacobian, rtol=1e-8)

    def test_ref2element_jacobian_quad(self, subdomain_mesh):
        index = (0, 0)
        center = np.array([[0, 0]])
        jacobian = subdomain_mesh.ref2element_jacobian(index, center)
        func = lambda x: subdomain_mesh.ref2element_map(index, x)
        approx_jacobian = numerical_jacobian(func, center)
        np.testing.assert_allclose(jacobian, approx_jacobian, rtol=1e-8)

    @pytest.mark.parametrize("mesh_fixture", ["subdomain_mesh", "quad_mesh"])
    def test_get_jacobian_tensor_trains_quad(self, mesh_fixture, request):
        mesh = request.getfixturevalue(mesh_fixture)
        jacobian_tensor_trains = mesh.get_jacobian_tensor_trains()
        jacs = mesh.get_jacobian_tensors()
        num_quadrature_points = mesh.quadrature_rule.get_points_weights()[0].shape[0]
        assert jacobian_tensor_trains.shape == (num_quadrature_points, 2, 2)

        delta = 0.
        for qindex in product((0, 3), repeat=mesh.mesh_size_exponent):
            qindex_array = np.array(qindex)
            index = qindex2dtuple(qindex_array)

            for q in range(num_quadrature_points):
                for i in range(2):
                    for j in range(2):
                        jtt = jacobian_tensor_trains[q, i, j]
                        jtt = jtt.full()

                        exact = jacs[index][q, i, j]
                        approx = jtt[qindex].item()

                        delta += np.abs(exact - approx)       

        assert delta < 1e-8

    @pytest.mark.parametrize("mesh_fixture", ["subdomain_mesh", "quad_mesh"])
    def test_get_jacobian_det_tensor_trains_quad(self, mesh_fixture, request):
        mesh = request.getfixturevalue(mesh_fixture)
        jacobian_det_tensor_trains = mesh.get_jacobian_det_tensor_trains()
        jacobian_dets = mesh.get_jacobian_dets()
        num_quadrature_points = mesh.quadrature_rule.get_points_weights()[0].shape[0]
        assert jacobian_det_tensor_trains.shape == (num_quadrature_points,)

        delta = 0.
        for qindex in product((0, 3), repeat=mesh.mesh_size_exponent):
            qindex_array = np.array(qindex)
            index = qindex2dtuple(qindex_array)

            for q in range(num_quadrature_points):
                exact = jacobian_dets[index][q]
                approx = jacobian_det_tensor_trains[q]
                approx = approx.full()
                approx = approx[qindex]
                delta += np.abs(exact - approx)

        assert delta < 1e-8

    @pytest.mark.parametrize("mesh_fixture", ["subdomain_mesh", "quad_mesh"])
    def test_get_jacobian_invdet_tensor_trains_quad(self, mesh_fixture, request):
        mesh = request.getfixturevalue(mesh_fixture)
        jacobian_invdet_tensor_trains = mesh.get_jacobian_invdet_tensor_trains()
        jacobian_invdets = mesh.get_jacobian_invdets()
        num_quadrature_points = mesh.quadrature_rule.get_points_weights()[0].shape[0]
        assert jacobian_invdet_tensor_trains.shape == (num_quadrature_points,)

        delta = 0.
        for qindex in product((0, 3), repeat=mesh.mesh_size_exponent):
            qindex_array = np.array(qindex)
            index = qindex2dtuple(qindex_array)

            for q in range(num_quadrature_points):
                exact = jacobian_invdets[index][q]
                approx = jacobian_invdet_tensor_trains[q]
                approx = approx.full()
                approx = approx[qindex]
                delta += np.abs(exact - approx)

        assert delta < 1e-8

    @pytest.mark.parametrize("mesh_fixture", ["subdomain_mesh", "quad_mesh"])
    def test_get_jacobian_tensors_quad(self, mesh_fixture, request):
        mesh = request.getfixturevalue(mesh_fixture)
        jacobian_tensors = mesh.get_jacobian_tensors()
        num_quadrature_points = mesh.quadrature_rule.get_points_weights()[0].shape[0]
        size = 2**mesh.mesh_size_exponent
        assert jacobian_tensors.shape == (size, size, num_quadrature_points, 2, 2)

        delta = 0.
        for qindex in product((0, 3), repeat=mesh.mesh_size_exponent):
            qindex_array = np.array(qindex)
            index = qindex2dtuple(qindex_array)

            for q in range(num_quadrature_points):
                for i in range(2):
                    for j in range(2):
                        exact = jacobian_tensors[index][q, i, j]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", UserWarning)
                            approx = mesh.ref2element_jacobian(index,
                            mesh.quadrature_rule.get_points_weights()[0][q:q+1])[0, i, j]
                        delta += np.abs(exact - approx)

        assert delta < 1e-8

    @pytest.mark.parametrize("mesh_fixture", ["subdomain_mesh", "quad_mesh"])
    def test_get_jacobian_dets_quad(self, mesh_fixture, request):
        mesh = request.getfixturevalue(mesh_fixture)
        jacobian_tensors = mesh.get_jacobian_tensors()
        jacobian_dets = mesh.get_jacobian_dets()
        size = 2**mesh.mesh_size_exponent      
        num_quadrature_points = mesh.quadrature_rule.get_points_weights()[0].shape[0]
        assert jacobian_dets.shape == (size, size, num_quadrature_points)

        delta = 0.
        for qindex in product((0, 3), repeat=mesh.mesh_size_exponent):
            qindex_array = np.array(qindex)
            index = qindex2dtuple(qindex_array)

            for q in range(num_quadrature_points):
                exact = jacobian_dets[index][q]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    approx = np.linalg.det(jacobian_tensors[index][q])
                delta += np.abs(exact - approx)

        assert delta < 1e-8

    @pytest.mark.parametrize("mesh_fixture", ["subdomain_mesh", "quad_mesh"])
    def test_get_jacobian_invdets_quad(self, mesh_fixture, request):
        mesh = request.getfixturevalue(mesh_fixture)
        jacobian_tensors = mesh.get_jacobian_tensors()
        jacobian_invdets = mesh.get_jacobian_invdets()
        size = 2**mesh.mesh_size_exponent      
        num_quadrature_points = mesh.quadrature_rule.get_points_weights()[0].shape[0]
        assert jacobian_invdets.shape == (size, size, num_quadrature_points)

        delta = 0.
        for qindex in product((0, 3), repeat=mesh.mesh_size_exponent):
            qindex_array = np.array(qindex)
            index = qindex2dtuple(qindex_array)

            for q in range(num_quadrature_points):
                exact = jacobian_invdets[index][q]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    approx = 1./np.linalg.det(jacobian_tensors[index][q])
                delta += np.abs(exact - approx)

        assert delta < 1e-8

    def test_tca_default(self, subdomain_mesh):
        mesh_size_exponent = subdomain_mesh.mesh_size_exponent
        random_tensor = np.random.randn(*[4]*mesh_size_exponent)
        print("random_tensor.shape", random_tensor.shape)
        def oracle(idx):
            vals = []
            for i in range(idx.shape[0]):
                idx_ = idx[i, :]
                vals_ = random_tensor[tuple(idx_)]
                vals.append(vals_)

            return np.stack(vals)


        tca_default = subdomain_mesh._SubdomainMesh2D__tca_default(oracle)
        assert tca_default.shape == [4]*mesh_size_exponent

    def test_cross_func(self, subdomain_mesh):
        qindex = np.array([[0, 0, 0], [3, 0, 1], [0, 1, 0], [0, 1, 1]])
        xi_eta = np.array([[0.5, -0.5]])
        cross_func = subdomain_mesh._cross_func

        jacobians = cross_func(qindex, xi_eta)
        num_indices = qindex.shape[0]
        assert jacobians.shape == (num_indices, 2, 2)

        for idx in range(qindex.shape[0]):
            single_bindex = np.array(qindex[idx, :])
            index = subdomain_mesh.index_map(single_bindex)
            jacobians_ist = jacobians[idx]
            jacobians_soll = subdomain_mesh.ref2element_jacobian(index, xi_eta)
            assert np.allclose(jacobians_ist, jacobians_soll)

    def test_cross_func_no_index_map(self, subdomain_mesh):
        subdomain_mesh._index_map = None
        with pytest.raises(ValueError, match="Index map is not defined"):
            subdomain_mesh._cross_func(np.array([[0, 0, 0]]), np.array([[0.5, -0.5]]))

    def test_cross_func_multiple_points(self, subdomain_mesh):
        qindex = np.array([[0, 0, 0], [3, 0, 1], [0, 1, 0], [0, 1, 1]])
        xi_eta = np.array([[0.5, -0.5], [0.5, 0.5]])
        with pytest.raises(ValueError,
                           match="Only one evaluation point is supported for TCA"):
            subdomain_mesh._cross_func(qindex, xi_eta)

    def test_plot_element(self, subdomain_mesh):
        try:
            subdomain_mesh.plot_element((0, 0), num_points=100)
            plt.close()
        except Exception as e:
            pytest.fail(f"Plotting failed with error: {e}")

    def test_plot(self, subdomain_mesh):
        try:
            subdomain_mesh.plot(num_points=100)
            plt.close()
        except Exception as e:
            pytest.fail(f"Plotting failed with error: {e}")

    def test_repr(self, subdomain_mesh):
        repr_str = repr(subdomain_mesh)
        assert "SubdomainMesh2D" in repr_str
        assert "subdomain=" in repr_str
        assert "quadrature_rule=" in repr_str
        assert "mesh_size_exponent=" in repr_str
        assert "num_points=" in repr_str


class TestQuadMesh:
    def test_initialization(self):
        mock_quad = MagicMock(spec=Quad)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule2D)
        
        quad_mesh = QuadMesh(
            quad=mock_quad,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2
        )
        
        assert quad_mesh.subdomain == mock_quad
        assert quad_mesh.quadrature_rule == mock_quadrature_rule
        assert quad_mesh.mesh_size_exponent == 2
        assert quad_mesh._tt_cross_config is not None
        assert quad_mesh.dimension == 2
        assert quad_mesh.num_points1d == 4 
        assert quad_mesh.num_points == 16 
        assert quad_mesh.num_elements1d == 3 
        assert quad_mesh.num_elements == 9 
        assert quad_mesh.index_map is not None
        assert quad_mesh._tca_strategy == quad_mesh._QuadMesh__linear_interpolation

    def test_linear_interpolation(self):
        mock_quad = MagicMock(spec=Quad)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule2D)
        
        quad_mesh = QuadMesh(
            quad=mock_quad,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2
        )
        
        mock_oracle = MagicMock()
        mock_oracle.return_value = np.array([1.0])
        
        mock_tt_interpolant = MagicMock(spec=TensorTrain)
        
        with patch('ttfemesh.mesh.subdomain_mesh.interpolate_linear2d',
                    return_value=mock_tt_interpolant):
            result = quad_mesh._QuadMesh__linear_interpolation(mock_oracle)
        
        assert result == mock_tt_interpolant

    def test_repr(self):
        mock_quad = MagicMock(spec=Quad)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule2D)
        
        quad_mesh = QuadMesh(
            quad=mock_quad,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2
        )
        
        repr_str = repr(quad_mesh)
        
        assert "QuadMesh(subdomain=" in repr_str
        assert "mesh_size_exponent=2" in repr_str
        assert "quadrature_rule=" in repr_str
        assert "num_points=16" in repr_str
        assert "num_elements=9" in repr_str 
