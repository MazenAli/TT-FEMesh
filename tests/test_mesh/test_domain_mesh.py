from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ttfemesh.basis.basis import BilinearBasis, TensorProductBasis
from ttfemesh.domain import Domain, Domain2D, Quad
from ttfemesh.domain.subdomain import Subdomain2D
from ttfemesh.domain.subdomain_connection import CurveConnection2D, VertexConnection2D
from ttfemesh.domain.subdomain_factory import QuadFactory
from ttfemesh.mesh import DomainBilinearMesh2D, DomainMesh, DomainMesh2D, QuadMesh, SubdomainMesh2D
from ttfemesh.quadrature.quadrature import GaussLegendre2D, QuadratureRule
from ttfemesh.tt_tools.meshgrid import map2canonical2d
from ttfemesh.types import TensorTrain


class TestDomainMesh:
    def test_abstract_base_class_cannot_be_instantiated(self):
        mock_domain = MagicMock(spec=Domain)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        mock_basis = MagicMock(spec=TensorProductBasis)

        with pytest.raises(TypeError):
            DomainMesh(
                domain=mock_domain,
                quadrature_rule=mock_quadrature_rule,
                mesh_size_exponent=2,
                basis=mock_basis,
            )

    def test_abstract_methods_must_be_implemented(self):
        class IncompleteDomainMesh(DomainMesh):
            pass

        mock_domain = MagicMock(spec=Domain)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        mock_basis = MagicMock(spec=TensorProductBasis)

        with pytest.raises(TypeError):
            IncompleteDomainMesh(
                domain=mock_domain,
                quadrature_rule=mock_quadrature_rule,
                mesh_size_exponent=2,
                basis=mock_basis,
            )

    def test_initialization_with_valid_parameters(self):
        class TestDomainMesh(DomainMesh):
            def _create_subdomain_meshes(self):
                return []

            def get_concatenation_maps(self):
                return {}

        mock_domain = MagicMock(spec=Domain)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        mock_basis = MagicMock(spec=TensorProductBasis)

        domain_mesh = TestDomainMesh(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
            basis=mock_basis,
        )

        assert domain_mesh.domain == mock_domain
        assert domain_mesh.quadrature_rule == mock_quadrature_rule
        assert domain_mesh.mesh_size_exponent == 2
        assert domain_mesh.basis == mock_basis
        assert domain_mesh._tt_cross_config is None
        assert domain_mesh.subdomain_meshes == []

    def test_get_subdomain_mesh_with_valid_index(self):
        class TestDomainMesh(DomainMesh):
            def _create_subdomain_meshes(self):
                return [MagicMock(), MagicMock()]

            def get_concatenation_maps(self):
                return {}

        mock_domain = MagicMock(spec=Domain)
        mock_domain.num_subdomains = 2
        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        mock_basis = MagicMock(spec=TensorProductBasis)

        domain_mesh = TestDomainMesh(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
            basis=mock_basis,
        )

        subdomain_mesh = domain_mesh.get_subdomain_mesh(0)

        assert subdomain_mesh == domain_mesh.subdomain_meshes[0]

    def test_get_subdomain_mesh_with_invalid_index(self):
        class TestDomainMesh(DomainMesh):
            def _create_subdomain_meshes(self):
                return [MagicMock(), MagicMock()]

            def get_concatenation_maps(self):
                return {}

        mock_domain = MagicMock(spec=Domain)
        mock_domain.num_subdomains = 2
        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        mock_basis = MagicMock(spec=TensorProductBasis)

        domain_mesh = TestDomainMesh(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
            basis=mock_basis,
        )

        with pytest.raises(
            ValueError,
            match="Invalid subdomain index: 2. Valid indices are in the range \\[0, 2\\)",
        ):
            domain_mesh.get_subdomain_mesh(2)

        with pytest.raises(
            ValueError,
            match="Invalid subdomain index: -1. Valid indices are in the range \\[0, 2\\)",
        ):
            domain_mesh.get_subdomain_mesh(-1)

    def test_get_element2global_index_map(self):
        class TestDomainMesh(DomainMesh):
            def _create_subdomain_meshes(self):
                return []

            def get_concatenation_maps(self):
                return {}

        mock_domain = MagicMock(spec=Domain)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        mock_basis = MagicMock(spec=TensorProductBasis)
        mock_ttmaps = np.array([[MagicMock(), MagicMock()], [MagicMock(), MagicMock()]])
        mock_basis.get_all_element2global_ttmaps.return_value = mock_ttmaps

        domain_mesh = TestDomainMesh(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
            basis=mock_basis,
        )

        ttmaps = domain_mesh.get_element2global_index_map()

        mock_basis.get_all_element2global_ttmaps.assert_called_once_with(2)
        assert ttmaps is mock_ttmaps

    def test_get_element2global_index_map_returns_correct_shape(self):
        class TestDomainMesh(DomainMesh):
            def _create_subdomain_meshes(self):
                return []

            def get_concatenation_maps(self):
                return {}

        mock_domain = MagicMock(spec=Domain)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        mock_basis = MagicMock(spec=TensorProductBasis)

        mock_tt = MagicMock(spec=TensorTrain)
        mock_tt.shape = [(2, 2), (2, 2)]

        mock_ttmaps = np.array([[mock_tt, mock_tt], [mock_tt, mock_tt]])
        mock_basis.get_all_element2global_ttmaps.return_value = mock_ttmaps

        domain_mesh = TestDomainMesh(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
            basis=mock_basis,
        )

        ttmaps = domain_mesh.get_element2global_index_map()

        assert ttmaps.shape == (2, 2)

        for i in range(2):
            for j in range(2):
                assert ttmaps[i, j].shape == [(2, 2), (2, 2)]

    def test_get_dirichlet_masks_with_no_boundary_condition(self):
        class TestDomainMesh(DomainMesh):
            def _create_subdomain_meshes(self):
                return []

            def get_concatenation_maps(self):
                return {}

        mock_domain = MagicMock(spec=Domain)
        mock_domain.boundary_condition = None
        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        mock_basis = MagicMock(spec=TensorProductBasis)

        domain_mesh = TestDomainMesh(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
            basis=mock_basis,
        )

        with patch("builtins.print") as mock_print:
            masks = domain_mesh.get_dirichlet_masks()

        mock_print.assert_called_once_with("No boundary condition specified.")
        assert masks is None

    def test_get_dirichlet_masks_with_boundary_condition(self):
        class TestDomainMesh(DomainMesh):
            def _create_subdomain_meshes(self):
                return []

            def get_concatenation_maps(self):
                return {}

        mock_domain = MagicMock(spec=Domain)
        mock_boundary_condition = MagicMock()
        mock_domain.boundary_condition = mock_boundary_condition
        mock_grouped = {0: [0, 1], 1: [2, 3]}
        mock_boundary_condition.group_by_subdomain.return_value = mock_grouped

        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        mock_basis = MagicMock(spec=TensorProductBasis)
        mock_boundary_mask = MagicMock(spec=TensorTrain)
        mock_basis.get_dirichlet_mask.return_value = mock_boundary_mask

        domain_mesh = TestDomainMesh(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
            basis=mock_basis,
        )

        masks = domain_mesh.get_dirichlet_masks()

        mock_boundary_condition.group_by_subdomain.assert_called_once()
        assert mock_basis.get_dirichlet_mask.call_count == 2
        assert masks == {0: mock_boundary_mask, 1: mock_boundary_mask}

    def test_get_dirichlet_masks_returns_correct_shape(self):
        class TestDomainMesh(DomainMesh):
            def _create_subdomain_meshes(self):
                return []

            def get_concatenation_maps(self):
                return {}

        mock_domain = MagicMock(spec=Domain)
        mock_boundary_condition = MagicMock()
        mock_domain.boundary_condition = mock_boundary_condition
        mock_grouped = {0: [0, 1], 1: [2, 3]}
        mock_boundary_condition.group_by_subdomain.return_value = mock_grouped

        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        mock_basis = MagicMock(spec=TensorProductBasis)

        mock_boundary_mask = MagicMock(spec=TensorTrain)
        mock_boundary_mask.shape = [2, 2, 2]
        mock_basis.get_dirichlet_mask.return_value = mock_boundary_mask

        domain_mesh = TestDomainMesh(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
            basis=mock_basis,
        )

        masks = domain_mesh.get_dirichlet_masks()

        assert set(masks.keys()) == {0, 1}

        for subdomain_index in masks:
            assert masks[subdomain_index].shape == [2, 2, 2]

    def test_repr(self):
        class TestDomainMesh(DomainMesh):
            def _create_subdomain_meshes(self):
                return []

            def get_concatenation_maps(self):
                return {}

        mock_domain = MagicMock(spec=Domain)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        mock_basis = MagicMock(spec=TensorProductBasis)

        domain_mesh = TestDomainMesh(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
            basis=mock_basis,
        )

        repr_str = repr(domain_mesh)

        assert "DomainMesh(domain=" in repr_str
        assert "mesh_size_exponent=2" in repr_str
        assert "quadrature_rule=" in repr_str
        assert "basis=" in repr_str


class DomainMesh2DTest(DomainMesh2D):
    def _create_subdomain_meshes(self):
        return [MagicMock(), MagicMock()]

    def get_concatenation_maps(self):
        return {}


class TestDomainMesh2D:
    def test_abstract_base_class_cannot_be_instantiated(self):
        mock_domain = MagicMock(spec=Domain)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        mock_basis = MagicMock(spec=TensorProductBasis)

        with pytest.raises(TypeError):
            DomainMesh2D(
                domain=mock_domain,
                quadrature_rule=mock_quadrature_rule,
                mesh_size_exponent=2,
                basis=mock_basis,
            )

    def test_initialization(self):
        mock_domain = MagicMock(spec=Domain)
        mock_domain.num_subdomains = 2
        mock_domain.get_subdomain.return_value = MagicMock(spec=Subdomain2D)

        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        mock_basis = MagicMock(spec=TensorProductBasis)

        domain_mesh = DomainMesh2DTest(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
            basis=mock_basis,
        )

        assert domain_mesh.domain == mock_domain
        assert domain_mesh.quadrature_rule == mock_quadrature_rule
        assert domain_mesh.mesh_size_exponent == 2
        assert domain_mesh.basis == mock_basis
        assert domain_mesh._tt_cross_config is None
        assert len(domain_mesh.subdomain_meshes) == 2

    def test_repr(self):
        mock_domain = MagicMock(spec=Domain)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        mock_basis = MagicMock(spec=TensorProductBasis)

        domain_mesh = DomainMesh2DTest(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
            basis=mock_basis,
        )

        repr_str = repr(domain_mesh)

        assert "DomainMesh2D(domain=" in repr_str
        assert "mesh_size_exponent=2" in repr_str
        assert "quadrature_rule=" in repr_str
        assert "basis=" in repr_str


class TestDomainBilinearMesh2D:
    def test_initialization(self):
        mock_domain = MagicMock(spec=Domain)
        mock_domain.num_subdomains = 2
        mock_domain.get_subdomain.return_value = MagicMock(spec=Subdomain2D)

        mock_quadrature_rule = MagicMock(spec=QuadratureRule)

        domain_mesh = DomainBilinearMesh2D(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
        )

        assert domain_mesh.domain == mock_domain
        assert domain_mesh.quadrature_rule == mock_quadrature_rule
        assert domain_mesh.mesh_size_exponent == 2
        assert isinstance(domain_mesh.basis, BilinearBasis)
        assert domain_mesh._tt_cross_config is None
        assert len(domain_mesh.subdomain_meshes) == 2
        assert mock_domain.get_subdomain.call_count == 2

    def test_get_concatenation_maps_with_vertex_connection(self):
        mock_domain = MagicMock(spec=Domain)
        mock_connection = MagicMock(spec=VertexConnection2D)
        mock_domain.get_connections.return_value = [mock_connection]

        mock_connection.get_connection_pairs.return_value = [((0, 1), (0, 1), ("start", "end"))]

        mock_quadrature_rule = MagicMock(spec=QuadratureRule)

        mock_tt_connectivity = MagicMock(spec=TensorTrain)

        domain_mesh = DomainBilinearMesh2D(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
        )

        with patch(
            "ttfemesh.mesh.domain_mesh.vertex_concatenation_tt", return_value=mock_tt_connectivity
        ):
            concatenation_maps = domain_mesh.get_concatenation_maps()

        mock_domain.get_connections.assert_called_once()
        mock_connection.get_connection_pairs.assert_called_once()
        assert concatenation_maps == {(0, 1): mock_tt_connectivity}

    def test_get_concatenation_maps_with_curve_connection(self):
        mock_domain = MagicMock(spec=Domain)
        mock_connection = MagicMock(spec=CurveConnection2D)
        mock_connection.subdomains_indices = (0, 1)
        mock_connection.curve_indices = (0, 1)
        mock_domain.get_connections.return_value = [mock_connection]

        mock_quadrature_rule = MagicMock(spec=QuadratureRule)

        mock_tt_connectivity = MagicMock(spec=TensorTrain)

        domain_mesh = DomainBilinearMesh2D(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
        )

        with patch(
            "ttfemesh.mesh.domain_mesh.side_concatenation_tt", return_value=mock_tt_connectivity
        ):
            concatenation_maps = domain_mesh.get_concatenation_maps()

        mock_domain.get_connections.assert_called_once()
        assert concatenation_maps == {(0, 1): mock_tt_connectivity}

    def test_get_concatenation_maps_with_unsupported_connection(self):
        mock_domain = MagicMock(spec=Domain)
        mock_connection = MagicMock()
        mock_domain.get_connections.return_value = [mock_connection]

        mock_quadrature_rule = MagicMock(spec=QuadratureRule)

        domain_mesh = DomainBilinearMesh2D(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
        )

        with pytest.raises(
            ValueError, match="Unsupported connection type: <class 'unittest.mock.MagicMock'>"
        ):
            domain_mesh.get_concatenation_maps()

    def test_get_concatenation_maps_with_multiple_connections(self):
        mesh_size_exponent = 3
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

        p1 = (0, 0)
        p2 = (3, 0)
        p3 = (4, 1)
        p4 = (0.8, 1.5)
        quad1 = QuadFactory.create(p1, p2, p3, p4)

        p1 = (3, 0)
        p2 = (5, 0)
        p3 = (6, 1)
        p4 = (4, 1)
        quad2 = QuadFactory.create(p1, p2, p3, p4)

        p1 = (3, 0)
        p2 = (1, -3)
        p3 = (7, -4)
        p4 = (5, -1)
        quad3 = QuadFactory.create(p1, p2, p3, p4)

        domain_idxs = [0, 1]
        curve_idxs = [1, 3]
        edge = CurveConnection2D(domain_idxs, curve_idxs)

        vertex_idxs = [(0, 0, "end"), (2, 3, "end")]
        vertex = VertexConnection2D(vertex_idxs)

        domain = Domain2D([quad1, quad2, quad3], [edge, vertex])

        order = 2
        qrule = GaussLegendre2D(order)
        basis2d = BilinearBasis()

        domain_mesh = DomainBilinearMesh2D(domain, qrule, mesh_size_exponent, basis2d)
        concatenation_maps = domain_mesh.get_concatenation_maps()

        assert (0, 1) in concatenation_maps
        assert (0, 2) in concatenation_maps

        side_exact = np.array(
            [
                [7, 0, 0, 0],
                [7, 1, 0, 1],
                [7, 2, 0, 2],
                [7, 3, 0, 3],
                [7, 4, 0, 4],
                [7, 5, 0, 5],
                [7, 6, 0, 6],
                [7, 7, 0, 7],
            ]
        )

        vertex_exact = np.array([[7, 0, 0, 0]])

        side_tt = concatenation_maps[(0, 1)][0]
        side_array = reshape_ttmap(side_tt)
        rows1, rows2, cols1, cols2 = np.where(side_array == 1.0)

        assert np.all(rows1 == side_exact[:, 0])
        assert np.all(rows2 == side_exact[:, 1])
        assert np.all(cols1 == side_exact[:, 2])
        assert np.all(cols2 == side_exact[:, 3])

        side_tt = concatenation_maps[(0, 1)][1]
        side_array = reshape_ttmap(side_tt)
        rows1, rows2, cols1, cols2 = np.where(side_array == -1.0)

        assert np.all(rows1 == side_exact[:, 0])
        assert np.all(rows2 == side_exact[:, 1])
        assert np.all(cols1 == side_exact[:, 0])
        assert np.all(cols2 == side_exact[:, 1])

        side_tt = concatenation_maps[(0, 1)][2]
        side_array = reshape_ttmap(side_tt)
        rows1, rows2, cols1, cols2 = np.where(side_array == -1.0)

        assert np.all(rows1 == side_exact[:, 2])
        assert np.all(rows2 == side_exact[:, 3])
        assert np.all(cols1 == side_exact[:, 2])
        assert np.all(cols2 == side_exact[:, 3])

        vertex_tt = concatenation_maps[(0, 2)][0]
        vertex_array = reshape_ttmap(vertex_tt)
        rows1, rows2, cols1, cols2 = np.where(vertex_array == 1.0)
        print(rows1, rows2, cols1, cols2)

        assert np.all(rows1 == vertex_exact[:, 0])
        assert np.all(rows2 == vertex_exact[:, 1])
        assert np.all(cols1 == vertex_exact[:, 2])
        assert np.all(cols2 == vertex_exact[:, 3])

        vertex_tt = concatenation_maps[(0, 2)][1]
        vertex_array = reshape_ttmap(vertex_tt)
        rows1, rows2, cols1, cols2 = np.where(vertex_array == -1.0)

        assert np.all(rows1 == vertex_exact[:, 0])
        assert np.all(rows2 == vertex_exact[:, 1])
        assert np.all(cols1 == vertex_exact[:, 0])
        assert np.all(cols2 == vertex_exact[:, 1])

        vertex_tt = concatenation_maps[(0, 2)][2]
        vertex_array = reshape_ttmap(vertex_tt)
        rows1, rows2, cols1, cols2 = np.where(vertex_array == -1.0)

        assert np.all(rows1 == vertex_exact[:, 2])
        assert np.all(rows2 == vertex_exact[:, 3])
        assert np.all(cols1 == vertex_exact[:, 2])
        assert np.all(cols2 == vertex_exact[:, 3])

    def test_create_subdomain_meshes_with_quad(self):
        mock_quad = MagicMock(spec=Quad)
        mock_domain = MagicMock(spec=Domain)
        mock_domain.num_subdomains = 1
        mock_domain.get_subdomain.return_value = mock_quad

        mock_quadrature_rule = MagicMock(spec=QuadratureRule)

        domain_mesh = DomainBilinearMesh2D(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
        )

        assert len(domain_mesh.subdomain_meshes) == 1
        assert isinstance(domain_mesh.subdomain_meshes[0], QuadMesh)
        assert domain_mesh.subdomain_meshes[0].subdomain == mock_quad
        assert domain_mesh.subdomain_meshes[0].quadrature_rule == mock_quadrature_rule
        assert domain_mesh.subdomain_meshes[0].mesh_size_exponent == 2

    def test_create_subdomain_meshes_with_subdomain2d(self):
        mock_subdomain = MagicMock(spec=Subdomain2D)
        mock_domain = MagicMock(spec=Domain)
        mock_domain.num_subdomains = 1
        mock_domain.get_subdomain.return_value = mock_subdomain

        mock_quadrature_rule = MagicMock(spec=QuadratureRule)

        domain_mesh = DomainBilinearMesh2D(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
        )

        assert len(domain_mesh.subdomain_meshes) == 1
        assert isinstance(domain_mesh.subdomain_meshes[0], SubdomainMesh2D)
        assert not isinstance(domain_mesh.subdomain_meshes[0], QuadMesh)
        assert domain_mesh.subdomain_meshes[0].subdomain == mock_subdomain
        assert domain_mesh.subdomain_meshes[0].quadrature_rule == mock_quadrature_rule
        assert domain_mesh.subdomain_meshes[0].mesh_size_exponent == 2

    def test_create_subdomain_meshes_with_mixed_subdomains(self):
        mock_quad = MagicMock(spec=Quad)
        mock_subdomain = MagicMock(spec=Subdomain2D)
        mock_domain = MagicMock(spec=Domain)
        mock_domain.num_subdomains = 2
        mock_domain.get_subdomain.side_effect = [mock_quad, mock_subdomain]

        mock_quadrature_rule = MagicMock(spec=QuadratureRule)

        domain_mesh = DomainBilinearMesh2D(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
        )

        assert len(domain_mesh.subdomain_meshes) == 2
        assert isinstance(domain_mesh.subdomain_meshes[0], QuadMesh)
        assert isinstance(domain_mesh.subdomain_meshes[1], SubdomainMesh2D)
        assert not isinstance(domain_mesh.subdomain_meshes[1], QuadMesh)

        assert domain_mesh.subdomain_meshes[0].subdomain == mock_quad
        assert domain_mesh.subdomain_meshes[0].quadrature_rule == mock_quadrature_rule
        assert domain_mesh.subdomain_meshes[0].mesh_size_exponent == 2

        assert domain_mesh.subdomain_meshes[1].subdomain == mock_subdomain
        assert domain_mesh.subdomain_meshes[1].quadrature_rule == mock_quadrature_rule
        assert domain_mesh.subdomain_meshes[1].mesh_size_exponent == 2

    def test_repr(self):
        mock_domain = MagicMock(spec=Domain)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule)

        domain_mesh = DomainBilinearMesh2D(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
        )

        repr_str = repr(domain_mesh)

        assert "DomainBilinearMesh2D(domain=" in repr_str
        assert "mesh_size_exponent=2" in repr_str
        assert "quadrature_rule=" in repr_str
        assert "basis=" in repr_str
