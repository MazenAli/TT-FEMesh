import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from ttfemesh.mesh.domain_mesh import DomainMesh, DomainMesh2D, DomainBilinearMesh2D
from ttfemesh.domain.domain import Domain
from ttfemesh.domain.subdomain import Subdomain2D
from ttfemesh.domain.subdomain_connection import CurveConnection2D, VertexConnection2D
from ttfemesh.basis.basis import TensorProductBasis, BilinearBasis
from ttfemesh.quadrature.quadrature import QuadratureRule
from ttfemesh.tn_tools.tensor_cross import TTCrossConfig
from ttfemesh.types import BoundarySide2D, BoundaryVertex2D, TensorTrain


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
                basis=mock_basis
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
                basis=mock_basis
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
            basis=mock_basis
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
            basis=mock_basis
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
            basis=mock_basis
        )
        
        with pytest.raises(ValueError,
                    match="Invalid subdomain index: 2. Valid indices are in the range \\[0, 2\\)"):
            domain_mesh.get_subdomain_mesh(2)
        
        with pytest.raises(ValueError,
                    match="Invalid subdomain index: -1. Valid indices are in the range \\[0, 2\\)"):
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
            basis=mock_basis
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
            basis=mock_basis
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
            basis=mock_basis
        )
        
        with patch('builtins.print') as mock_print:
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
            basis=mock_basis
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
            basis=mock_basis
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
            basis=mock_basis
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
            DomainMesh2D(domain=mock_domain, quadrature_rule=mock_quadrature_rule,
                         mesh_size_exponent=2, basis=mock_basis)

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
            basis=mock_basis
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
            basis=mock_basis
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
        mock_basis = MagicMock(spec=TensorProductBasis)
        
        domain_mesh = DomainBilinearMesh2D(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
            basis=mock_basis
        )
        
        assert domain_mesh.domain == mock_domain
        assert domain_mesh.quadrature_rule == mock_quadrature_rule
        assert domain_mesh.mesh_size_exponent == 2
        assert domain_mesh.basis == mock_basis
        assert domain_mesh._tt_cross_config is None
        assert len(domain_mesh.subdomain_meshes) == 2
        assert mock_domain.get_subdomain.call_count == 2

    def test_get_concatenation_maps_with_vertex_connection(self):
        mock_domain = MagicMock(spec=Domain)
        mock_connection = MagicMock(spec=VertexConnection2D)
        mock_domain.get_connections.return_value = [mock_connection]
        
        mock_connection.get_connection_pairs.return_value = [
            ((0, 1), (0, 1), ("start", "end"))
        ]
        
        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        mock_basis = MagicMock(spec=TensorProductBasis)
        
        mock_tt_connectivity = MagicMock(spec=TensorTrain)
        
        domain_mesh = DomainBilinearMesh2D(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
            basis=mock_basis
        )
        
        with patch('ttfemesh.mesh.domain_mesh.vertex_concatenation_tt',
                   return_value=mock_tt_connectivity):
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
        mock_basis = MagicMock(spec=TensorProductBasis)
        
        mock_tt_connectivity = MagicMock(spec=TensorTrain)
        
        domain_mesh = DomainBilinearMesh2D(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
            basis=mock_basis
        )
        
        with patch('ttfemesh.mesh.domain_mesh.side_concatenation_tt',
                   return_value=mock_tt_connectivity):
            concatenation_maps = domain_mesh.get_concatenation_maps()
        
        mock_domain.get_connections.assert_called_once()
        assert concatenation_maps == {(0, 1): mock_tt_connectivity}

    def test_get_concatenation_maps_with_unsupported_connection(self):
        mock_domain = MagicMock(spec=Domain)
        mock_connection = MagicMock()
        mock_domain.get_connections.return_value = [mock_connection]
        
        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        mock_basis = MagicMock(spec=TensorProductBasis)
        
        domain_mesh = DomainBilinearMesh2D(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
            basis=mock_basis
        )
        
        with pytest.raises(ValueError,
                           match="Unsupported connection type: <class 'unittest.mock.MagicMock'>"):
            domain_mesh.get_concatenation_maps()

    def test_get_concatenation_maps_with_multiple_connections(self):
        mock_domain = MagicMock(spec=Domain)
        
        side_connection = MagicMock(spec=CurveConnection2D)
        side_connection.subdomains_indices = (0, 1)
        side_connection.curve_indices = (2, 0)
        
        vertex_connection = MagicMock(spec=VertexConnection2D)
        vertex_connection.get_connection_pairs.return_value = [
            ((1, 2), (1, 0), ("end", "start"))
        ]
        
        mock_domain.get_connections.return_value = [side_connection, vertex_connection]
        
        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        mock_basis = MagicMock(spec=TensorProductBasis)
        
        mock_side_tt = MagicMock(spec=TensorTrain)
        mock_vertex_tt = MagicMock(spec=TensorTrain)
        
        mock_side_tt.shape = [(4, 4), (4, 4)]
        mock_side_tt.full.return_value = np.zeros((16, 16))
        side_array = np.zeros((16, 16))
        for i in range(8):
            side_array[i*8 + 7, i*8] = 1.0  # (i, 7) ~ (i, 0)
        mock_side_tt.full.return_value = side_array
        
        mock_vertex_tt.shape = [(4, 4), (4, 4)]
        mock_vertex_tt.full.return_value = np.zeros((16, 16))
        vertex_array = np.zeros((16, 16))
        vertex_array[7*8 + 7, 7*8 + 7] = -1.0
        mock_vertex_tt.full.return_value = vertex_array
        
        domain_mesh = DomainBilinearMesh2D(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=3,
            basis=mock_basis
        )
        
        with patch('ttfemesh.mesh.domain_mesh.side_concatenation_tt', 
                  return_value=mock_side_tt), \
             patch('ttfemesh.mesh.domain_mesh.vertex_concatenation_tt', 
                  return_value=mock_vertex_tt):
            
            concatenation_maps = domain_mesh.get_concatenation_maps()
        
        mock_domain.get_connections.assert_called_once()
        vertex_connection.get_connection_pairs.assert_called_once()
        
        assert (0, 1) in concatenation_maps
        assert (1, 2) in concatenation_maps
        assert concatenation_maps[(0, 1)] == mock_side_tt
        assert concatenation_maps[(1, 2)] == mock_vertex_tt
        
        side_tt = concatenation_maps[(0, 1)]
        side_array = np.array(side_tt.full()).reshape((16, 16), order="F")
        rows, cols = np.where(side_array == 1.0)
        for r, c in zip(rows, cols):
            r1, r2 = r // 8, r % 8
            c1, c2 = c // 8, c % 8
            assert r1 == c1 
            assert r2 == 7 and c2 == 0 
        
        vertex_tt = concatenation_maps[(1, 2)]
        vertex_array = np.array(vertex_tt.full()).reshape((16, 16), order="F")
        rows, cols = np.where(vertex_array == -1.0)
        for r, c in zip(rows, cols):
            r1, r2 = r // 8, r % 8
            c1, c2 = c // 8, c % 8
            assert r1 == 7 and r2 == 7 
            assert c1 == 7 and c2 == 7 

    def test_repr(self):
        mock_domain = MagicMock(spec=Domain)
        mock_quadrature_rule = MagicMock(spec=QuadratureRule)
        mock_basis = MagicMock(spec=TensorProductBasis)
        
        domain_mesh = DomainBilinearMesh2D(
            domain=mock_domain,
            quadrature_rule=mock_quadrature_rule,
            mesh_size_exponent=2,
            basis=mock_basis
        )
        
        repr_str = repr(domain_mesh)
        
        assert "DomainBilinearMesh2D(domain=" in repr_str
        assert "mesh_size_exponent=2" in repr_str
        assert "quadrature_rule=" in repr_str
        assert "basis=" in repr_str 
