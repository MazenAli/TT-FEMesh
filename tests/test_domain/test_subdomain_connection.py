import pytest
import numpy as np
from ttfemesh.domain.subdomain_connection import (
    SubdomainConnection,
    SubdomainConnection2D,
    VertexConnection2D,
    CurveConnection2D,
)
from ttfemesh.domain.subdomain import Subdomain2D
from ttfemesh.domain.curve import Line2D
from ttfemesh.domain.subdomain_factory import RectangleFactory


class TestSubdomainConnection:
    def test_abstract_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            SubdomainConnection()

    def test_abstract_methods_must_be_implemented(self):
        class IncompleteConnection(SubdomainConnection):
            def validate(self):
                pass

        with pytest.raises(TypeError):
            IncompleteConnection()


class TestSubdomainConnection2D:
    def test_abstract_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            SubdomainConnection2D()

    def test_dimension(self):
        class TestConnection2D(SubdomainConnection2D):
            def validate(self):
                pass

            @property
            def num_connected_subdomains(self) -> int:
                return 2

        connection = TestConnection2D()
        assert connection.dimension == 2

@pytest.fixture
def sample_subdomains():
    subdomain1 = RectangleFactory.create((0., 0.), (3., 1.))
    subdomain2 = RectangleFactory.create((3., 0.), (4., 1.))
    subdomain3 = RectangleFactory.create((-1., -1.), (0., 0.))
    return [subdomain1, subdomain2, subdomain3]

class TestVertexConnection2D:
    def test_initialization(self):
        connection = [(0, 0, "start"), (1, 2, "end")]
        vertex_conn = VertexConnection2D(connection)
        assert vertex_conn.connection == connection

    def test_num_connected_subdomains(self):
        connection = [(0, 0, "start"), (1, 2, "end")]
        vertex_conn = VertexConnection2D(connection)
        assert vertex_conn.num_connected_subdomains == 2

    def test_validate_with_valid_connection(self, sample_subdomains):
        connection = [(0, 0, "start"), (2, 1, "end")]
        vertex_conn = VertexConnection2D(connection)
        vertex_conn.validate(sample_subdomains)

    def test_validate_with_one_connected_subdomain(self, sample_subdomains):
        connection = [(0, 0, "start")]
        vertex_conn = VertexConnection2D(connection)
        with pytest.raises(ValueError, match="Vertex connection must have at least two connected subdomains."):
            vertex_conn.validate([sample_subdomains[0]])

    def test_validate_with_invalid_subdomain_index(self, sample_subdomains):
        connection = [(3, 0, "start"), (1, 2, "end")]
        vertex_conn = VertexConnection2D(connection)
        with pytest.raises(ValueError, match="Subdomain index 3 is out of bounds"):
            vertex_conn.validate(sample_subdomains)

    def test_validate_with_invalid_curve_index(self, sample_subdomains):
        connection = [(0, 4, "start"), (1, 2, "end")]
        vertex_conn = VertexConnection2D(connection)
        with pytest.raises(ValueError, match="Curve index 4 is out of bounds"):
            vertex_conn.validate(sample_subdomains)

    def test_validate_with_invalid_position(self, sample_subdomains):
        connection = [(0, 0, "invalid"), (1, 2, "end")]
        vertex_conn = VertexConnection2D(connection)
        with pytest.raises(ValueError):
            vertex_conn.validate(sample_subdomains)

    def test_validate_with_non_matching_vertices(self, sample_subdomains):
        connection = [(0, 0, "start"), (1, 0, "start")]
        vertex_conn = VertexConnection2D(connection)
        with pytest.raises(ValueError, match="does not match the vertex"):
            vertex_conn.validate(sample_subdomains)

    def test_get_connection_pairs(self):
        connection = [(0, 0, "start"), (1, 2, "end"), (2, 1, "start")]
        vertex_conn = VertexConnection2D(connection)
        pairs = vertex_conn.get_connection_pairs()
        
        assert len(pairs) == 3
        assert ((0, 1), (0, 2), ("start", "end")) in pairs
        assert ((0, 2), (0, 1), ("start", "start")) in pairs
        assert ((1, 2), (2, 1), ("end", "start")) in pairs

    def test_get_shared_vertex(self, sample_subdomains):
        connection = [(0, 0, "start"), (2, 1, "end")]
        vertex_conn = VertexConnection2D(connection)
        vertex = vertex_conn.get_shared_vertex(sample_subdomains)
        
        assert np.allclose(vertex, np.array([0.0, 0.0]))

    def test_repr(self):
        connection = [(0, 0, "start"), (1, 2, "end")]
        vertex_conn = VertexConnection2D(connection)
        assert repr(vertex_conn) == f"VertexConnection2D({connection})"


class TestCurveConnection2D:
    def test_initialization(self):
        subdomains_indices = (0, 1)
        curve_indices = (0, 1)
        curve_conn = CurveConnection2D(subdomains_indices, curve_indices)
        assert curve_conn.subdomains_indices == subdomains_indices
        assert curve_conn.curve_indices == curve_indices

    def test_num_connected_subdomains(self):
        curve_conn = CurveConnection2D((0, 1), (1, 3))
        assert curve_conn.num_connected_subdomains == 2

    def test_validate_with_valid_connection(self, sample_subdomains):
        curve_conn = CurveConnection2D((0, 1), (1, 3))
        curve_conn.validate(sample_subdomains)

    def test_validate_with_invalid_subdomain_index(self, sample_subdomains):
        curve_conn = CurveConnection2D((3, 1), (0, 0))
        with pytest.raises(ValueError, match="Subdomain index 3 is out of bounds"):
            curve_conn.validate(sample_subdomains)

    def test_validate_with_invalid_subdomain_index_2(self, sample_subdomains):
        curve_conn = CurveConnection2D((0, 1), (1, 3))
        with pytest.raises(ValueError, match="Subdomain index 1 is out of bounds"):
            curve_conn.validate([sample_subdomains[0]])

    def test_validate_with_invalid_curve_index(self, sample_subdomains):
        curve_conn = CurveConnection2D((0, 1), (4, 0))
        with pytest.raises(ValueError, match="Curve index 4 is out of bounds"):
            curve_conn.validate(sample_subdomains)

    def test_validate_with_invalid_curve_index_2(self, sample_subdomains):
        curve_conn = CurveConnection2D((0, 1), (1, 4))
        with pytest.raises(ValueError, match="Curve index 4 is out of bounds"):
            curve_conn.validate(sample_subdomains)

    def test_validate_with_non_matching_curves(self, sample_subdomains):
        curve_conn = CurveConnection2D((0, 1), (1, 1))
        with pytest.raises(ValueError, match="are not equal"):
            curve_conn.validate(sample_subdomains)

    def test_get_shared_curve(self, sample_subdomains):
        curve_conn = CurveConnection2D((0, 1), (1, 3))
        shared_curve = curve_conn.get_shared_curve(sample_subdomains)
        
        assert isinstance(shared_curve, Line2D)
        assert np.allclose(shared_curve.start, np.array([3., 0.]))
        assert np.allclose(shared_curve.end, np.array([3., 1.0]))

    def test_repr(self):
        curve_conn = CurveConnection2D((0, 1), (1, 3))
        assert repr(curve_conn) == "CurveConnection((0, 1), (1, 3))" 
