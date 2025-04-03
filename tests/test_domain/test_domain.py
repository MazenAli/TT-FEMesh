import pytest
import numpy as np
import matplotlib.pyplot as plt
from ttfemesh.domain.domain import Domain, Domain2D
from ttfemesh.domain.subdomain import Subdomain2D
from ttfemesh.domain.curve import Line2D
from ttfemesh.domain.boundary_condition import DirichletBoundary2D
from ttfemesh.domain.subdomain_connection import (
    SubdomainConnection2D,
    VertexConnection2D,
    CurveConnection2D,
)


class TestDomain:
    def test_abstract_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            Domain([], [], None)

    def test_abstract_methods_must_be_implemented(self):
        class IncompleteDomain(Domain):
            def __init__(self, subdomains, connections, boundary_condition):
                super().__init__(subdomains, connections, boundary_condition)

        with pytest.raises(TypeError):
            IncompleteDomain([], [], None)


class TestDomain2D:
    @pytest.fixture
    def sample_curves(self):
        # Create sample curves for testing
        curve1 = Line2D((0.0, 0.0), (1.0, 0.0))
        curve2 = Line2D((1.0, 0.0), (1.0, 1.0))
        curve3 = Line2D((1.0, 1.0), (0.0, 1.0))
        curve4 = Line2D((0.0, 1.0), (0.0, 0.0))
        return [curve1, curve2, curve3, curve4]

    @pytest.fixture
    def sample_subdomains(self, sample_curves):
        # Create sample subdomains
        subdomain1 = Subdomain2D(sample_curves)
        subdomain2 = Subdomain2D(sample_curves)
        return [subdomain1, subdomain2]

    @pytest.fixture
    def sample_connections(self, sample_subdomains):
        # Create sample connections
        vertex_conn = VertexConnection2D(0, 0, 1, 2)  # Connect vertex 0 of subdomain 0 to vertex 2 of subdomain 1
        curve_conn = CurveConnection2D(0, 1, 1, 3)    # Connect curve 1 of subdomain 0 to curve 3 of subdomain 1
        return [vertex_conn, curve_conn]

    @pytest.fixture
    def sample_boundary_condition(self):
        return DirichletBoundary2D([(0, 0), (1, 1)])

    def test_initialization(self, sample_subdomains, sample_connections, sample_boundary_condition):
        domain = Domain2D(sample_subdomains, sample_connections, sample_boundary_condition)
        assert domain.subdomains == sample_subdomains
        assert domain.connections == sample_connections
        assert domain.boundary_condition == sample_boundary_condition

    def test_initialization_without_boundary_condition(self, sample_subdomains, sample_connections):
        domain = Domain2D(sample_subdomains, sample_connections)
        assert domain.subdomains == sample_subdomains
        assert domain.connections == sample_connections
        assert domain.boundary_condition is None

    def test_num_subdomains(self, sample_subdomains, sample_connections):
        domain = Domain2D(sample_subdomains, sample_connections)
        assert domain.num_subdomains == 2

    def test_num_connections(self, sample_subdomains, sample_connections):
        domain = Domain2D(sample_subdomains, sample_connections)
        assert domain.num_connections == 2

    def test_get_connections(self, sample_subdomains, sample_connections):
        domain = Domain2D(sample_subdomains, sample_connections)
        assert domain.get_connections() == sample_connections

    def test_get_subdomain(self, sample_subdomains, sample_connections):
        domain = Domain2D(sample_subdomains, sample_connections)
        assert domain.get_subdomain(0) == sample_subdomains[0]
        assert domain.get_subdomain(1) == sample_subdomains[1]

    def test_get_subdomain_invalid_index(self, sample_subdomains, sample_connections):
        domain = Domain2D(sample_subdomains, sample_connections)
        with pytest.raises(ValueError, match="Invalid subdomain index"):
            domain.get_subdomain(2)
        with pytest.raises(ValueError, match="Invalid subdomain index"):
            domain.get_subdomain(-1)

    def test_validate(self, sample_subdomains, sample_connections, sample_boundary_condition):
        domain = Domain2D(sample_subdomains, sample_connections, sample_boundary_condition)
        domain.validate()  # Should not raise any errors

    def test_validate_invalid_connection(self, sample_subdomains):
        # Create an invalid connection
        invalid_conn = VertexConnection2D(0, 5, 1, 2)  # Invalid vertex index
        with pytest.raises(ValueError):
            Domain2D(sample_subdomains, [invalid_conn])

    def test_validate_invalid_boundary_condition(self, sample_subdomains, sample_connections):
        # Create an invalid boundary condition
        invalid_bc = DirichletBoundary2D([(2, 0)])  # Invalid subdomain index
        with pytest.raises(ValueError):
            Domain2D(sample_subdomains, sample_connections, invalid_bc)

    def test_dimension(self, sample_subdomains, sample_connections):
        domain = Domain2D(sample_subdomains, sample_connections)
        assert domain.dimension == 2

    def test_repr(self, sample_subdomains, sample_connections):
        domain = Domain2D(sample_subdomains, sample_connections)
        assert repr(domain) == "Domain2D(2 subdomains, 2 connections)"

    def test_plot(self, sample_subdomains, sample_connections, sample_boundary_condition):
        domain = Domain2D(sample_subdomains, sample_connections, sample_boundary_condition)
        
        # Test that plot runs without errors
        try:
            domain.plot()
            plt.close()
        except Exception as e:
            pytest.fail(f"Plotting failed with error: {e}")

    def test_plot_with_num_points(self, sample_subdomains, sample_connections):
        domain = Domain2D(sample_subdomains, sample_connections)
        
        # Test that plot runs with different num_points
        try:
            domain.plot(num_points=50)
            plt.close()
        except Exception as e:
            pytest.fail(f"Plotting failed with error: {e}") 