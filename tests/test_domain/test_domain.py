import matplotlib.pyplot as plt
import pytest

from ttfemesh.domain.boundary_condition import DirichletBoundary2D
from ttfemesh.domain.domain import Domain, Domain2D
from ttfemesh.domain.subdomain_connection import CurveConnection2D, VertexConnection2D
from ttfemesh.domain.subdomain_factory import RectangleFactory


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
    def sample_subdomains(self):
        subdomain1 = RectangleFactory.create((0.0, 0.0), (3.0, 1.0))
        subdomain2 = RectangleFactory.create((3.0, 0.0), (4.0, 1.0))
        subdomain3 = RectangleFactory.create((-1.0, -1.0), (0.0, 0.0))
        return [subdomain1, subdomain2, subdomain3]

    @pytest.fixture
    def sample_connections(self):
        curve_conn = CurveConnection2D((0, 1), (1, 3))
        vertex_conn = VertexConnection2D([(0, 0, "start"), (2, 1, "end")])
        return [curve_conn, vertex_conn]

    @pytest.fixture
    def sample_boundary_condition(self):
        return DirichletBoundary2D([(0, 0), (1, 1)])

    @pytest.fixture
    def sample_test_domain(self):
        class TestDomain(Domain):
            def __init__(self, subdomains, connections, boundary_condition=None):
                super().__init__(subdomains, connections, boundary_condition)

            @property
            def dimension(self):
                return 2

        return TestDomain

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
        assert domain.num_subdomains == 3

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
        assert domain.get_subdomain(2) == sample_subdomains[2]

    def test_get_subdomain_invalid_index(self, sample_subdomains, sample_connections):
        domain = Domain2D(sample_subdomains, sample_connections)
        with pytest.raises(ValueError, match="Invalid subdomain index"):
            domain.get_subdomain(3)
        with pytest.raises(ValueError, match="Invalid subdomain index"):
            domain.get_subdomain(-1)

    def test_validate(self, sample_subdomains, sample_connections, sample_boundary_condition):
        domain = Domain2D(sample_subdomains, sample_connections, sample_boundary_condition)
        domain.validate()

    def test_validate_invalid_connection(self, sample_subdomains):
        invalid_conn = VertexConnection2D([(0, 5, "start"), (1, 2, "end")])
        with pytest.raises(ValueError):
            Domain2D(sample_subdomains, [invalid_conn])

    def test_validate_invalid_boundary_condition(self, sample_subdomains, sample_connections):
        invalid_bc = DirichletBoundary2D([(3, 0)])
        with pytest.raises(ValueError):
            Domain2D(sample_subdomains, sample_connections, invalid_bc)

    def test_dimension(self, sample_subdomains, sample_connections):
        domain = Domain2D(sample_subdomains, sample_connections)
        assert domain.dimension == 2

    def test_repr(self, sample_subdomains, sample_connections):
        domain = Domain2D(sample_subdomains, sample_connections)
        assert repr(domain) == "Domain2D(3 subdomains, 2 connections)"

    def test_repr_domain(self, sample_subdomains, sample_connections, sample_test_domain):
        domain = sample_test_domain(sample_subdomains, sample_connections)
        assert repr(domain) == "Domain(3 subdomains, 2 connections)"

    def test_plot(self, sample_subdomains, sample_connections, sample_boundary_condition):
        domain = Domain2D(sample_subdomains, sample_connections, sample_boundary_condition)

        try:
            domain.plot()
            plt.close()
        except Exception as e:
            pytest.fail(f"Plotting failed with error: {e}")

    def test_plot_domain(self, sample_subdomains, sample_connections, sample_test_domain):
        domain = sample_test_domain(sample_subdomains, sample_connections)
        try:
            domain.plot()
            plt.close()
        except Exception as e:
            pytest.fail(f"Plotting failed with error: {e}")

    def test_plot_with_num_points(self, sample_subdomains, sample_connections):
        domain = Domain2D(sample_subdomains, sample_connections)

        try:
            domain.plot(num_points=50)
            plt.close()
        except Exception as e:
            pytest.fail(f"Plotting failed with error: {e}")
