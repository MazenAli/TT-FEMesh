import pytest

from ttfemesh.domain.boundary_condition import BoundaryCondition, DirichletBoundary2D
from ttfemesh.domain.curve import Line2D
from ttfemesh.domain.subdomain import Subdomain2D


class TestBoundaryCondition:
    def test_abstract_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BoundaryCondition()

    def test_abstract_methods_must_be_implemented(self):
        class IncompleteBoundaryCondition(BoundaryCondition):
            def validate(self):
                pass

        with pytest.raises(TypeError):
            IncompleteBoundaryCondition()


class TestDirichletBoundary2D:
    @pytest.fixture
    def sample_curves(self):
        p1 = (0.0, 0.0)
        p2 = (1.0, 0.0)
        p3 = (1.0, 1.0)
        p4 = (0.0, 1.0)

        curve1 = Line2D(p1, p2)
        curve2 = Line2D(p2, p3)
        curve3 = Line2D(p3, p4)
        curve4 = Line2D(p4, p1)

        return [curve1, curve2, curve3, curve4]

    @pytest.fixture
    def sample_subdomains(self, sample_curves):
        subdomain1 = Subdomain2D(sample_curves)
        subdomain2 = Subdomain2D(sample_curves)
        return [subdomain1, subdomain2]

    def test_initialization(self):
        boundary = [(0, 0), (1, 1)]
        bc = DirichletBoundary2D(boundary)
        assert bc.boundary == boundary

    def test_validate_with_valid_indices(self, sample_subdomains):
        boundary = [(0, 0), (1, 1)]
        bc = DirichletBoundary2D(boundary)
        bc.validate(sample_subdomains)

    def test_validate_with_invalid_subdomain_index(self, sample_subdomains):
        boundary = [(2, 0)]
        bc = DirichletBoundary2D(boundary)
        with pytest.raises(ValueError, match="Subdomain index 2 out of range."):
            bc.validate(sample_subdomains)

    def test_validate_with_invalid_curve_index(self, sample_subdomains):
        boundary = [(0, 4)]
        bc = DirichletBoundary2D(boundary)
        with pytest.raises(ValueError, match="Curve index 4 out of range for subdomain 0."):
            bc.validate(sample_subdomains)

    def test_num_bcs(self):
        boundary = [(0, 0), (1, 1), (0, 2)]
        bc = DirichletBoundary2D(boundary)
        assert bc.num_bcs() == 3

    def test_group_by_subdomain(self):
        boundary = [(0, 0), (1, 1), (0, 2)]
        bc = DirichletBoundary2D(boundary)
        grouped = bc.group_by_subdomain()

        assert grouped == {0: [0, 2], 1: [1]}

    def test_repr(self):
        boundary = [(0, 0), (1, 1)]
        bc = DirichletBoundary2D(boundary)
        assert repr(bc) == "DirichletBoundary2D(num_bcs=2)"

    def test_empty_boundary(self):
        bc = DirichletBoundary2D([])
        assert bc.num_bcs() == 0
        assert bc.group_by_subdomain() == {}
        assert repr(bc) == "DirichletBoundary2D(num_bcs=0)"
