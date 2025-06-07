# Mesh Module

The mesh module provides classes and utilities for generating and managing finite element meshes. It includes support for creating domain meshes, subdomain meshes, and various mesh-related operations.

## Classes

### Domain Mesh Classes

#### `DomainMesh2D`
Base class for 2D domain meshes.

#### `DomainBilinearMesh2D`
A domain mesh using bilinear basis functions.

```python
from ttfemesh.mesh import DomainBilinearMesh2D
from ttfemesh.quadrature import GaussLegendre2D

mesh = DomainBilinearMesh2D(domain, quadrature_rule, mesh_size_exponent)
```

**Parameters:**
- `domain` (Domain2D): The domain to mesh
- `quadrature_rule` (GaussLegendre2D): The quadrature rule to use
- `mesh_size_exponent` (int): The mesh size exponent (number of discretization points will be 2^n)

**Methods:**
- `get_subdomain_mesh(subdomain_idx)`: Get the mesh for a specific subdomain
- `get_element2global_index_map()`: Get the element to global index mapping
- `get_dirichlet_masks()`: Get the Dirichlet masks for all subdomains
- `get_concatenation_maps()`: Get the concatenation maps for connected subdomains

### Subdomain Mesh Classes

#### `SubdomainMesh2D`
Base class for 2D subdomain meshes.

#### `QuadMesh`
A mesh for quadrilateral subdomains.

**Methods:**
- `get_jacobian_tensor_trains()`: Get the tensorized Jacobians
- `get_jacobian_det_tensor_trains()`: Get the tensorized Jacobian determinants
- `get_jacobian_invdet_tensor_trains()`: Get the tensorized inverse Jacobian determinants

### Utility Functions

#### `bindex2dtuple`
Convert a basis index to a tuple of indices.

#### `qindex2dtuple`
Convert a quadrature index to a tuple of indices.

## Examples

### Creating and Using a Domain Mesh

```python
from ttfemesh.domain import RectangleFactory, CurveConnection2D, DirichletBoundary2D, Domain2D
from ttfemesh.mesh import DomainBilinearMesh2D
from ttfemesh.quadrature import GaussLegendre2D

# Create a domain
rectangle1 = RectangleFactory.create((0, 0), (2, 1))
rectangle2 = RectangleFactory.create((2, 0), (3, 1))
connection = CurveConnection2D([0, 1], [1, 3])
bc = DirichletBoundary2D([(0, 3), (1, 1)])
domain = Domain2D([rectangle1, rectangle2], [connection], bc)

# Create a mesh
order = 1
qrule = GaussLegendre2D(order)
mesh_size_exponent = 3
mesh = DomainBilinearMesh2D(domain, qrule, mesh_size_exponent)

# Get subdomain mesh and tensorized quantities
subdmesh = mesh.get_subdomain_mesh(0)
jac_tts = subdmesh.get_jacobian_tensor_trains()
jac_dets = subdmesh.get_jacobian_det_tensor_trains()
jac_invdets = subdmesh.get_jacobian_invdet_tensor_trains()

# Get global mappings
element2global_map = mesh.get_element2global_index_map()
masks = mesh.get_dirichlet_masks()
concat_maps = mesh.get_concatenation_maps()
```

### Working with Tensor Trains

The mesh module provides tensor train representations of various quantities:

```python
# Get Jacobian tensor trains
jac_tts = subdmesh.get_jacobian_tensor_trains()
print(f"Jacobian tensor trains shape: {jac_tts.shape}")

# Get Jacobian determinant tensor trains
jac_dets = subdmesh.get_jacobian_det_tensor_trains()
print(f"Jacobian determinant tensor trains shape: {jac_dets.shape}")

# Get inverse Jacobian determinant tensor trains
jac_invdets = subdmesh.get_jacobian_invdet_tensor_trains()
print(f"Inverse Jacobian determinant tensor trains shape: {jac_invdets.shape}")
```

## Related Modules

- [Domain](domain.md) - For creating and managing domains
- [Basis](basis.md) - For basis functions used in the mesh
- [Quadrature](quadrature.md) - For numerical integration rules
- [Tensor Train Tools](tt_tools.md) - For working with tensor train representations 