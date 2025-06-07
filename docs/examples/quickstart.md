# Quickstart Guide

This guide provides a quick introduction to using TTFEMesh. For detailed information about the meaning of the tensorized Jacobians, Dirichlet masks, and concatenation maps, we refer to the original paper [arXiv:1802.02839](https://arxiv.org/abs/1802.02839).


## Installation

```bash
pip install ttfemesh
```

## Creating a Domain

We create a simple domain with two rectangles and an edge connecting them. We set the Dirichlet boundary for the left side and the right side.

```python
from ttfemesh.domain import RectangleFactory, CurveConnection2D, VertexConnection2D 
from ttfemesh.domain import DirichletBoundary2D, Domain2D

# Create first rectangle
lower_left = (0, 0)
upper_right = (2, 1)
rectangle1 = RectangleFactory.create(lower_left, upper_right)

# Create second rectangle
lower_left = (2, 0)
upper_right = (3, 1)
rectangle2 = RectangleFactory.create(lower_left, upper_right)

# Connect the rectangles
domain_idxs = [0, 1]
curve_idxs = [1, 3]
edge = CurveConnection2D(domain_idxs, curve_idxs)

# Set boundary conditions
bc = DirichletBoundary2D([(0, 3), (1, 1)])

# Create the domain
domain = Domain2D([rectangle1, rectangle2], [edge], bc)
domain.plot()
```

## Meshing a Domain

Generating a mesh for a domain is straightforward. It requires the domain, a quadrature rule, and a mesh size exponent. The mesh size exponent is used to control the size of the mesh, i.e., the number of discretization points in each direction will be $2^{n}$, where $n$ is the mesh size exponent.

```python
from ttfemesh.quadrature import GaussLegendre2D
from ttfemesh.mesh import DomainBilinearMesh2D

order = 1
qrule = GaussLegendre2D(order)
mesh_size_exponent = 3

mesh = DomainBilinearMesh2D(domain, qrule, mesh_size_exponent)
print(mesh)
```

## Working with Tensor Trains

### Jacobians and Determinants

For each of the subdomains, you can retrieve the tensorized Jacobians:

```python
subdmesh = mesh.get_subdomain_mesh(0)
jac_tts = subdmesh.get_jacobian_tensor_trains()
print(jac_tts.shape)
print(jac_tts)

jac_dets = subdmesh.get_jacobian_det_tensor_trains()
print(jac_dets.shape)
print(jac_dets)

jac_invdets = subdmesh.get_jacobian_invdet_tensor_trains()
print(jac_invdets.shape)
print(jac_invdets)
```

### Element to Global Index Map

You can retrieve the tensorized element to global index map:

```python
element2global_map = mesh.get_element2global_index_map()
print(element2global_map.shape)
print(element2global_map)
```

### Dirichlet Masks

For each of the subdomains, you can retrieve the tensorized Dirichlet masks:

```python
masks = mesh.get_dirichlet_masks()
print(masks)
```

### Concatenation Maps

You can retrieve the concatenation maps for all pairs of connected subdomains:

```python
concat_maps = mesh.get_concatenation_maps()
print(concat_maps)
```

## Next Steps

- Check out the [API Reference](../api/index.md) for detailed documentation of all available classes and functions
- Visit our [GitHub repository](https://github.com/MazenAli/TT-FEMesh) to contribute or report issues 