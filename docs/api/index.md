# API Reference

This section provides detailed documentation for all the modules and classes in TTFEMesh.

```{toctree}
:maxdepth: 2
:caption: API Reference

domain
mesh
basis
quadrature
tt_tools
types
utils
```

## Core Modules

- [Domain](domain.md) - Domain creation and management
  - `RectangleFactory`
  - `CurveConnection2D`
  - `VertexConnection2D`
  - `DirichletBoundary2D`
  - `Domain2D`

- [Mesh](mesh.md) - Mesh generation and operations
  - `DomainBilinearMesh2D`
  - `SubdomainMesh2D`

- [Basis](basis.md) - Basis functions and operations
  - `TensorProductBasis`
  - `BilinearBasis`

- [Quadrature](quadrature.md) - Quadrature rules
  - `GaussLegendre2D`

- [Tensor Train Tools](tt_tools.md) - Tensor train operations and utilities
  - Various tensor train manipulation functions

## Utility Modules

- [Types](types.md) - Type definitions and constants
- [Utils](utils.md) - General utility functions

## Examples

For practical examples of how to use these modules, please refer to the [Examples](../examples/index.md) section. 