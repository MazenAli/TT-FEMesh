# Domain Module

The domain module provides classes and utilities for creating and managing 2D domains for finite element meshing. It includes support for creating subdomains, connecting them, and specifying boundary conditions.

## Classes

### Domain Classes

#### `Domain2D`
Base class for 2D domains.

```python
from ttfemesh.domain import Domain2D

domain = Domain2D(subdomains, connections, boundary_condition)
```

**Parameters:**
- `subdomains` (List[Subdomain2D]): List of subdomains in the domain
- `connections` (List[Union[CurveConnection2D, VertexConnection2D]]): List of connections between subdomains
- `boundary_condition` (DirichletBoundary2D): Boundary conditions for the domain

**Methods:**
- `plot()`: Visualize the domain

### Subdomain Classes

#### `Subdomain2D`
Base class for 2D subdomains.

#### `Quad`
A quadrilateral subdomain defined by four vertices.

### Factory Classes

#### `RectangleFactory`
Factory class for creating rectangular subdomains.

```python
from ttfemesh.domain import RectangleFactory

rectangle = RectangleFactory.create(lower_left, upper_right)
```

**Parameters:**
- `lower_left` (Tuple[float, float]): Coordinates of the lower-left corner
- `upper_right` (Tuple[float, float]): Coordinates of the upper-right corner

#### `QuadFactory`
Factory class for creating quadrilateral subdomains.

### Connection Classes

#### `CurveConnection2D`
Represents a connection between two subdomains along a curve.

```python
from ttfemesh.domain import CurveConnection2D

connection = CurveConnection2D(domain_idxs, curve_idxs)
```

**Parameters:**
- `domain_idxs` (List[int]): Indices of the connected subdomains
- `curve_idxs` (List[int]): Indices of the curves in each subdomain

#### `VertexConnection2D`
Represents a connection between two subdomains at a vertex.

### Boundary Condition Classes

#### `DirichletBoundary2D`
Specifies Dirichlet boundary conditions for a domain.

```python
from ttfemesh.domain import DirichletBoundary2D

bc = DirichletBoundary2D(boundary_edges)
```

**Parameters:**
- `boundary_edges` (List[Tuple[int, int]]): List of (subdomain_idx, edge_idx) tuples specifying the boundary edges

### Curve Classes

#### `Curve`
Base class for curves.

#### `Line2D`
A straight line segment in 2D.

#### `CircularArc2D`
A circular arc in 2D.

#### `ParametricCurve2D`
A parametric curve in 2D.

## Examples

### Creating a Simple Domain

```python
from ttfemesh.domain import RectangleFactory, CurveConnection2D, DirichletBoundary2D, Domain2D

# Create two rectangles
rectangle1 = RectangleFactory.create((0, 0), (2, 1))
rectangle2 = RectangleFactory.create((2, 0), (3, 1))

# Connect them
connection = CurveConnection2D([0, 1], [1, 3])

# Set boundary conditions
bc = DirichletBoundary2D([(0, 3), (1, 1)])

# Create the domain
domain = Domain2D([rectangle1, rectangle2], [connection], bc)
domain.plot()
```

### Creating a Domain with a Circular Arc

```python
from ttfemesh.domain import RectangleFactory, CircularArc2D, CurveConnection2D, DirichletBoundary2D, Domain2D

# Create a rectangle
rectangle = RectangleFactory.create((0, 0), (1, 1))

# Create a circular arc
arc = CircularArc2D(center=(1, 0.5), radius=0.5, start_angle=0, end_angle=180)

# Connect them
connection = CurveConnection2D([0, 1], [1, 0])

# Set boundary conditions
bc = DirichletBoundary2D([(0, 0), (0, 2)])

# Create the domain
domain = Domain2D([rectangle, arc], [connection], bc)
domain.plot()
```

## Related Modules

- [Mesh](mesh.md) - For generating meshes from domains
- [Basis](basis.md) - For basis functions used in the finite element method
- [Quadrature](quadrature.md) - For numerical integration rules 