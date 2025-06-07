# Quadrature Module

The quadrature module provides classes and utilities for numerical integration in the finite element method. It includes support for Gauss-Legendre quadrature rules in both 1D and 2D.

## Classes

### Quadrature Rules

#### `QuadratureRule`
Base class for quadrature rules.

```python
from ttfemesh.quadrature import QuadratureRule

rule = QuadratureRule(order)
```

**Parameters:**
- `order` (int): The order of the quadrature rule

**Methods:**
- `get_points()`: Get the quadrature points
- `get_weights()`: Get the quadrature weights

#### `GaussLegendre`
1D Gauss-Legendre quadrature rule.

```python
from ttfemesh.quadrature import GaussLegendre

rule = GaussLegendre(order)
```

**Parameters:**
- `order` (int): The order of the quadrature rule (number of points will be order + 1)

**Methods:**
- `get_points()`: Get the 1D quadrature points
- `get_weights()`: Get the 1D quadrature weights

#### `GaussLegendre2D`
2D Gauss-Legendre quadrature rule.

```python
from ttfemesh.quadrature import GaussLegendre2D

rule = GaussLegendre2D(order)
```

**Parameters:**
- `order` (int): The order of the quadrature rule (number of points in each dimension will be order + 1)

**Methods:**
- `get_points()`: Get the 2D quadrature points
- `get_weights()`: Get the 2D quadrature weights

## Examples

### Using 1D Gauss-Legendre Quadrature

```python
from ttfemesh.quadrature import GaussLegendre
import numpy as np

# Create a 1D Gauss-Legendre quadrature rule
order = 2
rule = GaussLegendre(order)

# Get quadrature points and weights
points = rule.get_points()
weights = rule.get_weights()

print(f"Quadrature points: {points}")
print(f"Quadrature weights: {weights}")

# Example: Integrate x^2 from -1 to 1
def f(x):
    return x**2

integral = np.sum(weights * f(points))
print(f"Integral of x^2 from -1 to 1: {integral}")
```

### Using 2D Gauss-Legendre Quadrature

```python
from ttfemesh.quadrature import GaussLegendre2D
import numpy as np

# Create a 2D Gauss-Legendre quadrature rule
order = 2
rule = GaussLegendre2D(order)

# Get quadrature points and weights
points = rule.get_points()
weights = rule.get_weights()

print(f"Quadrature points shape: {points.shape}")
print(f"Quadrature weights shape: {weights.shape}")

# Example: Integrate x^2 + y^2 over [-1,1] x [-1,1]
def f(x, y):
    return x**2 + y**2

integral = np.sum(weights * f(points[:, 0], points[:, 1]))
print(f"Integral of x^2 + y^2 over [-1,1] x [-1,1]: {integral}")
```

### Using Quadrature in Finite Element Method

```python
from ttfemesh.quadrature import GaussLegendre2D
from ttfemesh.basis import BilinearBasis
import numpy as np

# Create quadrature rule and basis functions
order = 2
qrule = GaussLegendre2D(order)
basis = BilinearBasis()

# Get quadrature points and weights
points = qrule.get_points()
weights = qrule.get_weights()

# Evaluate basis functions at quadrature points
basis_values = np.array([basis.evaluate(x, y) for x, y in points])

# Example: Compute mass matrix
mass_matrix = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        mass_matrix[i, j] = np.sum(weights * basis_values[:, i] * basis_values[:, j])

print("Mass matrix:")
print(mass_matrix)
```

## Related Modules

- [Basis](basis.md) - For basis functions used in numerical integration
- [Mesh](mesh.md) - For generating meshes that use these quadrature rules
- [Domain](domain.md) - For creating domains that will be integrated over 