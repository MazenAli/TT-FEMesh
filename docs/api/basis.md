# Basis Module

The basis module provides classes and utilities for working with basis functions in the finite element method. It includes support for linear and bilinear basis functions, as well as tensor product basis functions.

## Classes

### Basis Classes

#### `TensorProductBasis`
Base class for tensor product basis functions.

```python
from ttfemesh.basis import TensorProductBasis

basis = TensorProductBasis(dim, basis_type)
```

**Parameters:**
- `dim` (int): The dimension of the basis functions
- `basis_type` (str): The type of basis functions to use

#### `LinearBasis`
Linear basis functions for 1D elements.

```python
from ttfemesh.basis import LinearBasis

basis = LinearBasis()
```

**Methods:**
- `evaluate(x)`: Evaluate the basis functions at point x
- `derivative(x)`: Evaluate the derivatives of the basis functions at point x

#### `BilinearBasis`
Bilinear basis functions for 2D elements.

```python
from ttfemesh.basis import BilinearBasis

basis = BilinearBasis()
```

**Methods:**
- `evaluate(x, y)`: Evaluate the basis functions at point (x, y)
- `derivative(x, y)`: Evaluate the derivatives of the basis functions at point (x, y)

## Examples

### Using Linear Basis Functions

```python
from ttfemesh.basis import LinearBasis
import numpy as np

# Create a linear basis
basis = LinearBasis()

# Evaluate basis functions at a point
x = 0.5
values = basis.evaluate(x)
print(f"Basis function values at x={x}: {values}")

# Evaluate derivatives at a point
derivatives = basis.derivative(x)
print(f"Basis function derivatives at x={x}: {derivatives}")
```

### Using Bilinear Basis Functions

```python
from ttfemesh.basis import BilinearBasis
import numpy as np

# Create a bilinear basis
basis = BilinearBasis()

# Evaluate basis functions at a point
x, y = 0.5, 0.5
values = basis.evaluate(x, y)
print(f"Basis function values at (x,y)=({x},{y}): {values}")

# Evaluate derivatives at a point
derivatives = basis.derivative(x, y)
print(f"Basis function derivatives at (x,y)=({x},{y}): {derivatives}")
```

### Using Tensor Product Basis Functions

```python
from ttfemesh.basis import TensorProductBasis, BilinearBasis

# Create a tensor product basis using bilinear basis functions
basis = TensorProductBasis(dim=2, basis_type=BilinearBasis)

# The tensor product basis can be used in the same way as the component basis
x, y = 0.5, 0.5
values = basis.evaluate(x, y)
print(f"Tensor product basis function values at (x,y)=({x},{y}): {values}")
```

## Related Modules

- [Mesh](mesh.md) - For generating meshes that use these basis functions
- [Quadrature](quadrature.md) - For numerical integration using these basis functions
- [Domain](domain.md) - For creating domains that will be discretized using these basis functions 