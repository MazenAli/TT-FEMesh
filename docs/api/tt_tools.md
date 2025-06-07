# Tensor Train Tools Module

The tensor train tools module provides utilities for working with tensor train representations, including interpolation, mesh grid operations, and tensor train cross approximation.

## Functions

### Tensor Train Cross Approximation

#### `tensor_train_cross_approximation`
Performs tensor train cross approximation on a given function.

```python
from ttfemesh.tt_tools import tensor_train_cross_approximation, TTCrossConfig

config = TTCrossConfig(
    max_rank=10,
    max_iter=100,
    tol=1e-6
)
tt = tensor_train_cross_approximation(func, config)
```

**Parameters:**
- `func`: The function to approximate
- `config` (TTCrossConfig): Configuration for the approximation

**Returns:**
- Tensor train representation of the function

#### `anova_init_tensor_train`
Initializes a tensor train using ANOVA decomposition.

#### `error_on_indices`
Computes the approximation error on specific indices.

#### `error_on_random_indices`
Computes the approximation error on random indices.

### Mesh Grid Operations

#### `range_meshgrid2d`
Creates a 2D mesh grid with given ranges.

```python
from ttfemesh.tt_tools import range_meshgrid2d

x, y = range_meshgrid2d(x_range, y_range)
```

**Parameters:**
- `x_range` (Tuple[float, float]): Range for x coordinates
- `y_range` (Tuple[float, float]): Range for y coordinates

**Returns:**
- Tuple of (x, y) mesh grids

#### `zmeshgrid2d`
Creates a 2D mesh grid in Z-order.

#### `map2canonical2d`
Maps coordinates to canonical domain.

### Interpolation

#### `interpolate_linear2d`
Performs linear interpolation in 2D.

```python
from ttfemesh.tt_tools import interpolate_linear2d

values = interpolate_linear2d(points, values, query_points)
```

**Parameters:**
- `points` (np.ndarray): Input points
- `values` (np.ndarray): Values at input points
- `query_points` (np.ndarray): Points to interpolate at

**Returns:**
- Interpolated values

### Tensor Operations

#### `levelwise_kron`
Computes level-wise Kronecker product.

#### `transpose_kron`
Computes transposed Kronecker product.

#### `zorder_kron`
Computes Z-order Kronecker product.

#### `unit_vector_binary_tt`
Creates a unit vector in binary tensor train format.

## Examples

### Tensor Train Cross Approximation

```python
from ttfemesh.tt_tools import tensor_train_cross_approximation, TTCrossConfig
import numpy as np

# Define a function to approximate
def f(x, y):
    return np.sin(x) * np.cos(y)

# Configure the approximation
config = TTCrossConfig(
    max_rank=10,
    max_iter=100,
    tol=1e-6
)

# Perform the approximation
tt = tensor_train_cross_approximation(f, config)

# Evaluate the approximation
x, y = 0.5, 0.5
value = tt.evaluate(x, y)
print(f"Approximated value at (x,y)=({x},{y}): {value}")
```

### Mesh Grid Operations

```python
from ttfemesh.tt_tools import range_meshgrid2d, zmeshgrid2d
import numpy as np

# Create a regular mesh grid
x_range = (-1, 1)
y_range = (-1, 1)
x, y = range_meshgrid2d(x_range, y_range)

# Create a Z-order mesh grid
z_x, z_y = zmeshgrid2d(x_range, y_range)

print(f"Regular grid shape: {x.shape}")
print(f"Z-order grid shape: {z_x.shape}")
```

### Interpolation

```python
from ttfemesh.tt_tools import interpolate_linear2d
import numpy as np

# Create input points and values
points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
values = np.array([0, 1, 1, 2])

# Create query points
query_points = np.array([[0.5, 0.5], [0.25, 0.75]])

# Perform interpolation
interpolated = interpolate_linear2d(points, values, query_points)
print(f"Interpolated values: {interpolated}")
```

## Related Modules

- [Mesh](mesh.md) - For generating meshes that use tensor train representations
- [Basis](basis.md) - For basis functions used in tensor train approximations
- [Quadrature](quadrature.md) - For numerical integration using tensor train representations 