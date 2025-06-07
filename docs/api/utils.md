# Utils Module

The utils module provides utility functions used throughout the TTFEMesh library.

## Functions

### `check_tensor_train`
Validates a tensor train representation.

```python
from ttfemesh.utils import check_tensor_train

is_valid = check_tensor_train(tt)
```

**Parameters:**
- `tt`: The tensor train to validate

**Returns:**
- `bool`: Whether the tensor train is valid

### `check_tensor_train_matrix`
Validates a tensor train matrix representation.

```python
from ttfemesh.utils import check_tensor_train_matrix

is_valid = check_tensor_train_matrix(tt_matrix)
```

**Parameters:**
- `tt_matrix`: The tensor train matrix to validate

**Returns:**
- `bool`: Whether the tensor train matrix is valid

### `compute_tensor_train_rank`
Computes the rank of a tensor train representation.

```python
from ttfemesh.utils import compute_tensor_train_rank

rank = compute_tensor_train_rank(tt)
```

**Parameters:**
- `tt`: The tensor train to compute the rank of

**Returns:**
- `int`: The rank of the tensor train

### `compute_tensor_train_size`
Computes the size of a tensor train representation.

```python
from ttfemesh.utils import compute_tensor_train_size

size = compute_tensor_train_size(tt)
```

**Parameters:**
- `tt`: The tensor train to compute the size of

**Returns:**
- `int`: The size of the tensor train

## Related Modules

- [Tensor Train Tools](tt_tools.md) - For tensor train operations
- [Mesh](mesh.md) - For mesh-related utilities
- [Types](types.md) - For type definitions used by utilities 