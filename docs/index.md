# TTFEMesh

[![GitHub](https://img.shields.io/badge/GitHub-TT--FEMesh-black?style=flat&logo=github)](https://github.com/MazenAli/TT-FEMesh)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)

TTFEMesh is a Python library for generating tensor train representations of finite element meshes.
It provides tools for creating domains, generating meshes, and computing tensorized Jacobians, Dirichlet masks, and concatenation maps.

This provides the backbone for solving partial
differential equations using TT decompositions. The package is designed
to be flexible and extensible, allowing users to create their own finite element
meshes and build their own solvers.

The package is built on top of the [`torchtt`](https://github.com/ion-g-ion/torchTT) package, which provides a framework
for handling TT decompositions, but this might change in the future.
This package is currently in a prototype stage, and the API is subject to change.

For more technical details on how to manipulate finite element meshes in the tensor train format,
please refer to the [original paper](https://arxiv.org/abs/1802.02839) and the
extensions in [this work](https://www.mdpi.com/2227-7390/12/20/3277).

## Installation

### System Dependencies

TTFEMesh requires BLAS and LAPACK libraries to be installed on your system:

- **Ubuntu/Debian**:
  ```bash
  sudo apt-get update
  sudo apt-get install libblas-dev liblapack-dev
  ```

- **macOS** (using Homebrew):
  ```bash
  brew install openblas lapack
  ```

- **Windows**:
  These libraries are typically included with scientific Python distributions like Anaconda.

### Python Package

You can install TTFEMesh using pip:

```bash
pip install ttfemesh
```

## Documentation

```{toctree}
:maxdepth: 1
:caption: User Guide

examples/quickstart
```

```{toctree}
:maxdepth: 1
:caption: Core Modules

api/domain
api/mesh
api/quadrature
api/basis
api/tt_tools
api/types
api/utils
```

```{toctree}
:maxdepth: 1
:caption: API Reference

api/reference
```

## Quick Start

Here's a quick example of how to use TTFEMesh:

```python
from ttfemesh.domain import RectangleFactory, CurveConnection2D, DirichletBoundary2D, Domain2D
from ttfemesh.quadrature import GaussLegendre2D
from ttfemesh.mesh import DomainBilinearMesh2D

# Create a domain with two rectangles
lower_left = (0, 0)
upper_right = (2, 1)
rectangle1 = RectangleFactory.create(lower_left, upper_right)

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

# Generate a mesh
order = 1
qrule = GaussLegendre2D(order)
mesh_size_exponent = 3
mesh = DomainBilinearMesh2D(domain, qrule, mesh_size_exponent)

# Get tensorized Jacobians
subdmesh = mesh.get_subdomain_mesh(0)
jac_tts = subdmesh.get_jacobian_tensor_trains()
jac_dets = subdmesh.get_jacobian_det_tensor_trains()
jac_invdets = subdmesh.get_jacobian_invdet_tensor_trains()

# Get element to global index map
element2global_map = mesh.get_element2global_index_map()

# Get Dirichlet masks
masks = mesh.get_dirichlet_masks()

# Get concatenation maps
concat_maps = mesh.get_concatenation_maps()
```

For more detailed examples and tutorials, check out our [Quickstart Guide](examples/quickstart.md).

## Contributing

We welcome contributions! Please visit our [GitHub repository](https://github.com/MazenAli/TT-FEMesh) to:
- Report bugs
- Suggest features
- Submit pull requests

## License

TTFEMesh is licensed under the MIT License - see the [LICENSE file](https://github.com/MazenAli/TT-FEMesh/blob/main/LICENSE) for details. 