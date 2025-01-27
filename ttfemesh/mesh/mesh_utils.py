from typing import Tuple

import numpy as np
import torch

from ttfemesh.types import BoundarySide2D, BoundaryVertex2D, TensorTrain


def bindex2dtuple(bindex: np.ndarray) -> Tuple[int, int]:
    """
    Convert a binary index to a 2D tuple.
    Implements the mapping (i0, j0, i1, j1, ...) -> (i, j), where i = i0 + 2*i1 + 4*i2 + ...
    and j = j0 + 2*j1 + 4*j2 + ...
    This is the common little-endian convention in QTT literature.

    Args:
        bindex (np.ndarray): A binary index of shape (2 * num_bits1d,).

    Returns:
        Tuple[int, int]: A 2D tuple.

    Raises:
        ValueError: If bindex is not a 1D array, does not contain an even number of elements, or
            contains values other than 0 or 1.
    """

    shape = bindex.shape
    if len(shape) != 1:
        raise ValueError(f"Invalid shape ({shape}) for binary index. Expected 1D array.")

    if shape[0] % 2 != 0:
        raise ValueError("Binary index must have even number of elements.")

    if not np.all((bindex == 0) | (bindex == 1)):
        raise ValueError("Binary index must contain only 0s and 1s.")

    i_bits = bindex[::2]
    j_bits = bindex[1::2]

    i = np.sum(i_bits * (2 ** np.arange(len(i_bits))))
    j = np.sum(j_bits * (2 ** np.arange(len(j_bits))))

    return (i, j)


def qindex2dtuple(index: np.ndarray) -> Tuple[int, int]:
    """
    Convert a quaternary index to a 2D tuple.
    The ordering of the quaternary index is assumed to be
    (0, 1, 2, 3) -> ((0, 0), (1, 0), (0, 1), (1, 1)), i.e., column-major with
    index = i + 2*j.
    This is consistent with the little-endian ordering of the binary index
    commonly used in QTT literature.

    Args:
        index (np.ndarray): An index of shape (num_quats,) with values in {0, 1, 2, 3}.

    Returns:
        Tuple[int, int]: A 2D tuple.

    Raises:
        ValueError: If index is not a 1D array or contains values other than {0, 1, 2, 3}.
    """

    shape = index.shape
    if len(shape) != 1:
        raise ValueError(f"Invalid shape ({shape}) for index. Expected 1D array.")

    if not np.all((0 <= index) & (index <= 3)):
        raise ValueError("Index must contain only values in {0, 1, 2, 3}.")

    j = index % 2
    i = index // 2

    binary_index = np.empty(2 * len(index), dtype=int)
    binary_index[::2] = i
    binary_index[1::2] = j

    return bindex2dtuple(binary_index)


def side_concatenation_core(side: BoundarySide2D) -> np.ndarray:
    """
    Get the TT-core for concatenation of a boundary side.
    See Section 5.2 of arXiv:1802.02839 for details.

    Args:
        side (BoundarySide2D): The boundary side.

    Returns:
        np.ndarray: The TT-core for concatenation of the boundary side.
    """

    side_values = {
        BoundarySide2D.BOTTOM: 0,
        BoundarySide2D.RIGHT: 0,
        BoundarySide2D.TOP: 0,
        BoundarySide2D.LEFT: 0,
    }
    side_values[side] = 1
    B, R, T, L = (
        side_values[BoundarySide2D.BOTTOM],
        side_values[BoundarySide2D.RIGHT],
        side_values[BoundarySide2D.TOP],
        side_values[BoundarySide2D.LEFT],
    )

    core = np.array([[B, R, L, T], [L, B, T, R]]).reshape([1, 2, 4, 1])

    return core


def vertex_concatenation_core(vertex: BoundaryVertex2D) -> np.ndarray:
    """
    Get the TT-core for concatenation of a vertex.
    See Section 5.2 of arXiv:1802.02839 for details.

    Returns:
        np.ndarray: The TT-core for concatenation of a vertex.
    """

    vertex_values = {
        BoundaryVertex2D.BOTTOM_LEFT: 0,
        BoundaryVertex2D.BOTTOM_RIGHT: 0,
        BoundaryVertex2D.TOP_RIGHT: 0,
        BoundaryVertex2D.TOP_LEFT: 0,
    }
    vertex_values[vertex] = 1
    BL, BR, TR, TL = (
        vertex_values[BoundaryVertex2D.BOTTOM_LEFT],
        vertex_values[BoundaryVertex2D.BOTTOM_RIGHT],
        vertex_values[BoundaryVertex2D.TOP_RIGHT],
        vertex_values[BoundaryVertex2D.TOP_LEFT],
    )

    core = np.array([BL, BR, TL, TR]).reshape([1, 1, 4, 1])

    return core


def concat_core2tt(core: np.ndarray, length: int, exchanged: bool = False) -> TensorTrain:
    """
    Convert a TT-core to a TensorTrain.
    The rank-1 TT-core is assumed to be in the format (1, m, n, 1).

    Args:
        core (np.ndarray): The TT-core.
        length (int): The length of the TT.
        exchanged (bool): Whether the rows of the TT-core are exchanged.

    Returns:
        TensorTrain: The TT-representation of the core.
    """

    core_ = core.copy()
    if exchanged and core_.size > 4:
        core_ = core[:, [1, 0], :, :]
    cores = [torch.tensor(core_)] * length
    return TensorTrain(cores)


def concat_ttmaps(
    tt_left: TensorTrain, tt_right: TensorTrain
) -> Tuple[TensorTrain, TensorTrain, TensorTrain]:
    """
    Put together the two TT-map primitives to form the full connectivity maps.
    See Section 5 of arXiv:1802.02839 for details.
    The connectivity maps are defined as follows:
    Pmp describes which nodes in the left domain are connected to which nodes in the right domain,
    Pmm describes which nodes in the left domain are to be connected,
    Ppp describes which nodes in the right domain are to be connected.

    Args:
        tt_left (TensorTrain): The left TT-map.
        tt_right (TensorTrain): The right TT-map.

    Returns:
        Tuple[TensorTrain, TensorTrain, TensorTrain]: The connectivity maps.
    """

    connect_tt = tt_left.t() @ tt_right
    count_left = -(tt_left.t() @ tt_left)
    count_right = -(tt_right.t() @ tt_right)

    return connect_tt, count_left, count_right


def side_concatenation_tt(
    side0: BoundarySide2D, side1: BoundarySide2D, length: int
) -> Tuple[TensorTrain, TensorTrain, TensorTrain]:
    """
    Get the TT-representation of the concatenation tensors of two boundary sides.
    See Section 5 of arXiv:1802.02839 for details.
    The connectivity maps are defined as follows:
    Pmp describes which nodes in the left domain are connected to which nodes in the right domain,
    Pmm describes which nodes in the left domain are to be connected,
    Ppp describes which nodes in the right domain are to be connected.

    Args:
        side0 (BoundarySide2D): The first boundary side.
        side1 (BoundarySide2D): The second boundary side.
        length (int): The length of the TT.

    Returns:
        Tuple[TensorTrain, TensorTrain, TensorTrain]: The connectivity maps.
    """

    core0 = side_concatenation_core(side0)
    core1 = side_concatenation_core(side1)

    tt_left = concat_core2tt(core0, length, exchanged=False)
    tt_right = concat_core2tt(core1, length, exchanged=True)

    return concat_ttmaps(tt_left, tt_right)


def vertex_concatenation_tt(
    vertex0: BoundaryVertex2D, vertex1: BoundaryVertex2D, length: int
) -> Tuple[TensorTrain, TensorTrain, TensorTrain]:
    """
    Get the TT-representation of the concatenation tensor of two boundary vertices.
    See Section 5 of arXiv:1802.02839 for details.
    The connectivity maps are defined as follows:
    Pmp describes which nodes in the left domain are connected to which nodes in the right domain,
    Pmm describes which nodes in the left domain are to be connected,
    Ppp describes which nodes in the right domain are to be connected.

    Args:
        vertex0 (BoundaryVertex2D): The first boundary vertex.
        vertex1 (BoundaryVertex2D): The second boundary vertex.
        length (int): The length of the TT.

    Returns:
        Tuple[TensorTrain, TensorTrain, TensorTrain]: The connectivity maps.
    """

    core0 = vertex_concatenation_core(vertex0)
    core1 = vertex_concatenation_core(vertex1)

    tt_left = concat_core2tt(core0, length, exchanged=False)
    tt_right = concat_core2tt(core1, length, exchanged=False)

    return concat_ttmaps(tt_left, tt_right)
