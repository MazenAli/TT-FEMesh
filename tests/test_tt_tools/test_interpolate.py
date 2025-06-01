import numpy as np
from torchtt import TT

from ttfemesh.mesh.mesh_utils import qindex2dtuple
from ttfemesh.tt_tools.interpolate import interpolate_linear2d
from ttfemesh.tt_tools.meshgrid import map2canonical2d


class TestInterpolateLinear2D:
    def test_constant_function(self):
        """Test interpolation of a constant function."""

        def constant_func(index):
            return 5.0

        d = 3
        result = interpolate_linear2d(constant_func, d)

        assert isinstance(result, TT)
        assert np.allclose(result.full(), 5.0)

    def test_linear_function_x(self):
        def linear_x_func(index):
            index2d = qindex2dtuple(index)
            return index2d[0]

        d = 3
        result = interpolate_linear2d(linear_x_func, d)

        assert isinstance(result, TT)
        zmap = map2canonical2d(d)
        full_tensor = np.array(result.full()).flatten("F")
        reordered = np.empty_like(full_tensor)
        reordered[zmap] = full_tensor
        final = reordered.reshape((2**d, 2**d), order="F")

        expected_column = np.arange(2**d)
        for j in range(2**d):
            assert np.allclose(final[:, j], expected_column)

    def test_linear_function_y(self):
        def linear_y_func(index):
            index2d = qindex2dtuple(index)
            return index2d[1]

        d = 3
        result = interpolate_linear2d(linear_y_func, d)

        assert isinstance(result, TT)
        zmap = map2canonical2d(d)
        full_tensor = np.array(result.full()).flatten("F")
        reordered = np.empty_like(full_tensor)
        reordered[zmap] = full_tensor
        final = reordered.reshape((2**d, 2**d), order="F")

        expected_row = np.arange(2**d)
        for i in range(2**d):
            assert np.allclose(final[i, :], expected_row)

    def test_linear_function_xy(self):
        def linear_xy_func(index):
            index2d = qindex2dtuple(index)
            return index2d[0] + index2d[1]

        d = 3
        result = interpolate_linear2d(linear_xy_func, d)

        assert isinstance(result, TT)
        zmap = map2canonical2d(d)
        full_tensor = np.array(result.full()).flatten("F")
        reordered = np.empty_like(full_tensor)
        reordered[zmap] = full_tensor
        final = reordered.reshape((2**d, 2**d), order="F")

        expected = np.empty((2**d, 2**d))
        for i in range(2**d):
            for j in range(2**d):
                expected[i, j] = i + j

        assert np.allclose(final, expected)
