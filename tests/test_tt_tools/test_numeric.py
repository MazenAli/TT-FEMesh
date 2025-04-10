import pytest
import torch
import numpy as np
from ttfemesh.tt_tools.numeric import integer_to_little_endian, unit_vector_binary_tt
from ttfemesh.types import TensorTrain


class TestIntegerToLittleEndian:
    def test_basic_conversion(self):
        assert integer_to_little_endian(3, 0) == [0, 0, 0]
        assert integer_to_little_endian(3, 1) == [1, 0, 0]
        assert integer_to_little_endian(3, 2) == [0, 1, 0]
        assert integer_to_little_endian(3, 3) == [1, 1, 0]
        assert integer_to_little_endian(3, 4) == [0, 0, 1]
        assert integer_to_little_endian(3, 5) == [1, 0, 1]
        assert integer_to_little_endian(3, 6) == [0, 1, 1]
        assert integer_to_little_endian(3, 7) == [1, 1, 1]

    def test_edge_cases(self):
        assert integer_to_little_endian(4, 15) == [1, 1, 1, 1]
        assert integer_to_little_endian(5, 31) == [1, 1, 1, 1, 1]

    def test_negative_input(self):
        with pytest.raises(ValueError):
            integer_to_little_endian(3, -1)

    def test_too_large_input(self):
        with pytest.raises(ValueError):
            integer_to_little_endian(3, 8)
        with pytest.raises(ValueError):
            integer_to_little_endian(4, 16)


class TestUnitVectorBinaryTT:
    def test_basic_vectors(self):
        length = 3
        for i in range(2**length):
            tt = unit_vector_binary_tt(length, i)
            assert isinstance(tt, TensorTrain)
            assert len(tt.cores) == length
            
            full_tensor = np.array(tt.full()).flatten("F")
            expected = np.zeros(2**length)
            expected[i] = 1.0
            assert np.allclose(full_tensor, expected)

    def test_core_structure(self):
        length = 3
        index = 5
        tt = unit_vector_binary_tt(length, index)
        
        for j, core in enumerate(tt.cores):
            assert core.shape == (1, 2, 1)
            expected_bit = (index >> j) & 1
            assert core[0, expected_bit, 0] == 1.0
            assert core[0, 1 - expected_bit, 0] == 0.0

    def test_edge_cases(self):
        length = 4
        tt_first = unit_vector_binary_tt(length, 0)
        tt_last = unit_vector_binary_tt(length, 15)
        
        assert np.allclose(tt_first.full().flatten(), [1.0] + [0.0] * 15)
        assert np.allclose(tt_last.full().flatten(), [0.0] * 15 + [1.0])

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            unit_vector_binary_tt(3, -1)
        with pytest.raises(ValueError):
            unit_vector_binary_tt(3, 8)
        with pytest.raises(ValueError):
            unit_vector_binary_tt(4, 16) 
