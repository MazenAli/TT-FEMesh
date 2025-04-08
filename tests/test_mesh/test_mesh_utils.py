import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

from ttfemesh.mesh.mesh_utils import (
    bindex2dtuple,
    qindex2dtuple,
    side_concatenation_core,
    vertex_concatenation_core,
    concat_core2tt,
    concat_ttmaps,
    side_concatenation_tt,
    vertex_concatenation_tt
)
from ttfemesh.types import BoundarySide2D, BoundaryVertex2D, TensorTrain


class TestBindex2dtuple:
    def test_valid_binary_index(self):
        # Test with a simple binary index
        bindex = np.array([0, 0, 1, 0, 0, 1])
        expected = (1, 2)  # i = 0 + 2*1 + 4*0 = 1, j = 0 + 2*0 + 4*1 = 2
        assert bindex2dtuple(bindex) == expected

        # Test with a more complex binary index
        bindex = np.array([1, 1, 0, 1, 1, 0])
        expected = (11, 5)  # i = 1 + 2*0 + 4*1 = 11, j = 1 + 2*1 + 4*0 = 5
        assert bindex2dtuple(bindex) == expected

    def test_invalid_shape(self):
        # Test with a 2D array
        bindex = np.array([[0, 1], [1, 0]])
        with pytest.raises(ValueError, match="Invalid shape"):
            bindex2dtuple(bindex)

    def test_odd_number_of_elements(self):
        # Test with an odd number of elements
        bindex = np.array([0, 1, 0])
        with pytest.raises(ValueError, match="Binary index must have even number of elements"):
            bindex2dtuple(bindex)

    def test_invalid_values(self):
        # Test with values other than 0 or 1
        bindex = np.array([0, 1, 2, 0])
        with pytest.raises(ValueError, match="Binary index must contain only 0s and 1s"):
            bindex2dtuple(bindex)


class TestQindex2dtuple:
    def test_valid_quaternary_index(self):
        # Test with a simple quaternary index
        qindex = np.array([0, 1, 2, 3])
        expected = (2, 1)  # i = 0 + 2*1 = 2, j = 0 + 2*1 = 1
        assert qindex2dtuple(qindex) == expected

        # Test with a more complex quaternary index
        qindex = np.array([3, 2, 1, 0])
        expected = (5, 3)  # i = 1 + 2*2 = 5, j = 1 + 2*1 = 3
        assert qindex2dtuple(qindex) == expected

    def test_invalid_shape(self):
        # Test with a 2D array
        qindex = np.array([[0, 1], [2, 3]])
        with pytest.raises(ValueError, match="Invalid shape"):
            qindex2dtuple(qindex)

    def test_invalid_values(self):
        # Test with values outside {0, 1, 2, 3}
        qindex = np.array([0, 1, 4, 3])
        with pytest.raises(ValueError, match="Index must contain only values in"):
            qindex2dtuple(qindex)


class TestSideConcatenationCore:
    def test_bottom_side(self):
        core = side_concatenation_core(BoundarySide2D.BOTTOM)
        expected = np.array([[[[1, 0, 0, 0], [0, 1, 0, 0]]]])
        np.testing.assert_array_equal(core, expected)

    def test_right_side(self):
        core = side_concatenation_core(BoundarySide2D.RIGHT)
        expected = np.array([[[[0, 1, 0, 0], [0, 0, 0, 1]]]])
        np.testing.assert_array_equal(core, expected)

    def test_top_side(self):
        core = side_concatenation_core(BoundarySide2D.TOP)
        expected = np.array([[[[0, 0, 0, 1], [0, 0, 1, 0]]]])
        np.testing.assert_array_equal(core, expected)

    def test_left_side(self):
        core = side_concatenation_core(BoundarySide2D.LEFT)
        expected = np.array([[[[0, 0, 1, 0], [1, 0, 0, 0]]]])
        np.testing.assert_array_equal(core, expected)


class TestVertexConcatenationCore:
    def test_bottom_left_vertex(self):
        core = vertex_concatenation_core(BoundaryVertex2D.BOTTOM_LEFT)
        expected = np.array([[[[1, 0, 0, 0]]]])
        np.testing.assert_array_equal(core, expected)

    def test_bottom_right_vertex(self):
        core = vertex_concatenation_core(BoundaryVertex2D.BOTTOM_RIGHT)
        expected = np.array([[[[0, 1, 0, 0]]]])
        np.testing.assert_array_equal(core, expected)

    def test_top_right_vertex(self):
        core = vertex_concatenation_core(BoundaryVertex2D.TOP_RIGHT)
        expected = np.array([[[[0, 0, 0, 1]]]])
        np.testing.assert_array_equal(core, expected)

    def test_top_left_vertex(self):
        core = vertex_concatenation_core(BoundaryVertex2D.TOP_LEFT)
        expected = np.array([[[[0, 0, 1, 0]]]])
        np.testing.assert_array_equal(core, expected)


class TestConcatCore2tt:
    def test_without_exchange(self):
        core = np.array([[[[1, 0, 0, 0], [0, 1, 0, 0]]]])
        length = 3
        
        tt = concat_core2tt(core, length, exchanged=False)
        
        assert isinstance(tt, TensorTrain)
        assert len(tt.cores) == length
        
        for i in range(length):
            assert torch.equal(tt.cores[i], torch.tensor(core))

    def test_with_exchange(self):
        core = np.array([[[[1, 0, 0, 0], [0, 1, 0, 0]]]])
        length = 3
        
        tt = concat_core2tt(core, length, exchanged=True)
        
        assert isinstance(tt, TensorTrain)
        assert len(tt.cores) == length
        
        expected_core = np.array([[[[0, 1, 0, 0], [1, 0, 0, 0]]]])
        for i in range(length):
            assert torch.equal(tt.cores[i], torch.tensor(expected_core))


class TestConcatTtmaps:
    def test_concatenation_maps(self):
        # Create mock TensorTrain objects
        mock_tt_left = MagicMock(spec=TensorTrain)
        mock_tt_right = MagicMock(spec=TensorTrain)
        
        # Set up the mock transpose and matrix multiplication
        mock_tt_left.t.return_value = mock_tt_left
        mock_tt_right.t.return_value = mock_tt_right
        
        mock_connect_tt = MagicMock(spec=TensorTrain)
        mock_count_left = MagicMock(spec=TensorTrain)
        mock_count_right = MagicMock(spec=TensorTrain)
        
        mock_tt_left.__matmul__.return_value = mock_connect_tt
        mock_tt_left.__matmul__.return_value.__matmul__.return_value = mock_count_left
        mock_tt_right.__matmul__.return_value.__matmul__.return_value = mock_count_right
        
        # Call the function
        connect_tt, count_left, count_right = concat_ttmaps(mock_tt_left, mock_tt_right)
        
        # Verify the results
        assert connect_tt == mock_connect_tt
        assert count_left == mock_count_left
        assert count_right == mock_count_right
        
        # Verify the method calls
        mock_tt_left.t.assert_called_once()
        mock_tt_right.t.assert_called_once()
        assert mock_tt_left.__matmul__.call_count == 2
        assert mock_tt_right.__matmul__.call_count == 1


class TestSideConcatenationTt:
    def test_side_concatenation_tt(self):
        # Create mock objects
        mock_core0 = MagicMock()
        mock_core1 = MagicMock()
        mock_tt_left = MagicMock(spec=TensorTrain)
        mock_tt_right = MagicMock(spec=TensorTrain)
        mock_connect_tt = MagicMock(spec=TensorTrain)
        mock_count_left = MagicMock(spec=TensorTrain)
        mock_count_right = MagicMock(spec=TensorTrain)
        
        # Set up the mocks
        with patch('ttfemesh.mesh.mesh_utils.side_concatenation_core') as mock_side_core:
            mock_side_core.side_effect = [mock_core0, mock_core1]
            
            with patch('ttfemesh.mesh.mesh_utils.concat_core2tt') as mock_concat_core:
                mock_concat_core.side_effect = [mock_tt_left, mock_tt_right]
                
                with patch('ttfemesh.mesh.mesh_utils.concat_ttmaps') as mock_concat_ttmaps:
                    mock_concat_ttmaps.return_value = (mock_connect_tt, mock_count_left, mock_count_right)
                    
                    # Call the function
                    result = side_concatenation_tt(BoundarySide2D.BOTTOM, BoundarySide2D.TOP, 3)
                    
                    # Verify the results
                    assert result == (mock_connect_tt, mock_count_left, mock_count_right)
                    
                    # Verify the method calls
                    assert mock_side_core.call_count == 2
                    assert mock_concat_core.call_count == 2
                    mock_concat_ttmaps.assert_called_once_with(mock_tt_left, mock_tt_right)


class TestVertexConcatenationTt:
    def test_vertex_concatenation_tt(self):
        # Create mock objects
        mock_core0 = MagicMock()
        mock_core1 = MagicMock()
        mock_tt_left = MagicMock(spec=TensorTrain)
        mock_tt_right = MagicMock(spec=TensorTrain)
        mock_connect_tt = MagicMock(spec=TensorTrain)
        mock_count_left = MagicMock(spec=TensorTrain)
        mock_count_right = MagicMock(spec=TensorTrain)
        
        # Set up the mocks
        with patch('ttfemesh.mesh.mesh_utils.vertex_concatenation_core') as mock_vertex_core:
            mock_vertex_core.side_effect = [mock_core0, mock_core1]
            
            with patch('ttfemesh.mesh.mesh_utils.concat_core2tt') as mock_concat_core:
                mock_concat_core.side_effect = [mock_tt_left, mock_tt_right]
                
                with patch('ttfemesh.mesh.mesh_utils.concat_ttmaps') as mock_concat_ttmaps:
                    mock_concat_ttmaps.return_value = (mock_connect_tt, mock_count_left, mock_count_right)
                    
                    # Call the function
                    result = vertex_concatenation_tt(BoundaryVertex2D.BOTTOM_LEFT, BoundaryVertex2D.TOP_RIGHT, 3)
                    
                    # Verify the results
                    assert result == (mock_connect_tt, mock_count_left, mock_count_right)
                    
                    # Verify the method calls
                    assert mock_vertex_core.call_count == 2
                    assert mock_concat_core.call_count == 2
                    mock_concat_ttmaps.assert_called_once_with(mock_tt_left, mock_tt_right) 