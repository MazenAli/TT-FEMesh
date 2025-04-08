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
        bindex = np.array([0, 0, 1, 0, 0, 1])
        expected = (2, 4)
        assert bindex2dtuple(bindex) == expected

        bindex = np.array([1, 1, 0, 1, 1, 0])
        expected = (5, 3)
        assert bindex2dtuple(bindex) == expected

    def test_invalid_shape(self):
        bindex = np.array([[0, 1], [1, 0]])
        with pytest.raises(ValueError, match="Invalid shape"):
            bindex2dtuple(bindex)

    def test_odd_number_of_elements(self):
        bindex = np.array([0, 1, 0])
        with pytest.raises(ValueError, match="Binary index must have even number of elements"):
            bindex2dtuple(bindex)

    def test_invalid_values(self):
        bindex = np.array([0, 1, 2, 0])
        with pytest.raises(ValueError, match="Binary index must contain only 0s and 1s"):
            bindex2dtuple(bindex)


class TestQindex2dtuple:
    def test_valid_quaternary_index(self):
        qindex = np.array([0, 1, 2, 3])
        expected = (10, 12)
        assert qindex2dtuple(qindex) == expected

        qindex = np.array([3, 2, 1, 0])
        expected = (5, 3)
        assert qindex2dtuple(qindex) == expected

    def test_invalid_shape(self):
        qindex = np.array([[0, 1], [2, 3]])
        with pytest.raises(ValueError, match="Invalid shape"):
            qindex2dtuple(qindex)

    def test_invalid_values(self):
        qindex = np.array([0, 1, 4, 3])
        with pytest.raises(ValueError, match="Index must contain only values in"):
            qindex2dtuple(qindex)


class TestSideConcatenationCore:
    def test_bottom_side(self):
        core = side_concatenation_core(BoundarySide2D.BOTTOM)
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]).reshape([1, 2, 4, 1])
        np.testing.assert_array_equal(core, expected)

    def test_right_side(self):
        core = side_concatenation_core(BoundarySide2D.RIGHT)
        expected = np.array([[0, 1, 0, 0], [0, 0, 0, 1]]).reshape([1, 2, 4, 1])
        np.testing.assert_array_equal(core, expected)

    def test_top_side(self):
        core = side_concatenation_core(BoundarySide2D.TOP)
        expected = np.array([[0, 0, 0, 1], [0, 0, 1, 0]]).reshape([1, 2, 4, 1])
        np.testing.assert_array_equal(core, expected)

    def test_left_side(self):
        core = side_concatenation_core(BoundarySide2D.LEFT)
        expected = np.array([[0, 0, 1, 0], [1, 0, 0, 0]]).reshape([1, 2, 4, 1])
        np.testing.assert_array_equal(core, expected)


class TestVertexConcatenationCore:
    def test_bottom_left_vertex(self):
        core = vertex_concatenation_core(BoundaryVertex2D.BOTTOM_LEFT)
        expected = np.array([1, 0, 0, 0]).reshape([1, 1, 4, 1])
        np.testing.assert_array_equal(core, expected)

    def test_bottom_right_vertex(self):
        core = vertex_concatenation_core(BoundaryVertex2D.BOTTOM_RIGHT)
        expected = np.array([0, 1, 0, 0]).reshape([1, 1, 4, 1])
        np.testing.assert_array_equal(core, expected)

    def test_top_right_vertex(self):
        core = vertex_concatenation_core(BoundaryVertex2D.TOP_RIGHT)
        expected = np.array([0, 0, 0, 1]).reshape([1, 1, 4, 1])
        np.testing.assert_array_equal(core, expected)

    def test_top_left_vertex(self):
        core = vertex_concatenation_core(BoundaryVertex2D.TOP_LEFT)
        expected = np.array([0, 0, 1, 0]).reshape([1, 1, 4, 1])
        np.testing.assert_array_equal(core, expected)


class TestConcatCore2tt:
    def test_without_exchange(self):
        core = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]).reshape([1, 2, 4, 1])
        length = 3
        
        tt = concat_core2tt(core, length, exchanged=False)
        
        assert isinstance(tt, TensorTrain)
        assert len(tt.cores) == length
        
        for i in range(length):
            assert torch.equal(tt.cores[i], torch.tensor(core))

    def test_with_exchange(self):
        core = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]).reshape([1, 2, 4, 1])
        length = 3
        
        tt = concat_core2tt(core, length, exchanged=True)
        
        assert isinstance(tt, TensorTrain)
        assert len(tt.cores) == length
        
        expected_core = np.array([[0, 1, 0, 0], [1, 0, 0, 0]]).reshape([1, 2, 4, 1])
        for i in range(length):
            assert torch.equal(tt.cores[i], torch.tensor(expected_core))


class TestConcatTtmaps:
    def test_concatenation_maps(self):
        mock_tt_left = MagicMock(spec=TensorTrain)
        mock_tt_right = MagicMock(spec=TensorTrain)
        
        mock_tt_left.t.return_value = mock_tt_left
        mock_tt_right.t.return_value = mock_tt_right
        
        mock_connect_tt = MagicMock(spec=TensorTrain)
        mock_tt_left.__matmul__.return_value = mock_connect_tt
        
        mock_count_left = MagicMock(spec=TensorTrain)
        mock_tt_left.__matmul__.return_value.__matmul__.return_value = mock_count_left
        mock_count_left.__neg__.return_value = mock_count_left
        
        mock_count_right = MagicMock(spec=TensorTrain)
        mock_tt_right.__matmul__.return_value.__matmul__.return_value = mock_count_right
        mock_count_right.__neg__.return_value = mock_count_right
        
        concat_ttmaps(mock_tt_left, mock_tt_right)
        
        mock_tt_left.t.assert_called()
        assert mock_tt_left.t.call_count == 2
        mock_tt_left.__matmul__.assert_called_with(mock_tt_left)
        mock_tt_right.t.assert_called_once()


class TestSideConcatenationTt:
    def test_side_concatenation_tt(self):
        mock_core0 = MagicMock()
        mock_core1 = MagicMock()
        mock_tt_left = MagicMock(spec=TensorTrain)
        mock_tt_right = MagicMock(spec=TensorTrain)
        mock_connect_tt = MagicMock(spec=TensorTrain)
        mock_count_left = MagicMock(spec=TensorTrain)
        mock_count_right = MagicMock(spec=TensorTrain)
        
        with patch('ttfemesh.mesh.mesh_utils.side_concatenation_core') as mock_side_core:
            mock_side_core.side_effect = [mock_core0, mock_core1]
            
            with patch('ttfemesh.mesh.mesh_utils.concat_core2tt') as mock_concat_core:
                mock_concat_core.side_effect = [mock_tt_left, mock_tt_right]
                
                with patch('ttfemesh.mesh.mesh_utils.concat_ttmaps') as mock_concat_ttmaps:
                    mock_concat_ttmaps.return_value = (mock_connect_tt, mock_count_left, mock_count_right)
                    
                    result = side_concatenation_tt(BoundarySide2D.BOTTOM, BoundarySide2D.TOP, 3)
                    
                    assert result == (mock_connect_tt, mock_count_left, mock_count_right)
                    
                    assert mock_side_core.call_count == 2
                    assert mock_concat_core.call_count == 2
                    mock_concat_ttmaps.assert_called_once_with(mock_tt_left, mock_tt_right)


class TestVertexConcatenationTt:
    def test_vertex_concatenation_tt(self):
        mock_core0 = MagicMock()
        mock_core1 = MagicMock()
        mock_tt_left = MagicMock(spec=TensorTrain)
        mock_tt_right = MagicMock(spec=TensorTrain)
        mock_connect_tt = MagicMock(spec=TensorTrain)
        mock_count_left = MagicMock(spec=TensorTrain)
        mock_count_right = MagicMock(spec=TensorTrain)
        
        with patch('ttfemesh.mesh.mesh_utils.vertex_concatenation_core') as mock_vertex_core:
            mock_vertex_core.side_effect = [mock_core0, mock_core1]
            
            with patch('ttfemesh.mesh.mesh_utils.concat_core2tt') as mock_concat_core:
                mock_concat_core.side_effect = [mock_tt_left, mock_tt_right]
                
                with patch('ttfemesh.mesh.mesh_utils.concat_ttmaps') as mock_concat_ttmaps:
                    mock_concat_ttmaps.return_value = (mock_connect_tt, mock_count_left, mock_count_right)
                    
                    result = vertex_concatenation_tt(BoundaryVertex2D.BOTTOM_LEFT, BoundaryVertex2D.TOP_RIGHT, 3)
                    
                    assert result == (mock_connect_tt, mock_count_left, mock_count_right)
                    
                    assert mock_vertex_core.call_count == 2
                    assert mock_concat_core.call_count == 2
                    mock_concat_ttmaps.assert_called_once_with(mock_tt_left, mock_tt_right) 