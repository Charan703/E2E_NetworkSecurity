import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from networksecurity.utils.ml_utils.model.estimator import NetwrokModel


class TestModelFunctionality:
    """Test model prediction functionality"""
    
    def test_model_output_format(self):
        """Test that model produces output in correct format"""
        # Create mock model and preprocessor
        mock_model = Mock()
        mock_preprocessor = Mock()
        
        # Mock model prediction
        mock_model.predict.return_value = np.array([0, 1, 0, 1])
        mock_preprocessor.transform.return_value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        
        # Create test data
        test_data = pd.DataFrame({
            'URL_Length': [50, 100, 75, 120],
            'having_At_Symbol': [0, 1, 0, 1],
            'SSLfinal_State': [1, 0, 1, 0]
        })
        
        # Test model
        network_model = NetwrokModel(model=mock_model, preprocessing_object=mock_preprocessor)
        predictions = network_model.predict(test_data)
        
        # Assertions
        assert isinstance(predictions, np.ndarray), "Predictions should be numpy array"
        assert len(predictions) == len(test_data), "Predictions length should match input"
        assert all(pred in [0, 1] for pred in predictions), "Predictions should be binary (0 or 1)"
    
    def test_model_handles_empty_data(self):
        """Test model handles empty input gracefully"""
        mock_model = Mock()
        mock_preprocessor = Mock()
        
        # Mock empty prediction
        mock_model.predict.return_value = np.array([])
        mock_preprocessor.transform.return_value = np.array([]).reshape(0, -1)
        
        empty_data = pd.DataFrame()
        
        network_model = NetwrokModel(model=mock_model, preprocessing_object=mock_preprocessor)
        
        # Should handle empty data without crashing
        try:
            predictions = network_model.predict(empty_data)
            assert len(predictions) == 0, "Empty input should return empty predictions"
        except Exception as e:
            # If it raises an exception, it should be a meaningful one
            assert "empty" in str(e).lower() or "no data" in str(e).lower()
    
    def test_prediction_values_are_valid(self):
        """Test that predictions are valid binary values"""
        mock_model = Mock()
        mock_preprocessor = Mock()
        
        # Mock valid predictions
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0])
        mock_preprocessor.transform.return_value = np.random.rand(5, 3)
        
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'feature3': [10, 20, 30, 40, 50]
        })
        
        network_model = NetwrokModel(model=mock_model, preprocessing_object=mock_preprocessor)
        predictions = network_model.predict(test_data)
        
        # Check all predictions are valid
        assert all(isinstance(pred, (int, np.integer)) for pred in predictions), "Predictions should be integers"
        assert all(pred >= 0 for pred in predictions), "Predictions should be non-negative"
        assert all(pred <= 1 for pred in predictions), "Predictions should be <= 1"