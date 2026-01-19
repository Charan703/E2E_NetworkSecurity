import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open
from networksecurity.utils.main_utils.utils import load_object, save_object


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_save_and_load_object(self):
        """Test saving and loading objects"""
        # Create test object
        test_data = {"key": "value", "number": 42}
        test_file = "test_object.pkl"
        
        try:
            # Test save
            save_object(test_file, test_data)
            
            # Test load
            loaded_data = load_object(test_file)
            
            assert loaded_data == test_data, "Loaded object should match saved object"
            
        finally:
            # Cleanup
            import os
            if os.path.exists(test_file):
                os.remove(test_file)
    
    def test_load_nonexistent_file_raises_error(self):
        """Test that loading non-existent file raises appropriate error"""
        with pytest.raises(Exception):
            load_object("nonexistent_file.pkl")
    
    def test_dataframe_operations(self):
        """Test basic dataframe operations used in pipeline"""
        # Create test dataframe
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [0, 1, 0, 1, 0]
        })
        
        # Test basic operations
        assert len(df) == 5, "DataFrame should have 5 rows"
        assert len(df.columns) == 3, "DataFrame should have 3 columns"
        assert 'target' in df.columns, "DataFrame should have target column"
        
        # Test data types
        assert df['feature1'].dtype in [np.int64, np.int32], "Feature1 should be integer"
        assert df['feature2'].dtype in [np.float64, np.float32], "Feature2 should be float"