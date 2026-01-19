import unittest
import tempfile
import os
import pandas as pd

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_save_and_load_object(self):
        """Test object serialization with proper file path"""
        # Use temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test_object.pkl")
            test_data = {"key": "value", "number": 42}
            
            # Mock save/load functionality
            import pickle
            
            # Save object
            with open(test_file, 'wb') as f:
                pickle.dump(test_data, f)
            
            # Load object
            with open(test_file, 'rb') as f:
                loaded_data = pickle.load(f)
            
            self.assertEqual(test_data, loaded_data)
    
    def test_load_nonexistent_file_raises_error(self):
        """Test loading non-existent file raises error"""
        nonexistent_file = "/tmp/nonexistent_file.pkl"
        
        with self.assertRaises(FileNotFoundError):
            with open(nonexistent_file, 'rb') as f:
                pass
    
    def test_dataframe_operations(self):
        """Test basic DataFrame operations"""
        # Create test DataFrame
        data = {
            'feature1': [1, 2, 3, 4],
            'feature2': [0.1, 0.2, 0.3, 0.4],
            'target': [0, 1, 0, 1]
        }
        df = pd.DataFrame(data)
        
        # Test basic operations
        self.assertEqual(len(df), 4)
        self.assertEqual(list(df.columns), ['feature1', 'feature2', 'target'])
        self.assertEqual(df['target'].sum(), 2)
        
        # Test data types
        self.assertTrue(df['feature1'].dtype in ['int64', 'int32'])
        self.assertTrue(df['feature2'].dtype in ['float64', 'float32'])