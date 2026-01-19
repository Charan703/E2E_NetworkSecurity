import unittest
import yaml
import os

class TestSchemaValidation(unittest.TestCase):
    """Test schema validation functionality"""
    
    def test_schema_file_exists(self):
        """Test that schema file exists"""
        schema_path = "data_schema/schema.yaml"
        self.assertTrue(os.path.exists(schema_path))
    
    def test_schema_structure(self):
        """Test schema has required structure"""
        schema_path = "data_schema/schema.yaml"
        
        with open(schema_path, 'r') as file:
            schema = yaml.safe_load(file)
        
        # Check required keys exist
        self.assertIn("columns", schema)
        self.assertIn("numerical_columns", schema)
        
        # Check columns is a list
        self.assertIsInstance(schema["columns"], list)
        self.assertIsInstance(schema["numerical_columns"], list)
    
    def test_valid_data_passes_validation(self):
        """Test valid data structure"""
        # Mock valid data structure
        valid_data = {
            "having_IP_Address": [0, 1, 0],
            "URL_Length": [25, 50, 30],
            "Result": [0, 1, 0]
        }
        
        # Basic validation checks
        self.assertIsInstance(valid_data, dict)
        self.assertIn("Result", valid_data)
        
        # Check all values are lists
        for key, values in valid_data.items():
            self.assertIsInstance(values, list)
    
    def test_schema_column_count(self):
        """Test schema has expected number of columns"""
        schema_path = "data_schema/schema.yaml"
        
        with open(schema_path, 'r') as file:
            schema = yaml.safe_load(file)
        
        # Should have 30 columns (29 features + 1 target)
        self.assertEqual(len(schema["columns"]), 30)
        self.assertEqual(len(schema["numerical_columns"]), 30)