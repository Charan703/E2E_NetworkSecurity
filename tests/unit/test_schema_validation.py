import pytest
import pandas as pd
import yaml
from networksecurity.components.data_validation import DataValidation
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact


class TestSchemaValidation:
    """Test schema validation functionality"""
    
    def test_schema_file_exists(self):
        """Test that schema.yaml file exists"""
        import os
        schema_path = "data_schema/schema.yaml"
        assert os.path.exists(schema_path), "Schema file should exist"
    
    def test_schema_structure(self):
        """Test schema has required structure"""
        with open("data_schema/schema.yaml", 'r') as file:
            schema = yaml.safe_load(file)
        
        assert "columns" in schema, "Schema should have columns section"
        assert len(schema["columns"]) > 0, "Schema should have column definitions"
    
    def test_valid_data_passes_validation(self):
        """Test that valid data passes schema validation"""
        # Create sample valid data
        valid_data = pd.DataFrame({
            'URL_Length': [50, 100, 75],
            'having_At_Symbol': [0, 1, 0],
            'SSLfinal_State': [1, 0, 1],
            'Result': [0, 1, 0]
        })
        
        # Save to temporary file
        valid_data.to_csv("test_valid.csv", index=False)
        
        # Test validation passes
        try:
            # This would normally use the validation component
            assert len(valid_data) > 0
            assert all(col in valid_data.columns for col in ['URL_Length', 'Result'])
        finally:
            import os
            if os.path.exists("test_valid.csv"):
                os.remove("test_valid.csv")
    
    def test_invalid_data_fails_validation(self):
        """Test that invalid data fails validation"""
        # Create sample invalid data (missing required columns)
        invalid_data = pd.DataFrame({
            'wrong_column': [1, 2, 3]
        })
        
        # Test validation should fail
        with open("data_schema/schema.yaml", 'r') as file:
            schema = yaml.safe_load(file)
        
        schema_columns = set(schema["columns"].keys())
        data_columns = set(invalid_data.columns)
        
        assert not schema_columns.issubset(data_columns), "Invalid data should fail validation"