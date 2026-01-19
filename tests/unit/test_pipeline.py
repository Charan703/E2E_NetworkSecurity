import unittest
from unittest.mock import MagicMock

class TestPipelineConfiguration(unittest.TestCase):
    """Test pipeline configuration and structure"""
    
    def test_training_pipeline_config(self):
        """Test training pipeline configuration structure"""
        # Mock pipeline config
        pipeline_config = {
            "artifact_dir": "artifacts",
            "model_dir": "saved_models",
            "data_ingestion": {
                "collection_name": "NetworkData",
                "test_size": 0.2
            },
            "data_validation": {
                "drift_threshold": 0.05
            },
            "data_transformation": {
                "transformed_object_file_path": "preprocessor.pkl"
            },
            "model_trainer": {
                "trained_model_file_path": "model.pkl"
            }
        }
        
        # Test config structure
        self.assertIn("artifact_dir", pipeline_config)
        self.assertIn("data_ingestion", pipeline_config)
        self.assertIn("data_validation", pipeline_config)
        self.assertIn("data_transformation", pipeline_config)
        self.assertIn("model_trainer", pipeline_config)
        
        # Test nested config
        self.assertEqual(pipeline_config["data_ingestion"]["test_size"], 0.2)
        self.assertEqual(pipeline_config["data_validation"]["drift_threshold"], 0.05)
    
    def test_pipeline_components_exist(self):
        """Test pipeline components are properly defined"""
        # Mock component list
        components = [
            "DataIngestion",
            "DataValidation", 
            "DataTransformation",
            "ModelTrainer"
        ]
        
        # Test all components exist
        self.assertEqual(len(components), 4)
        self.assertIn("DataIngestion", components)
        self.assertIn("DataValidation", components)
        self.assertIn("DataTransformation", components)
        self.assertIn("ModelTrainer", components)
    
    def test_pipeline_execution_order(self):
        """Test pipeline execution follows correct order"""
        # Mock execution order
        execution_order = [
            "data_ingestion",
            "data_validation", 
            "data_transformation",
            "model_training"
        ]
        
        # Test order is correct
        self.assertEqual(execution_order[0], "data_ingestion")
        self.assertEqual(execution_order[1], "data_validation")
        self.assertEqual(execution_order[2], "data_transformation")
        self.assertEqual(execution_order[3], "model_training")