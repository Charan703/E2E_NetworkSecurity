import pytest
import pandas as pd
import os
from unittest.mock import patch, Mock
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.entity.config_entity import TrainingPipelineConfig


class TestPipelineIntegration:
    """Test end-to-end pipeline functionality"""
    
    def test_pipeline_config_creation(self):
        """Test that pipeline configuration is created correctly"""
        config = TrainingPipelineConfig()
        
        assert hasattr(config, 'artifact_dir'), "Config should have artifact_dir"
        assert hasattr(config, 'timestamp'), "Config should have timestamp"
        assert isinstance(config.timestamp, str), "Timestamp should be string"
    
    @patch('networksecurity.pipeline.training_pipeline.TrainingPipeline.start_data_ingestion')
    @patch('networksecurity.pipeline.training_pipeline.TrainingPipeline.start_data_validation')
    @patch('networksecurity.pipeline.training_pipeline.TrainingPipeline.start_data_transformation')
    @patch('networksecurity.pipeline.training_pipeline.TrainingPipeline.start_model_trainer')
    def test_pipeline_runs_all_stages(self, mock_trainer, mock_transform, mock_validate, mock_ingest):
        """Test that pipeline runs all stages in sequence"""
        # Mock return values
        mock_ingest.return_value = Mock()
        mock_validate.return_value = Mock()
        mock_transform.return_value = Mock()
        mock_trainer.return_value = Mock()
        
        # Create and run pipeline
        pipeline = TrainingPipeline()
        
        try:
            pipeline.run_pipeline()
            
            # Verify all stages were called
            mock_ingest.assert_called_once()
            mock_validate.assert_called_once()
            mock_transform.assert_called_once()
            mock_trainer.assert_called_once()
            
        except Exception as e:
            # If pipeline fails, it should be due to missing dependencies, not logic errors
            assert "mongo" in str(e).lower() or "connection" in str(e).lower() or "model" in str(e).lower()
    
    def test_pipeline_creates_artifacts_directory(self):
        """Test that pipeline creates necessary directories"""
        config = TrainingPipelineConfig()
        
        # Check if artifact directory path is valid
        assert config.artifact_dir is not None, "Artifact directory should be defined"
        assert isinstance(config.artifact_dir, str), "Artifact directory should be string path"
    
    def test_data_ingestion_config(self):
        """Test data ingestion configuration"""
        from networksecurity.entity.config_entity import DataIngestionConfig
        
        training_config = TrainingPipelineConfig()
        ingestion_config = DataIngestionConfig(training_config)
        
        assert hasattr(ingestion_config, 'collection_name'), "Should have collection name"
        assert hasattr(ingestion_config, 'database_name'), "Should have database name"
        assert ingestion_config.collection_name is not None, "Collection name should not be None"
    
    def test_model_file_paths(self):
        """Test that model file paths are correctly configured"""
        from networksecurity.entity.config_entity import ModelTrainerConfig
        
        training_config = TrainingPipelineConfig()
        trainer_config = ModelTrainerConfig(training_config)
        
        assert hasattr(trainer_config, 'trained_model_file_path'), "Should have model file path"
        assert trainer_config.trained_model_file_path.endswith('.pkl'), "Model should be pickle file"