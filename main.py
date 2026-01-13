from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_training import ModelTrainer
import sys

if __name__ == "__main__":
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Data Ingestion started.")
        dataingestionartifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion completed.")
        print(dataingestionartifact)

        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(data_validation_config, dataingestionartifact)
        logging.info("Data Validation started.")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation completed.")
        print(data_validation_artifact)

        data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
        logging.info("Data Transformation started.")
        data_transformation = DataTransformation(data_transformation_config, data_validation_artifact)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data Transformation completed.")
        print(data_transformation_artifact)

        logging.info("Model Training started.")
        model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
        model_trainer = ModelTrainer(data_transformation_artifact, model_trainer_config)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model Training completed.")
        print(model_trainer_artifact)
    except Exception as e:
        raise NetworkSecurityException(e, sys)