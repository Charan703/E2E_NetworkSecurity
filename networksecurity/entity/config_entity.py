from datetime import datetime
import os
from networksecurity.constants import training_piepline

print(training_piepline.PIPELINE_NAME)
print(training_piepline.ARTIFACT_DIR)

class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = training_piepline.PIPELINE_NAME
        self.artifact_name = training_piepline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.timestamp: str = timestamp


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.database_name: str = training_piepline.DATA_INGESTION_DATABASE_NAME
        self.collection_name: str = training_piepline.DATA_INGESTION_COLLECTION_NAME
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_piepline.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, training_piepline.DATA_INGESTION_FEATURE_STORE_DIR, training_piepline.FILE_NAME
        )
        self.train_file_path: str = os.path.join(
            self.data_ingestion_dir, training_piepline.DATA_INGESTION_INGESTED_DIR, training_piepline.TRAIN_FILE_NAME
        )
        self.test_file_path: str = os.path.join(
            self.data_ingestion_dir, training_piepline.DATA_INGESTION_INGESTED_DIR, training_piepline.TEST_FILE_NAME
        )
        self.train_test_split_ratio: float = training_piepline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.test_size: float = training_piepline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        