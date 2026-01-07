from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

import os, sys
import pymongo
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info(f"{'='*20} Data Ingestion log started. {'='*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    def export_collection_as_dataframe(self):
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]
            data = pd.DataFrame(list(collection.find()))
            if "_id" in data.columns.to_list():
                data = data.drop(columns=["_id"], axis=1)
            
            data.replace({"na": np.nan}, inplace=True)
            return data

        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)
            dataframe.to_csv(self.data_ingestion_config.feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        try:
            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.test_size,
                random_state=42
            )
            logging.info("Performed train test split on the dataframe")
            logging.info(f"Exporting train and test data to file path")
            logging.info(f"Train data shape: {train_set.shape}")

            train_file_path = self.data_ingestion_config.train_file_path
            test_file_path = self.data_ingestion_config.test_file_path

            train_dir = os.path.dirname(train_file_path)
            test_dir = os.path.dirname(test_file_path)

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)

            train_set.to_csv(train_file_path, index=False, header=True)
            test_set.to_csv(test_file_path, index=False, header=True)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            dataingestionartifact = DataIngestionArtifact(
                training_file_path=self.data_ingestion_config.train_file_path,
                testing_file_path=self.data_ingestion_config.test_file_path
            )
            return dataingestionartifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

     