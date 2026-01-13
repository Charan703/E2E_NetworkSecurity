import os, sys
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from networksecurity.constants.training_piepline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from networksecurity.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact
)
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(f"{'>>'*20} Data Transformation {'<<'*20}")
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        logging.info("Creating Data Transformation Pipeline")
        try:
            logging.info("Creating Data Transformation Pipeline")
            imputer: KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            preprocessor: Pipeline = Pipeline(
                steps=[
                    ("Imputer", imputer)
                ]
            )
            return preprocessor
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Initiating Data Transformation")
        try:
            logging.info("Reading training and testing file")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            logging.info("Train and Test file read successfully")
            # Selecting input feature for train and test dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            # Selecting target feature for train and test dataframe
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1,0)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1,0)
            logging.info("Applying imputation on train and test data")
            # Get the transformer object
            preprocessor = DataTransformation.get_data_transformer_object()
            # Imputing missing values in train and test file
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            # Concatenating input and target feature
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            imputer = preprocessor.named_steps['Imputer']
            
            # Saving transformed train and test array
            logging.info("Saving transformed train and test array")
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.preprocessor_object_file_path, obj=imputer)

            save_object("final_model/preprocessor.pkl",obj = preprocessor)
            # Preparing artifacts
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                preprocessor_object_file_path=self.data_transformation_config.preprocessor_object_file_path
            )
            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e