from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.logging.logger import logging
import os, sys
import pandas as pd
from scipy.stats import ks_2samp
from networksecurity.constants.training_piepline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(
        self,
        data_validation_config: DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        try:
            logging.info(f"{'>>' * 20} Data Validation {'<<' * 20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self.schema_config["columns"])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    def is_numerical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        try:
            numerical_columns = self.schema_config["numerical_columns"]
            dataframe_columns = dataframe.columns

            numerical_column_present = True
            missing_numerical_columns = []
            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_column_present = False
                    missing_numerical_columns.append(num_column)

            logging.info(
                f"Missing numerical columns: [{missing_numerical_columns}]"
            )

            return numerical_column_present
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
   
    def detect_data_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold = 0.05) -> bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1, d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({column:
                               {"p_value":float(is_same_dist.pvalue),
                                "drift_status":is_found}})
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(file_path = drift_report_file_path, content = report)

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation process")
            # Load the training and testing data
            train_df = pd.read_csv(self.data_ingestion_artifact.training_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.testing_file_path)

            # Validate the data based on schema and drift detection
            status = self.validate_number_of_columns(dataframe = train_df)
            if not status:
                error_message = "Train data does not have required number of columns\n"
            status = self.validate_number_of_columns(dataframe = test_df)
            if not status:
                error_message = "Test data does not have required number of columns\n"
            
            status = self.is_numerical_column_exist(dataframe = train_df)
            if not status:
                error_message = "Train data does not have required numerical columns\n"
            status = self.is_numerical_column_exist(dataframe = test_df)
            if not status:
                error_message = "Test data does not have required numerical columns\n"
            
            status = self.detect_data_drift(base_df = train_df, current_df = test_df)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index = False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index = False, header=True)

            # For demonstration, we assume validation is successful
            data_validation_artifact = DataValidationArtifact(
                validation_status = status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info("Data validation process completed successfully")
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e