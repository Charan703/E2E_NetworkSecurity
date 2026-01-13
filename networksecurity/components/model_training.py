import os, sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.main_utils.utils import load_numpy_array_data, save_object, load_object, evaluate_models
from networksecurity.utils.ml_utils.model.estimator import NetwrokModel
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
import mlflow

import dagshub
dagshub.init(repo_owner='charanteja.kammari939', repo_name='E2E_NetworkSecurity', mlflow=True)

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    def track_mlflow(self, model, classification_metric):
        try:
            # Use DagsHub remote tracking (remove local tracking URI)
            with mlflow.start_run():
                f1_score = classification_metric.f1_score
                precision = classification_metric.precision_score
                recall = classification_metric.recall_score
                
                mlflow.log_metric("F1_Score", f1_score)
                mlflow.log_metric("Precision", precision)
                mlflow.log_metric("Recall", recall)
                mlflow.sklearn.log_model(model, "model")
                
                logging.info(f"MLflow tracking - F1: {f1_score}, Precision: {precision}, Recall: {recall}")
                
        except Exception as e:
            # Fallback to logging if MLflow fails
            logging.warning(f"MLflow tracking failed: {e}")
            logging.info(f"Model metrics - F1: {classification_metric.f1_score}, Precision: {classification_metric.precision_score}, Recall: {classification_metric.recall_score}")
    def train_model(self, x_train, y_train, x_test, y_test) -> ModelTrainerArtifact:
        logging.info("Training the model")
        try:
            logging.info("Splitting training and testing input data")
            models = {
                "LogisticRegression": LogisticRegression(),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "RandomForestClassifier": RandomForestClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
            }
            logging.info("Training the models")
            params = {
                "LogisticRegression": {"C": [0.1, 0.5, 1.0], "max_iter": [100, 200, 300]},
                "KNeighborsClassifier": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
                "RandomForestClassifier": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
                "GradientBoostingClassifier": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
                "AdaBoostClassifier": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
                "DecisionTreeClassifier": {"max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]},
            }
            logging.info("Evaluating the models")
            model_report: dict = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params,
            )
            logging.info(f"Model evaluation report: {model_report}")
            logging.info(f"Model report keys: {list(model_report.keys())}")
            logging.info(f"Model report values: {list(model_report.values())}")
            
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            
            # Fit the best model
            best_model.fit(x_train, y_train)
            y_train_pred = best_model.predict(x_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

            ## function to track in MLFLOW
            self.track_mlflow(best_model, classification_train_metric)

            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            self.track_mlflow(best_model, classification_test_metric)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.preprocessor_object_file_path)
            Network_model = NetwrokModel(model=best_model, preprocessing_object=preprocessing_obj)
            save_object(self.model_trainer_config.trained_model_file_path, obj=Network_model)

            save_object("final_model/model.pkl", obj=best_model)

            ## model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
            )
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading transformed training dataset")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_array = load_numpy_array_data(file_path=transformed_train_file_path)

            logging.info("Splitting training dataset into input and target feature")
            x_train, y_train = train_array[:, :-1], train_array[:, -1]

            logging.info("Loading transformed testing dataset")
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path
            test_array = load_numpy_array_data(file_path=transformed_test_file_path)
            logging.info("Splitting testing dataset into input and target feature")
            x_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            logging.info("Starting model training")
            model_trainer_artifact = self.train_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            logging.info("Model training completed successfully")
            return model_trainer_artifact
           
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e