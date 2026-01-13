import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os, sys
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as yaml_file:
            yaml.dump(content, yaml_file)

    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        logging.info("File path: {}".format(file_path))
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_object(file_path: str) -> object:
    try:
        logging.info("Entered the load_object method of MainUtils class")
        logging.info("File path: {}".format(file_path))
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logging.info("Exited the load_object method of MainUtils class")
        return obj
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    try:
        logging.info("Entered the load_numpy_array_data method of MainUtils class")
        logging.info("File path: {}".format(file_path))
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            array = np.load(file_obj, allow_pickle=True)
        logging.info("Exited the load_numpy_array_data method of MainUtils class")
        return array
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e

def evaluate_models(x_train, y_train, x_test, y_test, models: dict, params: dict) -> dict:
    try:
        model_report = {}
        for model_name, model in models.items():
            param = params[model_name]
            grid_search = GridSearchCV(model, param, cv=5)
            grid_search.fit(x_train, y_train)
            model.set_params(**grid_search.best_params_)
            model.fit(x_train, y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_metric = r2_score(y_test, y_test_pred)
            model_report[model_name] = test_metric
        return model_report
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e