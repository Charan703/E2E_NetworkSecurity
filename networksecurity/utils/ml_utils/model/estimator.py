from networksecurity.constants.training_piepline import SAVED_MODEL_DIR,  MODEL_FILE_NAME

import os, sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetwrokModel:
    def __init__(self, model, preprocessing_object):
        """
        model: ML model object
        preprocessing_object: preprocessing object
        """
        try:
            self.model = model
            self.preprocessing_object = preprocessing_object
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def predict(self, X):
        """
        X: input features
        return: predicted labels
        """
        try:
            X_transformed = self.preprocessing_object.transform(X)
            return self.model.predict(X_transformed)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e