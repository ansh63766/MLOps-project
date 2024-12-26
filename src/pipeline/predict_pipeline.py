import sys
import os
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Paths to the model and preprocessor files
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            # Log the existence of files
            logging.info(f"Checking if model file exists: {os.path.exists(model_path)}")
            logging.info(f"Checking if preprocessor file exists: {os.path.exists(preprocessor_path)}")

            # Log the paths to confirm they are correct
            logging.info(f"Model path: {model_path}")
            logging.info(f"Preprocessor path: {preprocessor_path}")

            logging.info("Before Loading Model and Preprocessor")

            # Attempt to load the model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info("After Loading Model and Preprocessor")

            # Log the shape of the input data
            logging.info(f"Input features shape: {features.shape}")
            
            # Preprocess the features and make predictions
            data_scaled = preprocessor.transform(features)
            logging.info(f"Scaled data shape: {data_scaled.shape}")

            preds = model.predict(data_scaled)
            logging.info(f"Predictions: {preds}")

            return preds
        except Exception as e:
            logging.error(f"Error in PredictPipeline: {e}")
            raise CustomException(e, sys)

class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            # Log the input data before creating DataFrame
            logging.info(f"Creating DataFrame with the following data: "
                         f"Gender: {self.gender}, Race/Ethnicity: {self.race_ethnicity}, "
                         f"Parental Education: {self.parental_level_of_education}, Lunch: {self.lunch}, "
                         f"Test Preparation Course: {self.test_preparation_course}, "
                         f"Reading Score: {self.reading_score}, Writing Score: {self.writing_score}")

            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Create and return the DataFrame
            df = pd.DataFrame(custom_data_input_dict)

            # Log the resulting DataFrame
            logging.info(f"DataFrame created: {df}")
            return df
        except Exception as e:
            logging.error(f"Error in get_data_as_data_frame: {e}")
            raise CustomException(e, sys)