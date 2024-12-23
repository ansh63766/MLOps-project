import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')
    train_data_path: str = os.path.join('artifacts', 'train_data.csv')
    test_data_path: str = os.path.join('artifacts', 'test_data.csv')
    test_size: float = 0.2
    random_state: int = 42

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process")
        try:
            df = pd.read_csv(self.config.raw_data_path)
            logging.info(f"Loaded dataset from {self.config.raw_data_path}")
            
            train_df, test_df = train_test_split(
                df, 
                test_size=self.config.test_size, 
                random_state=self.config.random_state
            )
            logging.info("Split data into train and test sets")

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)
            logging.info(f"Saved train data to {self.config.train_data_path}")
            logging.info(f"Saved test data to {self.config.test_data_path}")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)

if __name__ == "__main__":
    config = DataIngestionConfig()
    data_ingestion = DataIngestion(config)
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()