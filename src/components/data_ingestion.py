import sys, os
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts', 'raw.csv')
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info('Data Ingestion started.')
        try:
            df = pd.read_csv(os.path.join('notebooks/data/train', 'train.csv'))
            logging.info('Read data as pandas DataFrame')

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False)

            logging.info('Saved Raw Data')
            logging.info('Train test Split Started')

            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)

            train_set.to_csv(self.data_ingestion_config.train_data_path, header=True, index=False)
            logging.info('Train data saved')

            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)
            logging.info('Test data saved')
            logging.info('Data Ingestion Complete')


            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Error occured in Data Ingestion')
            raise CustomException(e, sys)