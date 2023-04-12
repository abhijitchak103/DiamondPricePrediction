import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation Started')

            # Define categorical and nnumerical columns
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # Define the ranking of ordinal features
            cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

            logging.info('Numeical Pipeline Initiated')
            num_pipeline = Pipeline(
                steps = [
                ('Imputer', SimpleImputer(strategy='median')),
                ('Scaler', StandardScaler())
                ]
            )

            logging.info('Numerical Pipeline Created')
            logging.info('Categorical Pipeline Initiated')
            cat_pipeline = Pipeline(
                steps=[
                ('Imputer', SimpleImputer(strategy='most_frequent')),
                ('Ordinal_Encoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
                ('Scaler', StandardScaler())
                ]
            )
            logging.info('Categorical Pipeline Created')

            logging.info('Column Transformer Initiated')
            preprocessor = ColumnTransformer(
                [
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
                ]
            )

            logging.info('Pipeline Completed')

            return preprocessor

        except Exception as e:
            logging.info('Error occured in Data Transformation Stage')
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data')
            logging.info(f'Train DataFrame Head: \n{train_df.head().to_string()}')
            logging.info(f"Test DataFrame Head: \n{test_df.head().to_string()}")

            logging.info("Obtaining preprocessor object")

            preprocessor_obj = self.get_data_transformation_object()

            target_column = 'price'
            columns_to_drop = ['id', target_column]

            input_feature_train_df = train_df.drop(columns = columns_to_drop, axis = 1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=columns_to_drop, axis = 1)
            target_feature_test_df = test_df[target_column]

            # Transforming using preprocessor obj
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info('Train and test data transformed.')

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path
                obj = preprocessor_obj
            )

            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info('Error Occured in initiate_data_transformation')
            raise CustomException(e, sys)    
