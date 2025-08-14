import os
import sys 
from src.exception import customException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# run the whole file usingh the Run from project root using -m command 
# python -m src.components.data_ingestion
@dataclass 
#This is a decorator from Python’s dataclasses module.
#It automatically generates special methods for the class such as:
#__init__ (constructor)
#__repr__ (string representation)
#__eq__ (equality comparison)
#This saves you from writing boilerplate code.
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

#This defines a configuration object where train_data_path defaults to the location of your training CSV file inside the artifacts folder.
#When you create a new DataIngestionConfig() object without parameters, it will automatically set train_data_path to "artifacts/train.csv".

# Here, the data ingestion knows where to store train,test and data due to this file path 

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Entered tge data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/StudentsPerformance.csv')
            logging.info('Read the dataset as dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e :
            raise customException(e,sys)
        
# when DataIngestion class is called the three paths defined in DataIngestionConfig will be saved in ingestion_config
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()