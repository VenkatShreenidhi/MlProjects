import os
import sys 
from src.exception import customException
from src.logger import logging
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_tranformation import DataTransformation
from src.components.data_tranformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

# run the whole file usingh the Run from project root using -m command 
# python -m src.components.data_ingestion
@dataclass 
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

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
    train_data,test_data=obj.initiate_data_ingestion()
    
    dataTransformation= DataTransformation()
    train_arr, test_arr,_= dataTransformation.initiate_data_transformation(train_data,test_data)

    modelTrainer=ModelTrainer()
    print(modelTrainer.initiate_model_trainer(train_arr,test_arr))
    


# Explain the code
#This code implements a Data Ingestion component that reads raw data, splits it into training and testing sets, 
# and saves them as separate CSV files. 

#1. @dataclass -> this is a class decorator. 
'''
It basically auto generates the __init__ constructor for this type of code
Meaning: 
If @dataclass is not used; we may have to define it as 
def __init__(self, train_data_path, test_data_path, raw_data_path):
    self.train_data_path = train_data_path
    self.test_data_path = test_data_path
    self.raw_data_path = raw_data_path

DataIngestionConfig -> Class defines where the training,test and raw data files should be stored on PC.
'''

# DataIngestion

'''
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

A class DataIngestion is created. 
here, self is the DataIngestion object 
i.e self.ingestion_config is an object/property of the class DataIngestion
when DataIngestionConfig is assigned to it. 
basically all the properties or objects defined in DataIngestionConfig
is stored in self.ingestion_config as well.
An instance of DataIngestionConfig is created in self.ingestion_config

'''

# def initiate data ingestion -> return the path of train and test path after data has been split 

'''
try:
            LOADING THE DATASET and reading as dataframe
            df = pd.read_csv('notebook/data/StudentsPerformance.csv')
            logging.info('Read the dataset as dataframe')
            Since all paths are save in one directory, only one time the directory is created and train data and checked if the directory 'artifacts is present' 
             
            #is a pandas DataFrame method that saves/exports a DataFrame to a CSV
            # why? to save the files locally in the csv format 

            Index = false (No indexing)
# Without index=False (DEFAULT):
# Saves with row numbers
   ,Name,Age,Score
0  ,John,25,85
1  ,Jane,30,92
2  ,Bob,22,78

# With index=False:  
# No row numbers saved
Name,Age,Score
John,25,85
Jane,30,92
Bob,22,78

# Header = True 
# With header=True (DEFAULT):
# Column names included
Name,Age,Score
John,25,85
Jane,30,92

# With header=False:
# No column names
John,25,85
Jane,30,92
self.ingestion_config.raw_data_path -> where to save the file 
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated")
            # normal splitting using Train_test_split 
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            # after splitting saving itt to the path 
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed")
'''
# test the file

'''
if __name__ == "__main__":
    obj = DataIngestion()           # Creates an object of DataIngestion class
    obj.initiate_data_ingestion()   # Runs/executes the main method

    This is like having a 'Run' button built into the file. When I execute the file directly, it automatically creates an object and tests the main function to make sure everything works correctly.

'''
