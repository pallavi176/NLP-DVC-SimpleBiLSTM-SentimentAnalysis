import os
import argparse
import logging
import pandas as pd
from src.utils.common import read_yaml, create_directories
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


STAGE = "STAGE 1"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    
    source_data = config["source_data"]
    input_data = os.path.join(source_data["data_dir"], source_data["data_file"])

    artifacts = config["artifacts"]
    prepared_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["PREPARED_DATA"])
    create_directories([prepared_data_dir_path])

    dataset_file = os.path.join(prepared_data_dir_path, artifacts["DATASET_FILE"])

    df = pd.read_csv(input_data, encoding='ISO-8859-1', header=None)
    #print(df.head())

    dataset = pd.DataFrame()
    dataset[['text','label']] = df[[5,0]]
    #print(dataset.head())

    dataset['label'].replace({4:1}, inplace=True)
    dataset.to_csv(dataset_file, index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
