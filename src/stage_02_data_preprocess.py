import os
import argparse
import logging
import random
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.utils.common import read_yaml, create_directories
from src.utils.data_management import preprocess_df
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


STAGE = "STAGE 2" 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path, params_path):
    ## read config files
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    
    artifacts = config["artifacts"]
    prepared_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["PREPARED_DATA"])
    dataset_file = os.path.join(prepared_data_dir_path, artifacts["DATASET_FILE"])

    preprocessed_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["PREPROCESSED_DATA"])
    create_directories([preprocessed_data_dir_path])

    train_file = os.path.join(preprocessed_data_dir_path, artifacts["TRAIN_DATA"])
    test_file = os.path.join(preprocessed_data_dir_path, artifacts["TEST_DATA"])

    split = params["preprocess"]["split"]
    seed = params["preprocess"]["seed"]
    batch_size = params["preprocess"]["batch_size"]
    buffer_size = params["preprocess"]["buffer_size"]
    random.seed(seed)

    dataset = pd.read_csv(dataset_file)
    df = preprocess_df(dataset)
    print(df.head())

    X = df[['text']]
    y = df['label']

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # train_ds = pd.concat([X_train, pd.DataFrame(y_train)], axis=1)
    # test_ds = pd.concat([X_test, pd.DataFrame(y_test)], axis=1)

    # train_ds.to_csv(train_file, index=False)
    # test_ds.to_csv(test_file, index=False)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train['text'].to_numpy(), y_train.to_list()))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test['text'].to_numpy(), y_test.to_list()))

    for example, label in train_ds.take(3):
        print(f"sample text:\n{example.numpy()}\n")
        print(f"label:\n{label.numpy()}\n")

    #shuffling and batching the training dataset
    train_ds = train_ds.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)  # Prefetch readys next batch data
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    for example, label in train_ds.take(1):
        print(f"sample text:\n{example.numpy()}\n")
        print(f"label:\n{label.numpy()}\n")
        print(f"label:\n{len(label.numpy())}\n")
        break

    tf.data.Dataset.save(train_ds, train_file)
    tf.data.Dataset.save(test_ds, test_file)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
