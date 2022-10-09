import os
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from src.utils.common import read_yaml, create_directories
from src.utils.models import callbacks
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


STAGE = "STAGE 3"

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
    preprocessed_data_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["PREPROCESSED_DATA"])
    train_file = os.path.join(preprocessed_data_dir_path, artifacts["TRAIN_DATA"])
    test_file = os.path.join(preprocessed_data_dir_path, artifacts["TEST_DATA"])

    train_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["TRAIN_DIR"])
    create_directories([train_dir_path])
    checkpoint_dir = os.path.join(train_dir_path, artifacts["CHECKPOINT_DIR"])
    tb_root_log_dir = os.path.join(train_dir_path, artifacts["TB_ROOT_LOG_DIR"])
    ckpt_model = artifacts["CHECKPOINT_MODEL"]
    model_dir = os.path.join(train_dir_path, artifacts["MODEL_DIR"])
    create_directories([model_dir])
    model_file = os.path.join(model_dir, artifacts["MODEL_FILE"])

    vocab_size = params["train"]["vocab_size"]
    output_dim = params["train"]["output_dim"]
    epochs = params["train"]["epochs"]
    lstm_units = params["train"]["lstm_units"]
    validation_steps = params["train"]["validation_steps"]

    train_ds = tf.data.Dataset.load(train_file)
    test_ds = tf.data.Dataset.load(test_file)
    #train_ds = tf.convert_to_tensor(X_train, dtype=tf.string)

    # text encoding
    encoder = tf.keras.layers.TextVectorization(max_tokens=vocab_size)
    encoder.adapt(train_ds.map(lambda text, label: text))

    # first 20 tokens - 
    vocab = np.array(encoder.get_vocabulary())
    print(vocab[:20])
    print(len(encoder.get_vocabulary()))

    embedding_layer = tf.keras.layers.Embedding(
                        input_dim = len(encoder.get_vocabulary()), # 1000  
                        output_dim = output_dim, # 64               
                        mask_zero = True
                        ) 

    Layers = [
          encoder, # text vectorization
          embedding_layer, # embedding
          tf.keras.layers.Bidirectional(
              tf.keras.layers.LSTM(lstm_units)
          ),
          tf.keras.layers.Dense(lstm_units, activation="relu"),
          tf.keras.layers.Dense(1)
    ]

    model = tf.keras.Sequential(Layers)
    print(model.summary())

    model.compile(
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=["accuracy"]
            )
    
    callback_list = callbacks(tb_root_log_dir, checkpoint_dir, ckpt_model)

    history = model.fit(train_ds,
                    epochs=epochs,
                    validation_data=test_ds,
                    validation_steps=validation_steps,
                    callbacks=callback_list)

    model.save(model_file)


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
