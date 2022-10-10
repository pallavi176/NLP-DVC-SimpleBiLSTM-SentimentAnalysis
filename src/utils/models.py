import os
import time
import tensorflow as tf

def callbacks(tb_root_log_dir, checkpoint_dir, ckpt_model):

    # tensorboard callbacks - 
    unique_log = time.asctime().replace(" ", "_").replace(":", "")
    tensorboard_log_dir = os.path.join(tb_root_log_dir, unique_log)
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir)

    # ckpt callback
    ckpt_file = os.path.join(checkpoint_dir, ckpt_model)
    os.makedirs(checkpoint_dir, exist_ok=True)

    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_file,
        save_best_only=True
    )

    callback_list = [
                    tb_cb,
                    ckpt_cb
                    ]

    return callback_list