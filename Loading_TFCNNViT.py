# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import tensorflow as tf

'''
TFCNNViT Model Calling Code

This code is a companion to the manuscript "C&G" submitted to [Journal Name].
It is used to call the MoE-XGB ground motion prediction model, which incorporates
station and seismic source latitude/longitude features.
If you encounter any issues, please leave a comment on the GitHub repository:
https://github.com/yourusername/TFCNNViT

Instructions:
- Modify the parameters between the "====" sections as indicated.
- The code between "****" sections is for calculations and should not be changed.
'''

# load adaptive weighted huber loss
class AdaptiveWeightedHuberLoss(tf.keras.losses.Loss):
    def __init__(self, base_delta=1.0, initial_alpha=2.0, reduction=tf.keras.losses.Reduction.AUTO, name="adaptive_weighted_huber_loss"):
        super().__init__(reduction=reduction, name=name)
        self.base_delta = base_delta  # Base value for delta in Huber loss
        self.alpha = tf.Variable(initial_alpha, trainable=True, dtype=tf.float16)

    def call(self, y_true, y_pred):
        error = tf.abs(y_true - y_pred)

        # Adaptively scale delta based on error magnitude
        # Larger errors lead to a larger delta → stronger L1 influence
        adaptive_delta = self.base_delta * (1.0 + tf.math.log(1.0 + error))

        # Compute Huber loss based on adaptive delta
        huber_loss = tf.where(
            error < adaptive_delta,
            0.5 * tf.square(error),
            adaptive_delta * (error - 0.5 * adaptive_delta)
        )

        # Apply sample weighting: assign higher weight to samples with y_true >= 6.5 (large-magnitude events)
        # Weight is trainable (alpha), learned by the model
        sample_weight = tf.where(y_true >= 6.5, self.alpha, 1.0)

        # Return weighted average loss
        return tf.reduce_mean(huber_loss * tf.cast(sample_weight, tf.float16))


def make_dataset(time_x, freq_x, aux, y, batch_size=256, training=True):
    ds = tf.data.Dataset.from_tensor_slices(((time_x, freq_x, aux), y))
    if training:
        ds = ds.shuffle(10000)
    ds = ds.batch(batch_size)
    ds = ds.cache()
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def read_dataset(data_path, acc_len, freq_len):
    data = pd.read_csv(data_path)
    data = data.sample(frac=1).reset_index(drop=True)
    time_x = data.iloc[:, :acc_len].values.reshape(-1, 1, acc_len, 1)
    frequency_x = data.iloc[:, acc_len:acc_len + freq_len].values.reshape(-1, 1, freq_len, 1)
    aux = data.loc[:, ['EpiDist', 'Depth', "Vs30"]].values.reshape(-1, 1, 3, 1)
    y = data.loc[:, 'Mag'].values.reshape(-1, 1)

    return time_x, frequency_x, aux, y


# 1. load the data file path
# ===========================================================================
# Example: r'C:\Users\TFCNNViT\Dataset\300_256.csv'
data_path = r''
# ===========================================================================

# 2. load the model file path
# ===========================================================================
# Example: r'C:\Users\TFCNNViT\Model\300'
model_path = r''
# ===========================================================================

# 3. load model and predict
# ===========================================================================
# Example--3s time window
# loade dataset:
    # Each sample is composed of three components: a time-domain sequence (length 300), a frequency-domain sequence (length 256),
    # and auxiliary information including 'EpiDist', 'Depth', and 'Vs30'. The corresponding label is the earthquake magnitude (mag).
    # the shape is [batch, 300+256+3+1]

acc_len = 300
if acc_len in [300, 400, 500]:
    freq_len = 256
else:
    freq_len = 512
time_x, frequency_x, aux, y = read_dataset(data_path, acc_len, freq_len)

# load model
model = tf.keras.models.load_model(model_path, custom_objects={'AdaptiveWeightedHuberLoss': AdaptiveWeightedHuberLoss})
mag_pred = model.predict((time_x, frequency_x, aux))
mag_df = pd.DataFrame({
        'mag_true': y.flatten(),
        'mag_pred': mag_pred.flatten()
    })
# ===========================================================================

# 4. Set the path for the model prediction data output
# ===========================================================================
# Example: r'C:\Users\TFCNNViT\Result\300.csv'
save_path = r''
mag_df.to_csv(save_path)
# ===========================================================================