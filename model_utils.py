import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, utils, optimizers
import h5py

from data_utils import *

# defines and compiles, then returns model
# requires:
#   - INPUT_SHAPE
#   - CATEGORICAL
#   - NUM_TYPES
def define_and_compile_model(input_shape, categorical, num_types=None):
    y, x, _ = input_shape
    
    model = models.Sequential()

    model.add(layers.ZeroPadding2D(padding=(0,1), input_shape=input_shape))
    model.add(layers.Conv2D(y, (y, 3), activation='elu'))
    model.add(layers.Reshape((y, x, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.ZeroPadding2D(padding=(0,1)))
    model.add(layers.Conv2D(y, (y, 3), activation='elu'))
    model.add(layers.Reshape((y, x, 1)))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.ZeroPadding2D(padding=(0,1)))
    model.add(layers.Conv2D(int(y/2), (int(y/2), 3), activation='elu'))
    model.add(layers.Reshape((int(y/2), int(x/2), 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.ZeroPadding2D(padding=(0,1)))
    model.add(layers.Conv2D(int(y/2), (int(y/2), 3), activation='elu'))
    model.add(layers.Reshape((int(y/2), int(x/2), 1)))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.3))

    if categorical:
        model.add(layers.Dense(num_types, activation='softmax'))
    else:
        model.add(layers.Dense(1, activation='sigmoid'))
    
    opt = optimizers.Adam(learning_rate=1e-3)
    loss = 'sparse_categorical_crossentropy' if categorical else 'binary_crossentropy'
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=['accuracy'])

    return model

# main data retrieval function
# requires:
#   - heatmaps_path
#   - get_images
#   - stratified_split
#   - extract_ids
def split_and_retrieve_data(heatmaps_path):
    raw_dataset = tf.data.TFRecordDataset(
        ["{}/{}".format(heatmaps_path, f.name) for f in os.scandir(heatmaps_path) if "tfrecord" in f.name], 
        num_parallel_reads=80)
    dataset = raw_dataset.map(lambda x: get_images(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())

    train_set, val_set, test_set = stratified_split(dataset)
    train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE).cache()
    val_set = val_set.prefetch(tf.data.experimental.AUTOTUNE).cache()
    test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE).cache()

    return extract_ids(train_set, val_set, test_set)

# train the model, returns trained model & training log
# requires:
#   - model
#   - train_set, val_set
#   - NUM_EPOCHS
#   - batch_size
def train(model, train_set, val_set, batch_size, num_epochs):
    train_set = train_set.batch(batch_size)
    val_set = val_set.batch(batch_size)

    history = model.fit(
        train_set,
        epochs=num_epochs,
        validation_data=val_set,
        verbose=1)

    return model, history

def test(model, test_set, batch_size):
    test_set = test_set.batch(batch_size)
    results = model.evaluate(test_set)
    print("test loss: {}, test accuracy: {}".format(test_results[0], test_results[1]))