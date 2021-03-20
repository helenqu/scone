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
def define_and_compile_model(input_shape, categorical, num_types):
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
def split_and_retrieve_data(heatmaps_path, train_proportion, include_test_set, input_shape, categorical, batch_size):
    raw_dataset = tf.data.TFRecordDataset(
        ["{}/{}".format(heatmaps_path, f.name) for f in os.scandir(heatmaps_path) if "tfrecord" in f.name], 
        num_parallel_reads=80)
    dataset = raw_dataset.map(lambda x: get_images(x, input_shape, categorical), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())

    train_set, val_set, test_set = stratified_split(dataset, train_proportion, include_test_set)
    train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE).cache()
    val_set = val_set.prefetch(tf.data.experimental.AUTOTUNE).cache()
    test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE).cache() if test_set else None

    return extract_ids_and_batch(train_set, val_set, test_set, batch_size)

# train the model, returns trained model & training log
# requires:
#   - model
#   - train_set, val_set
#   - NUM_EPOCHS
#   - batch_size
def train(model, train_set, val_set, num_epochs):
    history = model.fit(
        train_set,
        epochs=num_epochs,
        validation_data=val_set,
        verbose=0)

    return model, history

def get_predictions(model, dataset, dataset_ids, categorical):
    predictions = model.predict(dataset, verbose=0)
    if categorical:
        predictions = np.argmax(predictions, axis=1) #TODO: is this the best way to return categorical results?
    predictions = predictions.flatten()
    df_dict = {'snid': dataset_ids, 'pred': predictions}
    return df_dict

def test(model, test_set):
    results = model.evaluate(test_set)
    return results[1]
