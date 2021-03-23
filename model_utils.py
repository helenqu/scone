import os
import numpy as np
import yaml
import tensorflow as tf
from tensorflow.keras import layers, models, utils, optimizers
import h5py

from data_utils import *

class SconeClassifier():
    def __init__(self, config):
        self.heatmaps_path = config['heatmaps_path']
        self.output_path = config['output_path']
        self.batch_size = config.get('batch_size', 32)
        self.num_epochs = config['num_epochs']
        self.input_shape = (config['num_wavelength_bins'], config['num_mjd_bins'], 2)
        self.categorical = config['categorical']
        self.num_types = config.get('num_types', None)
        if self.categorical and not self.num_types:
            raise KeyError('cannot perform categorical classification without knowing the number of source types! please specify the `num_types` key in your config file to reflect this information')
            # TODO: should i write num types info into a file after create heatmaps? maybe ids file will be large
            # ids_file = h5py.File(config['ids_path'], "r")
            # types = [x.decode('utf-8').split("_")[0] for x in ids_file["names"]]
            # ids_file.close()
            # self.num_types = len(np.unique(types))
        self.train_proportion = config.get('train_proportion', 0.8)
        self.use_test_set = True if config["mode"] == "predict" else False
        self.trained_model = None # TODO: or the location of a saved trained model
        self.train_set, self.val_set, self.test_set, self.train_ids, self.val_ids, self.test_ids = self._split_and_retrieve_data()

    def run(self):
        if not self.trained_model:
            _, history = self.train()
        dataset, dataset_ids = self.get_test_set() if self.use_test_set else self.get_train_set()
        preds_dict = self.predict(dataset, dataset_ids)

        if self.use_test_set:
            test_acc = self.test()
            history.history["test_accuracy"] = test_acc

        return preds_dict, history

    # train the model, returns trained model & training log
    # requires:
    #   - model
    #   - train_set, val_set
    #   - NUM_EPOCHS
    #   - batch_size
    def train(self):
        model = self._define_and_compile_model()
        history = model.fit(
            self.train_set,
            epochs=self.num_epochs,
            validation_data=self.val_set,
            verbose=0)

        self.trained_model = model
        return model, history

    def predict(self, dataset, dataset_ids):
        if not self.trained_model:
            raise RuntimeError('model has not been trained! call `train` on the SconeClassifier instance before predict!')
        predictions = self.trained_model.predict(dataset, verbose=0)
        if self.categorical:
            predictions = np.argmax(predictions, axis=1) #TODO: is this the best way to return categorical results? doesnt preserve confidence info
        predictions = predictions.flatten()
        df_dict = {'snid': dataset_ids, 'pred': predictions}
        return df_dict

    def test(self):
        if not self.use_test_set:
            raise RuntimeError('no testing in train mode')
        if not self.trained_model:
            raise RuntimeError('model has not been trained! call `train` on the SconeClassifier instance before test!')
        results = self.trained_model.evaluate(self.test_set)
        return results[1]

    def get_train_set(self):
        if not self.train_set:
            self.train_set, self.val_set, self.test_set, self.train_ids, self.val_ids, self.test_ids = self._split_and_retrieve_data()
        return self.train_set, self.train_ids

    def get_val_set(self):
        if not self.val_set:
            self.train_set, self.val_set, self.test_set, self.train_ids, self.val_ids, self.test_ids = self._split_and_retrieve_data()
        return self.val_set, self.val_ids

    def get_test_set(self):
        if not self.use_test_set:
            raise RuntimeError('no test set in train mode')
        if not self.test_set:
            self.train_set, self.val_set, self.test_set, self.train_ids, self.val_ids, self.test_ids = self._split_and_retrieve_data()
        return self.test_set, self.test_ids

    # defines and compiles, then returns model
    # requires:
    #   - INPUT_SHAPE
    #   - CATEGORICAL
    #   - NUM_TYPES
    def _define_and_compile_model(self):
        y, x, _ = self.input_shape
        
        model = models.Sequential()

        model.add(layers.ZeroPadding2D(padding=(0,1), input_shape=self.input_shape))
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

        if self.categorical:
            model.add(layers.Dense(self.num_types, activation='softmax'))
        else:
            model.add(layers.Dense(1, activation='sigmoid'))
        
        opt = optimizers.Adam(learning_rate=1e-3)
        loss = 'sparse_categorical_crossentropy' if self.categorical else 'binary_crossentropy'
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
    def _split_and_retrieve_data(self):
        raw_dataset = tf.data.TFRecordDataset(
            ["{}/{}".format(self.heatmaps_path, f.name) for f in os.scandir(self.heatmaps_path) if "tfrecord" in f.name], 
            num_parallel_reads=80)
        dataset = raw_dataset.map(lambda x: get_images(x, self.input_shape, self.categorical), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.apply(tf.data.experimental.ignore_errors())

        train_set, val_set, test_set = stratified_split(dataset, self.train_proportion, self.use_test_set)
        train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE).cache()
        val_set = val_set.prefetch(tf.data.experimental.AUTOTUNE).cache()
        test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE).cache() if self.use_test_set else None

        return extract_ids_and_batch(train_set, val_set, test_set, self.batch_size)

